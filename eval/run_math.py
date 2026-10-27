"""Nâng cấp eval GSM8K / MATH-500 cho cả Base Instruct và Reasoning Model (Phase 0 & Phase 1).

Các chỉ số đo:
- acc:            Tỉ lệ đáp án trong \\boxed{} khớp ground truth
- boxed_rate:     Tỉ lệ sinh được \\boxed{}
- truncated_rate: Tỉ lệ bị cắt do vượt max_tokens
- think_rate:     Tỉ lệ mở VÀ đóng thẻ <think>...</think> đúng chuẩn (metric sống còn ở P1)
- avg_tokens:     Độ dài trung bình của câu trả lời (theo dõi xem model có suy nghĩ dài hơn không)

Hỗ trợ:
- --thinking: dùng system prompt yêu cầu <think>...</think>
- --budget_forcing_wait: kỹ thuật từ paper s1 — nếu model đóng </think> quá sớm (< min_think_tokens),
  ép nối thêm "\\nWait, let me double check my calculation.\\n" và generate tiếp.
"""
import argparse, json, pathlib, re
from datasets import load_dataset
from vllm import LLM, SamplingParams

BASE_SYSTEM = (
    "Solve the problem step by step. "
    "Put the final numeric answer in \\boxed{}."
)

THINK_SYSTEM = (
    "Solve the problem step by step. "
    "First think carefully inside <think>...</think> tags, "
    "then provide the clear solution and put the final answer in \\boxed{}."
)

THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.S)


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth, buf = idx + len("\\boxed{"), 1, []
    while i < len(text) and depth:
        c = text[i]
        depth += (c == "{") - (c == "}")
        if depth:
            buf.append(c)
        i += 1
    return "".join(buf).strip() if depth == 0 else None


def norm(s):
    s = (s or "").replace(",", "").replace("$", "").replace("\\!", "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def generate_with_budget_forcing(llm, prompts, max_tokens: int, n_wait: int = 1):
    """Kỹ thuật s1 budget forcing: dừng tại </think>, nối thêm 'Wait' để ép suy nghĩ sâu hơn."""
    # Bước 1: generate tới khi gặp </think>
    sp_think = SamplingParams(temperature=0.0, max_tokens=max_tokens - 512, stop=["</think>"])
    outs1 = llm.generate(prompts, sp_think)

    forced_prompts = []
    for p, o in zip(prompts, outs1):
        partial = o.outputs[0].text
        if not partial.lstrip().startswith("<think>"):
            partial = "<think>\n" + partial
        for _ in range(n_wait):
            partial += "\nWait, let me double check my steps carefully.\n"
        forced_prompts.append(p + partial)

    # Bước 2: generate phần còn lại
    sp_rest = SamplingParams(temperature=0.0, max_tokens=1024)
    outs2 = llm.generate(forced_prompts, sp_rest)

    combined = []
    for o1, o2 in zip(outs1, outs2):
        text = o1.outputs[0].text + "\nWait, let me double check my steps carefully.\n" + o2.outputs[0].text
        n_tok = len(o1.outputs[0].token_ids) + len(o2.outputs[0].token_ids)
        trunc = o2.outputs[0].finish_reason == "length"
        combined.append((text, n_tok, trunc))
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["gsm8k", "math500"], default="gsm8k")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--thinking", action="store_true", help="Dùng system prompt có <think>...</think>")
    ap.add_argument("--budget_forcing_wait", type=int, default=0, help="Số lần ép 'Wait' trước khi đóng </think>")
    ap.add_argument("--save_samples", default=None, help="Lưu 10 mẫu đầu ra file jsonl để đọc tay")
    args = ap.parse_args()

    if args.task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.select(range(min(args.n, len(ds))))
        questions = ds["question"]
        golds = [gt.split("####")[-1].strip() for gt in ds["answer"]]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(args.n, len(ds))))
        questions = ds["problem"]
        golds = ds["answer"]

    sys_prompt = THINK_SYSTEM if args.thinking else BASE_SYSTEM
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_util, max_model_len=max(4096, args.max_tokens + 512))
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    if args.budget_forcing_wait > 0:
        results_raw = generate_with_budget_forcing(llm, prompts, args.max_tokens, args.budget_forcing_wait)
    else:
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        results_raw = [
            (o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason == "length")
            for o in outs
        ]

    correct = boxed = truncated = valid_think = total_tokens = 0
    sample_logs = []

    for q, (text, n_tok, is_trunc), gold_str in zip(questions, results_raw, golds):
        # Hanya tìm \boxed{} ở phần SAU </think> nếu có </think> để tránh lấy nhầm nháp trong <think>
        after_think = text.split("</think>")[-1] if "</think>" in text else text
        pred = extract_boxed(after_think)
        boxed += pred is not None
        truncated += is_trunc
        valid_think += bool(THINK_CLOSED_RE.search(text))
        total_tokens += n_tok

        if args.task == "gsm8k":
            is_ok = norm(pred) is not None and norm(pred) == norm(gold_str)
        else:
            is_ok = (pred is not None and pred.strip() == gold_str.strip()) or (
                norm(pred) is not None and norm(pred) == norm(gold_str)
            )
        correct += is_ok

        if len(sample_logs) < 10:
            sample_logs.append({
                "question": q,
                "gold": gold_str,
                "pred": pred,
                "correct": bool(is_ok),
                "n_tokens": n_tok,
                "has_closed_think": bool(THINK_CLOSED_RE.search(text)),
                "output": text,
            })

    n_eval = len(questions)
    res = {
        "model": args.model,
        "task": f"{args.task}[{n_eval}]" + ("+think" if args.thinking else ""),
        "acc": correct / n_eval,
        "boxed_rate": boxed / n_eval,
        "truncated_rate": truncated / n_eval,
        "think_rate": valid_think / n_eval,
        "avg_tokens": round(total_tokens / n_eval, 1),
    }
    print(json.dumps(res, indent=2))

    path = pathlib.Path("results/baseline.md")
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(
            f"| {res['model']} | {res['task']} | {res['acc']:.3f} | "
            f"{res['boxed_rate']:.3f} | {res['truncated_rate']:.3f} | "
            f"think={res['think_rate']:.2f}, tok={res['avg_tokens']} |\n"
        )

    if args.save_samples:
        sp = pathlib.Path(args.save_samples)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("w") as f:
            for s in sample_logs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
