"""Xây dựng dataset SFT cho Reasoning Cold-Start (Phase 1).

Chuẩn hoá các dataset reasoning (simplescaling/s1K-1.1, GAIR/LIMO, open-r1/OpenR1-Math-220k)
về định dạng ChatML chuẩn với block <think>...</think> rõ ràng:

  system: Solve the problem step by step. Enclose your internal reasoning in <think>...</think> and put the final answer in \boxed{}.
  user:   <question>
  assistant: <think>\n<thinking_trace>\n</think>\n\n<final_solution>

Quan trọng cho model nhỏ (0.5B - 3B):
- Trace quá dài (> 6000 tokens) trên model nhỏ rất dễ gây hiện tượng lặp vô hạn
  (degeneration loop) hoặc không bao giờ đóng thẻ </think> khi inference.
- Script này hỗ trợ lọc theo độ dài token (--max_tokens) và lọc mẫu có \boxed{}.
"""
import argparse, json, pathlib, re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

REASONING_SYSTEM = (
    "Solve the problem step by step. "
    "First think carefully inside <think>...</think> tags, "
    "then provide the clear solution and put the final answer in \\boxed{}."
)

THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.+)$", re.S)


def extract_s1k(row):
    """simplescaling/s1K-1.1 hoặc s1K: có cột question, solution, thinking_trajectories."""
    q = row.get("question") or row.get("problem") or ""
    sol = row.get("solution") or row.get("attempt") or ""
    # s1K-1.1 lưu thinking trong deepseek_thinking_trajectory hoặc gemini_thinking_trajectory hoặc thinking_trajectories
    think = (
        row.get("deepseek_thinking_trajectory")
        or row.get("gemini_thinking_trajectory")
        or row.get("thinking_trajectories")
        or ""
    )
    if isinstance(think, list):
        think = think[0] if think else ""
    return q.strip(), str(think).strip(), str(sol).strip()


def extract_limo(row):
    """GAIR/LIMO: question, solution (solution thường là long CoT hoàn chỉnh)."""
    q = row.get("question") or ""
    sol = row.get("solution") or ""
    m = THINK_BLOCK_RE.match(sol)
    if m:
        return q.strip(), m.group(1).strip(), m.group(2).strip()
    # Nếu LIMO chưa tách <think>, tìm đoạn cuối chứa \boxed{} làm câu trả lời tóm tắt
    # hoặc đặt toàn bộ quá trình giải vào <think> và lặp lại kết luận cuối ở ngoài
    idx = sol.rfind("\\boxed{")
    if idx != -1:
        last_para_start = sol.rfind("\n\n", 0, idx)
        final_part = sol[last_para_start:].strip() if last_para_start != -1 else sol[idx:].strip()
        return q.strip(), sol.strip(), final_part
    return q.strip(), sol.strip(), ""


def extract_openr1(row):
    """open-r1/OpenR1-Math-220k: có messages hoặc problem + generations."""
    if "messages" in row and isinstance(row["messages"], list):
        q, assistant = "", ""
        for m in row["messages"]:
            if m.get("role") == "user":
                q = m.get("content", "")
            elif m.get("role") == "assistant":
                assistant = m.get("content", "")
        m = THINK_BLOCK_RE.match(assistant)
        if m:
            return q.strip(), m.group(1).strip(), m.group(2).strip()
    return "", "", ""


def format_assistant(think: str, solution: str) -> str:
    think = think.replace("<think>", "").replace("</think>", "").strip()
    solution = solution.replace("<think>", "").replace("</think>", "").strip()
    return f"<think>\n{think}\n</think>\n\n{solution}"


def build_dataset(
    source: str,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_tokens: int = 4096,
    min_think_tokens: int = 64,
    max_samples: int = 2000,
    require_boxed: bool = True,
):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    if source == "s1k":
        ds = load_dataset("simplescaling/s1K-1.1", split="train")
        extractor = extract_s1k
    elif source == "limo":
        ds = load_dataset("GAIR/LIMO", split="train")
        extractor = extract_limo
    elif source == "openr1":
        ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
        extractor = extract_openr1
    else:
        raise ValueError(f"Unknown source: {source}")

    records = []
    stats = {"total": 0, "empty": 0, "no_boxed": 0, "too_long": 0, "too_short": 0, "kept": 0}

    for row in ds:
        stats["total"] += 1
        q, think, sol = extractor(row)
        if not q or not think or not sol:
            stats["empty"] += 1
            continue
        if require_boxed and "\\boxed{" not in sol:
            stats["no_boxed"] += 1
            continue

        assistant_text = format_assistant(think, sol)
        messages = [
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_text},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        n_tok = len(tok(rendered, add_special_tokens=False)["input_ids"])
        n_think = len(tok(think, add_special_tokens=False)["input_ids"])

        if n_think < min_think_tokens:
            stats["too_short"] += 1
            continue
        if n_tok > max_tokens:
            stats["too_long"] += 1
            continue

        records.append({
            "messages": messages,
            "n_tokens": n_tok,
            "n_think_tokens": n_think,
            "source": source,
        })
        stats["kept"] += 1
        if len(records) >= max_samples:
            break

    print(json.dumps(stats, indent=2))
    return Dataset.from_list(records), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["s1k", "limo", "openr1"], default="s1k")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--out", default="data/processed/reasoning_sft.jsonl")
    args = ap.parse_args()

    ds, _ = build_dataset(
        source=args.source,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
    )
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(out_path), force_ascii=False)
    print(f"Saved {len(ds)} samples to {out_path}")


if __name__ == "__main__":
    main()
