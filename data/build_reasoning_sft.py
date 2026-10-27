"""Xây dựng dataset SFT cho Reasoning Cold-Start (Phase 1).

Chuẩn hoá các dataset reasoning (simplescaling/s1K-1.1, GAIR/LIMO, open-r1/OpenR1-Math-220k)
về định dạng ChatML chuẩn với block <think>...</think> rõ ràng:

  system: Solve the problem step by step. First think carefully inside <think>...</think> tags...
  user:   <question>
  assistant: <think>\n<thinking_trace>\n</think>\n\n<final_solution>

Bài học thực tế từ dữ liệu s1K-1.1:
1. Cột `solution` là lời giải tham khảo gốc (đa số viết dạng "The answer is X" hoặc câu văn,
   KHÔNG có `\boxed{}`). Trong khi đó cột `deepseek_attempt` / `gemini_attempt` (lời giải do
   model teacher sinh ra sau khi think) hoặc đoạn cuối của `deepseek_thinking_trajectory` mới
   chứa `\boxed{}`. Nếu chỉ tìm `\boxed{}` trong `solution` sẽ mất ~70% dữ liệu (`no_boxed: 705`).
2. Các mẫu còn lại của s1K-1.1 được chọn theo tiêu chí "khó + trace dài" (trung bình 5k–16k tokens),
   nên nếu cắt ở 4096 tokens với s1K sẽ mất thêm hầu hết mẫu (`too_long: 284`).
3. Giải pháp chuẩn cho Colab (1.5B–3B, max_seq_len=4096..8192):
   - Với `s1k`: ưu tiên lấy `deepseek_attempt` (nếu có `\boxed{}`), hoặc tự động trích đáp án
     cuối bọc vào `\boxed{}` nếu cần; đồng thời cho phép nâng `--max_tokens` (VD 8192) hoặc
     kết hợp thêm `openr1` / `limo` nơi có rất nhiều trace chất lượng cao ở mức 1500–4096 tokens.
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
ANSWER_IS_RE = re.compile(
    r"(?:final answer is|answer is|therefore,?\s*the answer is)\s*[:$]?\s*([^\.\n$]+)",
    re.I,
)


def ensure_boxed(sol: str, fallback_text: str = "") -> str:
    """Nếu `sol` đã có \\boxed{} thì giữ nguyên; nếu chưa có thì thử tìm trong fallback_text
    (ví dụ deepseek_attempt hoặc cuối thinking trace) để gắn thêm \\boxed{} vào cuối lời giải."""
    if "\\boxed{" in sol:
        return sol
    # 1. Tìm \boxed{...} trong fallback_text (lấy cái cuối cùng)
    idx = fallback_text.rfind("\\boxed{")
    if idx >= 0:
        i, depth, buf = idx + len("\\boxed{"), 1, []
        while i < len(fallback_text) and depth:
            c = fallback_text[i]
            depth += (c == "{") - (c == "}")
            if depth:
                buf.append(c)
            i += 1
        if depth == 0 and buf:
            ans = "".join(buf).strip()
            return f"{sol.rstrip()}\n\nTherefore, the final answer is $\\boxed{{{ans}}}$."
    # 2. Tìm mẫu "The final answer is X" trong sol
    m = ANSWER_IS_RE.search(sol)
    if m:
        ans = m.group(1).strip()
        if len(ans) <= 40:
            return f"{sol.rstrip()}\n\n$\\boxed{{{ans}}}$"
    return sol


def extract_s1k(row):
    """simplescaling/s1K-1.1:
    - question
    - deepseek_thinking_trajectory / gemini_thinking_trajectory / thinking_trajectories
    - deepseek_attempt / gemini_attempt / attempt / solution
    """
    q = row.get("question") or row.get("problem") or ""
    think = (
        row.get("deepseek_thinking_trajectory")
        or row.get("gemini_thinking_trajectory")
        or row.get("thinking_trajectories")
        or ""
    )
    if isinstance(think, list):
        think = think[0] if think else ""
    think = str(think).strip()

    # Ưu tiên deepseek_attempt vì cùng style với deepseek_thinking_trajectory và thường có \boxed{}
    attempt = (
        row.get("deepseek_attempt")
        or row.get("gemini_attempt")
        or row.get("attempt")
        or ""
    )
    if isinstance(attempt, list):
        attempt = attempt[0] if attempt else ""
    attempt = str(attempt).strip()

    # Nếu attempt có chứa cả <think>...</think> thì tách phần sau </think>
    if "</think>" in attempt:
        attempt = attempt.split("</think>", 1)[-1].strip()

    sol_ref = str(row.get("solution") or "").strip()
    sol = attempt if ("\\boxed{" in attempt or not sol_ref) else sol_ref
    sol = ensure_boxed(sol, fallback_text=f"{attempt}\n{think[-1000:]}\n{sol_ref}")
    return q.strip(), think, sol.strip()


def extract_limo(row):
    """GAIR/LIMO: question, solution (long CoT hoàn chỉnh, kết thúc bằng \\boxed{})."""
    q = row.get("question") or ""
    sol = str(row.get("solution") or "").strip()
    m = THINK_BLOCK_RE.match(sol)
    if m:
        return q.strip(), m.group(1).strip(), m.group(2).strip()
    idx = sol.rfind("\\boxed{")
    if idx != -1:
        last_para_start = sol.rfind("\n\n", 0, idx)
        if last_para_start != -1 and last_para_start > len(sol) // 3:
            think_part = sol[:last_para_start].strip()
            final_part = sol[last_para_start:].strip()
            return q.strip(), think_part, final_part
        final_part = sol[idx:].strip()
        return q.strip(), sol, f"Therefore, the final answer is {final_part}"
    return q.strip(), sol, ""


def extract_openr1(row):
    """open-r1/OpenR1-Math-220k: có messages hoặc problem + solution + generations."""
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
    q = str(row.get("problem") or "").strip()
    gens = row.get("generations") or []
    if q and gens:
        m = THINK_BLOCK_RE.match(str(gens[0]))
        if m:
            return q, m.group(1).strip(), m.group(2).strip()
    return "", "", ""


def format_assistant(think: str, solution: str) -> str:
    think = think.replace("<think>", "").replace("</think>", "").strip()
    solution = solution.replace("<think>", "").replace("</think>", "").strip()
    return f"<think>\n{think}\n</think>\n\n{solution}"


def iter_source(source: str):
    if source == "s1k":
        ds = load_dataset("simplescaling/s1K-1.1", split="train")
        for r in ds:
            yield "s1k", extract_s1k(r)
    elif source == "limo":
        ds = load_dataset("GAIR/LIMO", split="train")
        for r in ds:
            yield "limo", extract_limo(r)
    elif source == "openr1":
        # Streaming để không phải tải toàn bộ 220k mẫu trên Colab
        ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)
        for r in ds:
            yield "openr1", extract_openr1(r)
    elif source == "mix":
        # Kết hợp s1k + limo + bổ sung từ openr1 (streaming) cho đủ max_samples ở ngưỡng max_tokens
        for sub in ("s1k", "limo", "openr1"):
            yield from iter_source(sub)
    else:
        raise ValueError(f"Unknown source: {source}")


def build_dataset(
    source: str,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_tokens: int = 4096,
    min_think_tokens: int = 64,
    max_samples: int = 1000,
    require_boxed: bool = True,
):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    records = []
    stats = {
        "total": 0,
        "empty": 0,
        "no_boxed": 0,
        "too_long": 0,
        "too_short": 0,
        "kept": 0,
        "by_source": {},
    }

    for src_name, (q, think, sol) in iter_source(source):
        stats["total"] += 1
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
            "source": src_name,
        })
        stats["kept"] += 1
        stats["by_source"][src_name] = stats["by_source"].get(src_name, 0) + 1
        if len(records) >= max_samples:
            break

    print(json.dumps(stats, indent=2))
    return Dataset.from_list(records), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["s1k", "limo", "openr1", "mix"], default="mix")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--no_require_boxed", action="store_true")
    ap.add_argument("--out", default="data/processed/s1k_filtered.jsonl")
    args = ap.parse_args()

    ds, _ = build_dataset(
        source=args.source,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
        require_boxed=not args.no_require_boxed,
    )
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(out_path), force_ascii=False)
    print(f"Saved {len(ds)} samples to {out_path}")


if __name__ == "__main__":
    main()
