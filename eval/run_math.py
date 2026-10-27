"""Baseline eval GSM8K bằng vLLM (Phase 0).

Chạy:  python eval/run_math.py --model Qwen/Qwen2.5-1.5B-Instruct --n 500
Kết quả ghi thêm vào results/baseline.md
"""
import argparse, json, pathlib, re
from datasets import load_dataset
from vllm import LLM, SamplingParams

SYSTEM = ("Solve the problem step by step. "
          "Put the final numeric answer in \\boxed{}.")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_util, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in ds["question"]]

    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    correct = boxed = truncated = 0
    for o, gt in zip(outs, ds["answer"]):
        text = o.outputs[0].text
        pred = extract_boxed(text)
        boxed += pred is not None
        truncated += o.outputs[0].finish_reason == "length"
        gold = norm(gt.split("####")[-1])
        correct += (norm(pred) is not None and norm(pred) == gold)

    res = {"model": args.model, "task": f"gsm8k[{args.n}]",
           "acc": correct / args.n, "boxed_rate": boxed / args.n,
           "truncated_rate": truncated / args.n}
    print(json.dumps(res, indent=2))
    path = pathlib.Path("results/baseline.md")
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(f"| {res['model']} | {res['task']} | {res['acc']:.3f} | "
                f"{res['boxed_rate']:.3f} | {res['truncated_rate']:.3f} |\n")


if __name__ == "__main__":
    main()
