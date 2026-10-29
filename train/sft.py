"""Huấn luyện SFT (Cold-Start Reasoning) với loss mask chuẩn (Phase 1).

Đặc điểm:
- Tái sử dụng đúng hàm `build_labels` từ `train/chat_utils.py` (Phase 0) qua custom DataCollator,
  đảm bảo CHỈ tính loss trên phần assistant (`<think>...</think>\n\n<solution><|im_end|>`).
- Không thêm `<think>` và `</think>` làm special tokens mới vào vocab (vì với LoRA trên model
  1.5B, thêm token mới phải mở `embed_tokens` + `lm_head` tốn VRAM và dễ làm lệch phân phối
  khởi đầu). Thay vào đó dùng chuỗi ký tự `<think>` có sẵn trong vocab Qwen2.5.
- Hỗ trợ cả Unsloth (nếu cài) và chuẩn Transformers + PEFT + TRL.

Chạy thử trên T4 (0.5B debug):
  python train/sft.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data/processed/reasoning_sft.jsonl --output_dir checkpoints/qwen0.5b-sft-think \
    --max_seq_len 2048 --epochs 1 --batch_size 2 --grad_accum 4

Chạy chính thức trên L4/A100 (1.5B):
  python train/sft.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/processed/reasoning_sft.jsonl --output_dir checkpoints/qwen1.5b-sft-think \
    --max_seq_len 4096 --epochs 3 --batch_size 2 --grad_accum 8 --lr 1e-4
"""
import argparse, math, os, pathlib, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback, PrinterCallback
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from chat_utils import build_labels, IGNORE_INDEX


def log(msg: str, log_file: str | None = None):
    """print + flush ngay (Kaggle/Colab/torchrun không phải TTY → stdout bị buffer)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


class ProgressLogger(TrainerCallback):
    """Thay tqdm bằng log từng dòng: step/total, loss, lr, grad_norm, tốc độ, ETA, VRAM."""

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        self.t0 = None

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        if state.is_world_process_zero:
            log(f"Bắt đầu train: max_steps={state.max_steps}", self.log_file)

    def on_log(self, args, state, control, logs=None, **kw):
        if not state.is_world_process_zero or not logs or "loss" not in logs:
            return
        step, total = state.global_step, max(state.max_steps, 1)
        elapsed = time.time() - self.t0
        sec_per_step = elapsed / max(step, 1)
        eta_min = sec_per_step * (total - step) / 60
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        log(
            f"step {step}/{total} ({100 * step / total:.1f}%) | "
            f"loss {logs['loss']:.4f} | lr {logs.get('learning_rate', 0):.2e} | "
            f"grad_norm {logs.get('grad_norm', float('nan')):.3f} | "
            f"{sec_per_step:.1f}s/step | ETA {eta_min:.0f} min | peak VRAM {mem:.1f}GB",
            self.log_file,
        )

    def on_save(self, args, state, control, **kw):
        if state.is_world_process_zero:
            log(f"Checkpoint saved at step {state.global_step}", self.log_file)

    def on_train_end(self, args, state, control, **kw):
        if state.is_world_process_zero:
            log(f"Train xong sau {(time.time() - self.t0) / 60:.1f} phút", self.log_file)


class AssistantOnlyCollator:
    """Pad batch của (input_ids, labels) đã qua build_labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(labs + [IGNORE_INDEX] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_and_mask(dataset, tokenizer, max_seq_len: int):
    def _proc(row):
        ids, labs = build_labels(tokenizer, row["messages"], tools=row.get("tools"))
        return {"input_ids": ids[:max_seq_len], "labels": labs[:max_seq_len]}

    ds = dataset.map(_proc, remove_columns=dataset.column_names)
    # Loại bỏ mẫu bị cắt cụt hết phần assistant (toàn -100)
    ds = ds.filter(lambda x: any(l != IGNORE_INDEX for l in x["labels"]))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/s1k_filtered.jsonl")
    ap.add_argument("--output_dir", default="checkpoints/qwen1.5b-sft-think")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--hub_model_id", default=None, help="VD: username/qwen1.5b-sft-think")
    ap.add_argument("--merge_and_save", action="store_true", help="Merge LoRA vào base để chạy vLLM")
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--log_file", default=None, help="Ghi thêm log ra file, VD logs/sft.log")
    ap.add_argument("--tqdm", action="store_true", help="Bật thanh tqdm (chỉ hữu ích khi chạy trong terminal)")
    args = ap.parse_args()
    if args.log_file:
        pathlib.Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    raw_ds = load_dataset("json", data_files=args.data, split="train")
    train_ds = tokenize_and_mask(raw_ds, tok, args.max_seq_len)
    print(f"Loaded {len(train_ds)} masked samples (max_seq_len={args.max_seq_len})")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16  # T4 (sm75) không có bf16 → fp16

    # Multi-GPU: chạy bằng `torchrun --nproc_per_node N` → DDP, mỗi process giữ 1 bản model
    # trên đúng GPU của nó. device_map="auto" sẽ chia layer qua các GPU (chạy tuần tự, không
    # nhanh hơn) và xung đột với DDP, nên chỉ dùng khi chạy 1 process.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank} if world_size > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    # fp16 AMP yêu cầu tham số trainable ở fp32 (nếu không: "Attempting to unscale FP16 gradients").
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    # transformers v5 bỏ `warmup_ratio` → tự tính warmup_steps (int) để tương thích cả v4/v5.
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum * world_size))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.05 * total_steps))
    print(f"world_size={world_size}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=bf16_ok,
        fp16=not bf16_ok and torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,  # bắt buộc với LoRA + grad checkpointing trong DDP
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=not args.tqdm,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=AssistantOnlyCollator(tok.pad_token_id),
    )
    # Khi tắt tqdm, Trainer tự thêm PrinterCallback in dict thô → bỏ đi, dùng ProgressLogger.
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(ProgressLogger(args.log_file))
    trainer.train()

    trainer.save_model(args.output_dir)
    if not trainer.is_world_process_zero():
        return  # chỉ rank 0 lưu tokenizer & merge
    tok.save_pretrained(args.output_dir)

    if args.merge_and_save:
        merged_dir = f"{args.output_dir}-merged"
        print(f"Merging LoRA into base model -> {merged_dir}")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)


if __name__ == "__main__":
    main()
