# Phase 1 — Reasoning SFT / Cold-Start Distillation
# Định dạng "percent": mỗi "# %%" là một cell trên Colab.

# %% [markdown]
# ## Mục tiêu Phase 1
# - Dạy `Qwen2.5-1.5B-Instruct` thói quen suy nghĩ sâu trong thẻ `<think>...</think>` trước khi kết luận.
# - Hiểu rõ 2 bệnh kinh điển khi SFT reasoning trên model nhỏ (0.5B–3B):
#   1. **Unclosed `<think>`**: trace quá dài khi train khiến model sinh lan man hết `max_tokens` mà không đóng `</think>`.
#   2. **Math accuracy drop tạm thời trên câu dễ (GSM8K)** nếu ép trace quá dài (overthinking), trong khi câu khó (MATH-500) lại tăng.
#
# > **Nguyên tắc tiết kiệm Compute Units trên Colab:**
# > Luôn chạy thử bước 1–3 với `Qwen2.5-0.5B-Instruct` (100 mẫu, 1 epoch, ~3 phút trên T4) trước khi bật L4/A100 cho `1.5B`.

# %% [markdown]
# ## 0. Sửa môi trường Colab
# Colab cài sẵn `torchao` bản cũ; PEFT mới kiểm tra version torchao khi wrap LoRA và báo
# `ImportError: Found an incompatible version of torchao`. Pipeline này không dùng torchao
# (không quantization FP8/int4) → gỡ đi. Sau đó Runtime > Restart session.
# %%
# !pip uninstall -y torchao
# !python -c "import peft, transformers, torch; print(peft.__version__, transformers.__version__, torch.__version__)"

# %% [markdown]
# ## 1. Chuẩn bị dữ liệu Reasoning Cold-Start (`s1K-1.1` + `LIMO` + `OpenR1` streaming)
# Lọc chỉ giữ các mẫu có độ dài $\le 4096$ tokens và có `\boxed{}` để model 1.5B không bị ngợp bởi các trace 16k tokens.
# - Dùng `--source s1k` để xem số mẫu thu được từ riêng `s1K-1.1` (sau khi cứu `\boxed{}` từ `deepseek_attempt`).
# - Dùng `--source mix` (mặc định) để lấy hết các mẫu $\le 4096$ của `s1K` + `LIMO` và bù thêm từ `OpenR1-Math-220k` (streaming, không tốn đĩa) cho đủ đúng 1,000 mẫu chất lượng cao.
# %%
# !python data/build_reasoning_sft.py \
#   --source mix \
#   --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
#   --max_tokens 4096 \
#   --max_samples 1000 \
#   --out data/s1k_filtered.jsonl

# %% [markdown]
# ## 2. Kiểm tra bằng mắt 1 mẫu dữ liệu & vùng tính Loss (`show_mask`)
# Đảm bảo `<think>...</think>` và câu trả lời nằm trong `[[ ... ]]`, còn system/user nằm ngoài.
# %%
import json, sys
sys.path.insert(0, "train")
from transformers import AutoTokenizer
from chat_utils import build_labels, show_mask

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
with open("data/s1k_filtered.jsonl") as f:
    sample = json.loads(f.readline())

ids, labs = build_labels(tok, sample["messages"])
print(f"Total tokens: {len(ids)} | Learned tokens: {sum(l != -100 for l in labs)}")
print(show_mask(tok, ids[:400], labs[:400]))

# %% [markdown]
# ## 3. Smoke Test trên T4 với `Qwen2.5-0.5B-Instruct` (5 phút)
# Kiểm tra toàn bộ pipeline (build_labels -> AssistantOnlyCollator -> LoRA -> merge) không lỗi OOM.
# %%
# !python train/sft.py \
#   --model Qwen/Qwen2.5-0.5B-Instruct \
#   --data data/s1k_filtered.jsonl \
#   --output_dir checkpoints/qwen0.5b-sft-think \
#   --max_seq_len 2048 \
#   --epochs 1 \
#   --batch_size 2 \
#   --grad_accum 4 \
#   --merge_and_save

# %% [markdown]
# ## 4. Train chính thức `Qwen2.5-1.5B-Instruct` (L4 hoặc A100, ~25–40 phút)
# Lưu checkpoint mỗi 50 steps. Nếu có HF token, thêm `--hub_model_id <user>/qwen1.5b-sft-think` để tự động backup.
# %%
# !python train/sft.py \
#   --model Qwen/Qwen2.5-1.5B-Instruct \
#   --data data/s1k_filtered.jsonl \
#   --output_dir checkpoints/qwen1.5b-sft-think \
#   --max_seq_len 4096 \
#   --epochs 3 \
#   --batch_size 2 \
#   --grad_accum 8 \
#   --lr 1e-4 \
#   --merge_and_save

# %% [markdown]
# ## 5. Đánh giá sau SFT: GSM8K & MATH-500 + Kiểm tra `think_rate`
# So sánh trực tiếp với baseline trong `results/baseline.md` và xuất 10 rollout ra file để đọc tay.
# %%
# Baseline MATH-500 của model gốc (chưa có trong P0):
# !python eval/run_math.py --model Qwen/Qwen2.5-1.5B-Instruct --task math500 --n 200

# Eval model sau SFT (bật --thinking):
# !python eval/run_math.py \
#   --model checkpoints/qwen1.5b-sft-think-merged \
#   --task gsm8k --n 500 --max_tokens 4096 --thinking \
#   --save_samples results/samples_p1_gsm8k.jsonl

# !python eval/run_math.py \
#   --model checkpoints/qwen1.5b-sft-think-merged \
#   --task math500 --n 200 --max_tokens 4096 --thinking \
#   --save_samples results/samples_p1_math500.jsonl

# %% [markdown]
# ## 6. Thí nghiệm Test-Time Scaling (Budget Forcing từ paper s1)
# Thử ép model "Wait, let me double check..." 1 lần trước khi đóng `</think>` xem accuracy có tăng không.
# %%
# !python eval/run_math.py \
#   --model checkpoints/qwen1.5b-sft-think-merged \
#   --task math500 --n 200 --max_tokens 4096 --thinking \
#   --budget_forcing_wait 1
