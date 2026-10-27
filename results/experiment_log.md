# Experiment Log

Mẫu mỗi entry: **Giả thuyết → Thiết lập → Kết quả → Đọc rollouts → Bài học**

## P0 — Infra & Baseline

- [ ] Trả lời 4 câu hỏi chat template (xem `notebooks/p0_infra.py`)
- [x] `pytest tests` pass
- [x] Baseline GSM8K 0.5B (`acc=0.505`) / 1.5B (`acc=0.756`, `boxed=0.986`, `trunc=0.008`) — xem `baseline.md`

## P1 — Reasoning SFT (Cold-Start Distillation)

- **Giả thuyết**: SFT `Qwen2.5-1.5B-Instruct` trên < 1,000 mẫu reasoning chất lượng cao (`s1K-1.1`, lọc $\le 4096$ tokens) sẽ dạy model sinh `<think>...</think>` ổn định (`think_rate > 95%`) và tăng điểm trên bài toán khó (`MATH-500`), làm bàn đạp cold-start cho GRPO ở Phase 2.
- [ ] Chạy `data/build_reasoning_sft.py` tạo tập `data/processed/s1k_filtered.jsonl`
- [ ] Smoke test 0.5B trên T4 & train 1.5B (3 epochs) bằng `train/sft.py`
- [ ] Đo baseline `MATH-500` cho 1.5B gốc và so sánh với `qwen1.5b-sft-think-merged` trên cả `GSM8K` & `MATH-500`
- [ ] Đọc tay 10 mẫu trong `results/samples_p1_math500.jsonl`: model có thực sự tự kiểm tra ("Wait...", backtrack) hay chỉ bắt chước văn phong?
- [ ] Thử `--budget_forcing_wait 1` và ghi nhận kết quả

