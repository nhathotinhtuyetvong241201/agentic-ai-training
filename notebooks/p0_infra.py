# Phase 0 — Infra, Chat Template & Baseline
# Định dạng "percent": mỗi "# %%" là một cell. Dán vào Colab hoặc dùng
# File > Upload notebook sau khi convert:  jupytext --to ipynb p0_infra.py

# %% [markdown]
# ## 1. Cài đặt (pin version khi đã chạy ổn, ghi lại bằng `pip freeze`)
# %%
# !pip install -q "transformers" "datasets" "trl" "peft" "wandb" "pytest" "huggingface_hub"
# !pip install -q vllm   # có thể cần restart runtime sau bước này

# %% [markdown]
# ## 2. Clone repo / mount Drive để giữ checkpoint
# %%
# from google.colab import drive; drive.mount('/content/drive')
# !git clone <repo-url> /content/agentic-ai-training  (hoặc upload thủ công)
# %cd /content/agentic-ai-training

# %% [markdown]
# ## 3. Đọc từng token của prompt có tool
# %%
import sys; sys.path.insert(0, "train")
from transformers import AutoTokenizer
from chat_utils import build_labels, show_mask

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather of a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
messages = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Weather in Hanoi?"},
    {"role": "assistant", "content": "<think>Need weather.</think>",
     "tool_calls": [{"type": "function", "function": {
         "name": "get_weather", "arguments": {"city": "Hanoi"}}}]},
    {"role": "tool", "content": '{"temp": 28}'},
    {"role": "assistant", "content": "<think>Done.</think>It is 28C."},
]
print(tok.apply_chat_template(messages, tools=tools, tokenize=False))

# %% [markdown]
# ## 4. Kiểm tra loss mask: chỉ phần trong [[ ]] được học
# %%
ids, labels = build_labels(tok, messages, tools)
print(show_mask(tok, ids, labels))

# %% [markdown]
# **Câu hỏi tự kiểm tra** (ghi đáp án vào results/experiment_log.md):
# 1. Template Qwen2.5 render `tool_calls` thành dạng gì? Nằm trong thẻ nào?
# 2. Nhiều tool message liên tiếp được gộp ra sao? Vì sao điều đó phá cách tokenize tăng dần?
# 3. Template có xoá `<think>` ở các turn cũ không? (thử với Qwen3 để thấy khác biệt)
# 4. `<|im_end|>` có nằm trong vùng học không? Vì sao cần?

# %% [markdown]
# ## 5. Unit test
# %%
# !python -m pytest tests -q

# %% [markdown]
# ## 6. Baseline eval (GSM8K) — chạy 0.5B trước để debug, sau đó 1.5B/3B
# %%
# !python eval/run_math.py --model Qwen/Qwen2.5-0.5B-Instruct --n 200
# !python eval/run_math.py --model Qwen/Qwen2.5-1.5B-Instruct --n 500
