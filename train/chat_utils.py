"""Chat template + loss-mask utilities (Phase 0).

Cách làm: render TOÀN BỘ hội thoại một lần bằng chat template, sau đó dùng
offset mapping của tokenizer để đánh dấu token thuộc các block assistant.
Cách này không phụ thuộc vào việc template có "prefix-consistent" hay không
(Qwen2.5 gộp các tool message liên tiếp vào cùng một turn user và chỉ đóng
thẻ ở message cuối, nên cách tokenize tăng dần sẽ sai).

Chỉ hỗ trợ ChatML (Qwen2.5/Qwen3). Với template khác, chỉnh ASSISTANT_RE.
"""
import re

IGNORE_INDEX = -100

# Phần assistant sinh ra: sau "<|im_start|>assistant\n" đến hết "<|im_end|>".
# <|im_end|> được tính loss để model học cách kết thúc turn.
ASSISTANT_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S
)


def build_labels(tokenizer, messages, tools=None):
    """Trả về (input_ids, labels). labels = -100 cho mọi token không phải assistant."""
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    spans = [m.span(1) for m in ASSISTANT_RE.finditer(text)]
    labels = []
    for tok_id, (start, end) in zip(input_ids, offsets):
        in_assistant = any(s <= start and end <= e for s, e in spans)
        labels.append(tok_id if in_assistant else IGNORE_INDEX)
    return input_ids, labels


def show_mask(tokenizer, input_ids, labels):
    """In ra đoạn text được học (giữa [[ ]]) để kiểm tra bằng mắt."""
    out, learning = [], False
    for tok_id, lab in zip(input_ids, labels):
        is_l = lab != IGNORE_INDEX
        if is_l and not learning:
            out.append("[[")
        if not is_l and learning:
            out.append("]]")
        learning = is_l
        out.append(tokenizer.decode([tok_id]))
    if learning:
        out.append("]]")
    return "".join(out)
