import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "train"))

from build_reasoning_sft import format_assistant, extract_s1k, THINK_BLOCK_RE
from sft import AssistantOnlyCollator
from chat_utils import IGNORE_INDEX


def test_format_assistant_structure():
    out = format_assistant("step 1 -> step 2", "Final answer is \\boxed{42}.")
    m = THINK_BLOCK_RE.match(out)
    assert m is not None
    assert m.group(1).strip() == "step 1 -> step 2"
    assert "\\boxed{42}" in m.group(2)


def test_extract_s1k():
    row = {
        "question": "What is 2+2?",
        "deepseek_thinking_trajectory": "Let me calculate 2+2=4. So \\boxed{4}.",
        "solution": "The answer is 4.",
    }
    q, think, sol = extract_s1k(row)
    assert q == "What is 2+2?"
    assert "2+2=4" in think
    assert "\\boxed{4}" in sol



def test_collator_padding():
    collator = AssistantOnlyCollator(pad_token_id=0)
    batch = collator([
        {"input_ids": [10, 20, 30], "labels": [IGNORE_INDEX, 20, 30]},
        {"input_ids": [10, 20], "labels": [IGNORE_INDEX, 20]},
    ])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 20, 30], [IGNORE_INDEX, 20, IGNORE_INDEX]]
