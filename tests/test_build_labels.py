import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "train"))

import pytest
from transformers import AutoTokenizer
from chat_utils import build_labels, IGNORE_INDEX

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

MESSAGES = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Weather in Hanoi?"},
    {"role": "assistant", "content": "<think>Need weather.</think>",
     "tool_calls": [{"type": "function", "function": {
         "name": "get_weather", "arguments": {"city": "Hanoi"}}}]},
    {"role": "tool", "content": '{"temp": 28}'},
    {"role": "assistant", "content": "<think>Done.</think>It is 28C."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


def learned_text(tok, ids, labels):
    return tok.decode([i for i, l in zip(ids, labels) if l != IGNORE_INDEX])


def test_lengths_match(tok):
    ids, labels = build_labels(tok, MESSAGES, TOOLS)
    assert len(ids) == len(labels)


def test_assistant_learned(tok):
    ids, labels = build_labels(tok, MESSAGES, TOOLS)
    learned = learned_text(tok, ids, labels)
    assert "get_weather" in learned          # tool call do policy sinh
    assert "It is 28C." in learned
    assert "<|im_end|>" in learned


def test_non_assistant_masked(tok):
    ids, labels = build_labels(tok, MESSAGES, TOOLS)
    learned = learned_text(tok, ids, labels)
    assert "Weather in Hanoi?" not in learned   # user
    assert "You are a helpful agent." not in learned  # system
    assert '{"temp": 28}' not in learned         # tool output KHÔNG được học
    assert "Get weather of a city" not in learned  # tool schema
