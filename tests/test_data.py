import pytest

from ntkmirror.data import Example, batches, ensure_pad_token, load_jsonl_examples


def test_batches():
    xs = [Example("p", " c") for _ in range(5)]
    assert [len(b) for b in batches(xs, 2)] == [2, 2, 1]


class _NoPadNoEosTokenizer:
    pad_token_id = None
    eos_token_id = None


def test_ensure_pad_token_refuses_to_add_new_token_without_resize():
    with pytest.raises(ValueError, match="does not add new special tokens implicitly"):
        ensure_pad_token(_NoPadNoEosTokenizer())


def test_load_jsonl_examples_accepts_chat_messages_without_template(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        '{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}\n',
        encoding="utf-8",
    )
    examples = load_jsonl_examples(path, chat_template="none", loss_on="assistant")
    assert len(examples) == 1
    assert "user: Hi" in examples[0].prompt
    assert examples[0].completion == " Hello"
