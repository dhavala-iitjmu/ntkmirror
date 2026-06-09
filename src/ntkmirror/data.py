from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class Example:
    """One supervised causal-LM example.

    The prompt is context. The completion is teacher-forced and contributes to
    the loss. JSONL rows should usually look like:

        {"prompt": "Question...\nAnswer:", "completion": " worked solution..."}

    A row with only {"text": ...} is also accepted; all tokens after the
    first one are supervised. Chat rows with {"messages": [...]} are accepted
    by :func:`load_jsonl_examples`; by default they train only on the final
    assistant turn and use the tokenizer chat template when one is available.
    """

    prompt: str
    completion: str


_ROLE_ORDER = {"system", "user", "assistant", "tool"}


def _require_str(row: Mapping[str, Any], key: str, *, path: Path, line: int) -> str:
    if key not in row:
        raise ValueError(f"{path}:{line} missing required key {key!r}")
    value = row[key]
    if not isinstance(value, str):
        raise ValueError(f"{path}:{line} key {key!r} must be a string")
    return value


def _normalise_messages(row: Mapping[str, Any], *, path: Path, line: int) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}:{line} messages must be a non-empty list")
    out: list[dict[str, str]] = []
    for i, msg in enumerate(raw):
        if not isinstance(msg, Mapping):
            raise ValueError(f"{path}:{line} messages[{i}] must be an object")
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in _ROLE_ORDER:
            raise ValueError(f"{path}:{line} messages[{i}].role must be one of {sorted(_ROLE_ORDER)}")
        if not isinstance(content, str):
            raise ValueError(f"{path}:{line} messages[{i}].content must be a string")
        out.append({"role": role, "content": content})
    return out


def _render_messages_plain(messages: Sequence[Mapping[str, str]]) -> str:
    """Dependency-free fallback chat serialization.

    This is intentionally explicit rather than pretending to match a model's
    native chat template. It is used only when the user opts out of templates or
    no tokenizer/template is available.
    """

    return "".join(f"{m['role']}: {m['content']}\n" for m in messages)


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and callable(getattr(tokenizer, "apply_chat_template", None))


def chat_messages_to_example(
    messages: Sequence[Mapping[str, str]],
    *,
    tokenizer=None,
    chat_template: str = "auto",
    loss_on: str = "assistant",
) -> Example:
    """Convert a chat row into an :class:`Example`.

    Parameters
    ----------
    messages:
        OpenAI/HF-style messages with role/content keys.
    tokenizer:
        Optional tokenizer. When ``chat_template='auto'`` and the tokenizer has
        ``apply_chat_template``, the prompt side is rendered with the model's
        native template.
    chat_template:
        ``'auto'`` uses tokenizer.apply_chat_template when available. ``'none'``
        uses a simple role-prefixed fallback serialization.
    loss_on:
        ``'assistant'`` trains only on the final assistant message. ``'all'``
        trains on the complete rendered conversation.
    """

    if chat_template not in {"auto", "none"}:
        raise ValueError("chat_template must be 'auto' or 'none'")
    if loss_on not in {"assistant", "all"}:
        raise ValueError("loss_on must be 'assistant' or 'all'")
    msgs = [dict(m) for m in messages]
    if not msgs:
        raise ValueError("messages must be non-empty")

    if loss_on == "all":
        if chat_template == "auto" and tokenizer is not None and _has_chat_template(tokenizer):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            text = _render_messages_plain(msgs)
        return Example("", str(text))

    # Train on the final assistant turn only. This is the safest default for
    # chat/instruct models: user/system/tool tokens are context, not targets.
    assistant_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not assistant_idx:
        raise ValueError("loss_on='assistant' requires at least one assistant message")
    last = assistant_idx[-1]
    prompt_msgs = msgs[:last]
    completion = str(msgs[last]["content"])
    if not completion:
        raise ValueError("final assistant message is empty")
    if chat_template == "auto" and tokenizer is not None and _has_chat_template(tokenizer):
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = _render_messages_plain(prompt_msgs) + "assistant:"
    # Ensure ordinary chat completions separate from the assistant marker under
    # the fallback serializer. Native chat templates decide their own spacing.
    if not completion.startswith((" ", "\n", "\t")) and not (chat_template == "auto" and tokenizer is not None and _has_chat_template(tokenizer)):
        completion = " " + completion
    return Example(str(prompt), completion)


def _row_to_example(
    row: Mapping[str, Any],
    *,
    path: Path,
    line: int,
    tokenizer=None,
    chat_template: str = "auto",
    loss_on: str = "assistant",
) -> Example:
    if "messages" in row:
        return chat_messages_to_example(
            _normalise_messages(row, path=path, line=line),
            tokenizer=tokenizer,
            chat_template=chat_template,
            loss_on=loss_on,
        )
    if "prompt" in row and "completion" in row:
        return Example(_require_str(row, "prompt", path=path, line=line), _require_str(row, "completion", path=path, line=line))
    if "instruction" in row and "response" in row:
        return Example(_require_str(row, "instruction", path=path, line=line), _require_str(row, "response", path=path, line=line))
    if "question" in row and "answer" in row:
        return Example(str(row["question"]).rstrip() + "\nAnswer:", " " + str(row["answer"]))
    if "text" in row:
        return Example("", _require_str(row, "text", path=path, line=line))
    raise ValueError(
        f"{path}:{line} must contain prompt/completion, instruction/response, "
        "question/answer, text, or messages"
    )


def load_jsonl_examples(
    path: str | Path,
    *,
    tokenizer=None,
    chat_template: str = "auto",
    loss_on: str = "assistant",
) -> list[Example]:
    """Load examples from JSONL.

    Accepted schemas:
      - {"prompt": str, "completion": str}
      - {"instruction": str, "response": str}
      - {"question": str, "answer": str}
      - {"text": str}
      - {"messages": [{"role": ..., "content": ...}, ...]}

    Chat rows train on the final assistant turn by default. Pass a tokenizer to
    use its chat template; otherwise a transparent role-prefixed fallback is
    used. This function validates row types eagerly so data errors fail before a
    model is loaded for long training runs.
    """

    out: list[Example] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{i} invalid JSON: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{p}:{i} JSONL row must be an object")
            out.append(
                _row_to_example(
                    row,
                    path=p,
                    line=i,
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    loss_on=loss_on,
                )
            )
    if not out:
        raise ValueError(f"no examples found in {p}")
    return out


def save_jsonl_examples(path: str | Path, examples: Sequence[Example]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"prompt": ex.prompt, "completion": ex.completion}, ensure_ascii=False) + "\n")


def _encode(tokenizer, text: str, *, add_special_tokens: bool = False) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def encode_example(tokenizer, ex: Example, max_length: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input_ids and labels.

    Labels are -100 for prompt tokens and equal to input_ids for completion
    tokens. Loss is computed manually with the standard causal shift.
    """
    prompt_ids = _encode(tokenizer, ex.prompt, add_special_tokens=False) if ex.prompt else []
    completion_ids = _encode(tokenizer, ex.completion, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("completion tokenized to zero tokens")

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) < 2:
        raise ValueError("example must contain at least two tokens")

    if len(input_ids) > max_length:
        # Keep the right edge, where completions normally sit.
        drop = len(input_ids) - max_length
        input_ids = input_ids[drop:]
        labels = labels[drop:]
        if all(x == -100 for x in labels[1:]):
            raise ValueError("max_length truncated away all supervised completion tokens")

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if tokenizer.eos_token_id is not None and eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                "tokenizer has neither pad_token_id nor eos_token_id; set a pad token before batching. "
                "ntkmirror does not add new special tokens implicitly because the model embeddings "
                "would also need to be resized."
            )


def make_batch(
    tokenizer,
    examples: Sequence[Example],
    *,
    device: torch.device | str,
    max_length: int = 1024,
) -> dict[str, torch.Tensor]:
    ensure_pad_token(tokenizer)
    encoded = [encode_example(tokenizer, ex, max_length=max_length) for ex in examples]
    input_ids = pad_sequence(
        [x[0] for x in encoded], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence([x[1] for x in encoded], batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def batches(items: Sequence[Example], batch_size: int) -> Iterable[list[Example]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def tiny_arithmetic_train() -> list[Example]:
    return [
        Example("Problem: 14 + 27 = ?\nSolution:", " Add ones: 4+7=11, write 1 carry 1. Tens: 1+2+1=4. Answer: 41"),
        Example("Problem: 36 + 18 = ?\nSolution:", " Add ones: 6+8=14, write 4 carry 1. Tens: 3+1+1=5. Answer: 54"),
        Example("Problem: 45 + 29 = ?\nSolution:", " Add ones: 5+9=14, write 4 carry 1. Tens: 4+2+1=7. Answer: 74"),
        Example("Problem: 63 + 28 = ?\nSolution:", " Add ones: 3+8=11, write 1 carry 1. Tens: 6+2+1=9. Answer: 91"),
        Example("Problem: 58 + 37 = ?\nSolution:", " Add ones: 8+7=15, write 5 carry 1. Tens: 5+3+1=9. Answer: 95"),
    ]


def tiny_arithmetic_eval() -> list[Example]:
    return [
        Example("Problem: 47 + 36 = ?\nSolution:", " Add ones: 7+6=13, write 3 carry 1. Tens: 4+3+1=8. Answer: 83"),
        Example("Problem: 69 + 25 = ?\nSolution:", " Add ones: 9+5=14, write 4 carry 1. Tens: 6+2+1=9. Answer: 94"),
    ]
