from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class Example:
    """One supervised causal-LM example.

    The prompt is context. The completion is teacher-forced and contributes to
    the loss. JSONL rows should usually look like:

        {"prompt": "Question...\nAnswer:", "completion": " worked solution..."}

    A row with only {"text": ...} is also accepted; all tokens after the
    first one are supervised.
    """

    prompt: str
    completion: str


def load_jsonl_examples(path: str | Path) -> list[Example]:
    """Load examples from JSONL.

    Accepted schemas:
      - {"prompt": str, "completion": str}
      - {"instruction": str, "response": str}
      - {"question": str, "answer": str}
      - {"text": str}
    """
    out: list[Example] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row and "completion" in row:
                out.append(Example(str(row["prompt"]), str(row["completion"])))
            elif "instruction" in row and "response" in row:
                out.append(Example(str(row["instruction"]), str(row["response"])))
            elif "question" in row and "answer" in row:
                out.append(Example(str(row["question"]).rstrip() + "\nAnswer:", " " + str(row["answer"])))
            elif "text" in row:
                out.append(Example("", str(row["text"])))
            else:
                raise ValueError(
                    f"{p}:{i} must contain prompt/completion, instruction/response, "
                    "question/answer, or text"
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
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


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
