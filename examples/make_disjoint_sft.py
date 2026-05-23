#!/usr/bin/env python3
"""Build tiny disjoint SFT tasks for controller-composition tests.

Default tasks:
  - GSM8K math reasoning: openai/gsm8k, main
  - MBPP code generation: Muennighoff/mbpp, sanitized/full when available

The output is four JSONL files with prompt/completion rows:
  gsm8k_train.jsonl, gsm8k_eval.jsonl, mbpp_train.jsonl, mbpp_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_gsm8k():
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main")


def _load_mbpp():
    from datasets import load_dataset

    attempts = [
        ("Muennighoff/mbpp", "sanitized"),
        ("Muennighoff/mbpp", "full"),
        ("Muennighoff/mbpp", None),
        ("google-research-datasets/mbpp", None),
    ]
    last = None
    for name, config in attempts:
        try:
            return load_dataset(name, config) if config is not None else load_dataset(name)
        except Exception as e:  # pragma: no cover - depends on Hub availability
            last = e
    raise RuntimeError(f"could not load an MBPP dataset variant: {last}")


def _gsm_row(row: dict[str, Any]) -> dict[str, str]:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    return {
        "prompt": "Solve the grade-school maths problem step by step. End with #### <answer>.\n\nQuestion: "
        + question
        + "\nAnswer:",
        "completion": " " + answer,
    }


def _mbpp_row(row: dict[str, Any]) -> dict[str, str]:
    task = str(row.get("text") or row.get("prompt") or row.get("task") or row.get("description") or "").strip()
    code = str(row.get("code") or row.get("target") or row.get("canonical_solution") or row.get("solution") or "").strip()
    tests = row.get("test_list") or row.get("test") or row.get("tests") or []
    if isinstance(tests, list):
        tests_text = "\n".join(str(x) for x in tests[:3])
    else:
        tests_text = str(tests)
    prompt = "Write a Python function for the programming task. Return only valid Python code.\n\nTask: " + task
    if tests_text.strip():
        prompt += "\n\nTests:\n" + tests_text
    prompt += "\n\nCode:\n"
    return {"prompt": prompt, "completion": code + "\n"}


def _take(ds, split_candidates: list[str], n: int, offset: int, seed: int, formatter) -> list[dict[str, str]]:
    split = next((s for s in split_candidates if s in ds), None)
    if split is None:
        split = next(iter(ds.keys()))
    rows = list(ds[split])
    random.Random(seed).shuffle(rows)
    out = []
    for row in rows[offset:]:
        ex = formatter(row)
        if ex["prompt"].strip() and ex["completion"].strip():
            out.append(ex)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only built {len(out)} examples from split {split}, requested {n}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/disjoint_sft/data")
    ap.add_argument("--train-size", type=int, default=64)
    ap.add_argument("--eval-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    gsm = _load_gsm8k()
    mbpp = _load_mbpp()
    _write(out / "gsm8k_train.jsonl", _take(gsm, ["train"], args.train_size, 0, args.seed, _gsm_row))
    _write(out / "gsm8k_eval.jsonl", _take(gsm, ["test", "validation", "train"], args.eval_size, args.train_size, args.seed + 1, _gsm_row))
    _write(out / "mbpp_train.jsonl", _take(mbpp, ["train", "prompt", "validation", "test"], args.train_size, 0, args.seed + 2, _mbpp_row))
    _write(out / "mbpp_eval.jsonl", _take(mbpp, ["test", "validation", "evaluation", "train"], args.eval_size, args.train_size, args.seed + 3, _mbpp_row))
    print(out)


if __name__ == "__main__":
    main()
