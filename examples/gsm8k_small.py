#!/usr/bin/env python3
"""Create small GSM8K-style JSONL files for ntkmirror.

Requires: pip install datasets

Usage:
  python examples/gsm8k_small.py --train-size 64 --eval-size 32 --out-dir runs/gsm8k_small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=64)
    ap.add_argument("--eval-size", type=int, default=32)
    ap.add_argument("--out-dir", default="runs/gsm8k_small")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(split, n, path):
        with path.open("w", encoding="utf-8") as f:
            for row in ds[split].select(range(n)):
                prompt = "Question: " + row["question"].strip() + "\nAnswer:"
                completion = " " + row["answer"].strip()
                f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")

    write("train", args.train_size, out / "train.jsonl")
    write("test", args.eval_size, out / "eval.jsonl")
    print(out)


if __name__ == "__main__":
    main()
