#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
STORE=${STORE:-runs/memory_demo/store}
GATES=${GATES:-512}
STEPS=${STEPS:-40}
DEVICE=${DEVICE:-cuda}
mkdir -p runs/memory_demo

ntkmirror memory add \
  --model "$MODEL" \
  --device "$DEVICE" \
  --store "$STORE" \
  --id arithmetic-carrying \
  --train examples/math_train.jsonl \
  --text "worked addition arithmetic with carrying, answer format, two digit sums" \
  --tags math,arithmetic \
  --gates "$GATES" \
  --steps "$STEPS" \
  --batch-size 2 \
  --score-batches 4 \
  --max-length 512 \
  --overwrite

ntkmirror memory search \
  --store "$STORE" \
  --query "addition problem with carrying" \
  --top-k 2

ntkmirror memory generate \
  --model "$MODEL" \
  --device "$DEVICE" \
  --store "$STORE" \
  --query "addition problem with carrying" \
  --prompt "Problem: 47 + 36 = ?\nSolution:" \
  --top-k 1 \
  --max-new-tokens 80
