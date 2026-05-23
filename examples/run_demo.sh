#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DEVICE=${DEVICE:-cuda}

ntkmirror fit \
  --model "$MODEL" \
  --device "$DEVICE" \
  --train examples/math_train.jsonl \
  --out runs/math_controller.pt \
  --gates ${GATES:-1024} \
  --steps ${STEPS:-80}

ntkmirror eval \
  --model "$MODEL" \
  --device "$DEVICE" \
  --controller runs/math_controller.pt \
  --eval examples/math_eval.jsonl

ntkmirror generate \
  --model "$MODEL" \
  --device "$DEVICE" \
  --controller runs/math_controller.pt \
  --prompt "Problem: 47 + 36 = ?\nSolution:" \
  --max-new-tokens 80
