#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DEVICE=${DEVICE:-cuda}
GATES=${GATES:-512}
STEPS=${STEPS:-40}
OUT=${OUT:-runs/memory_demo}
export OUT
mkdir -p "$OUT"

python - <<'PY'
import os
from ntkmirror.data import save_jsonl_examples, tiny_arithmetic_train
from pathlib import Path
out = Path(os.environ["OUT"])
save_jsonl_examples(out / "carry_train.jsonl", tiny_arithmetic_train())
PY

ntkmirror fit \
  --model "$MODEL" --device "$DEVICE" \
  --train "$OUT/carry_train.jsonl" \
  --out "$OUT/carry_controller.pt" \
  --gates "$GATES" --steps "$STEPS" --batch-size 2 --score-batches 4 --max-length 512

ntkmirror memory add \
  --store "$OUT/store" \
  --id arithmetic_carrying \
  --controller "$OUT/carry_controller.pt" \
  --text "Two-digit addition with carrying. Add ones first, carry, then add tens." \
  --tags math,arithmetic \
  --overwrite

ntkmirror memory search \
  --store "$OUT/store" \
  --query "solve a two digit addition problem with carrying" \
  --top-k 2

ntkmirror memory generate \
  --model "$MODEL" --device "$DEVICE" \
  --store "$OUT/store" \
  --query "solve a two digit addition problem with carrying" \
  --prompt $'Problem: 47 + 36 = ?\nSolution:' \
  --top-k 1 --max-new-tokens 80
