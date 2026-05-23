#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-runs/disjoint_composition}
TRAIN_SIZE=${TRAIN_SIZE:-64}
EVAL_SIZE=${EVAL_SIZE:-32}
GATES=${GATES:-5000}
STEPS=${STEPS:-240}
LR=${LR:-5e-3}
SEED=${SEED:-0}

mkdir -p "$OUT"
python examples/make_disjoint_sft.py \
  --out-dir "$OUT/data" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --seed "$SEED"

ntkmirror fit --model "$MODEL" --device "$DEVICE" \
  --train "$OUT/data/gsm8k_train.jsonl" \
  --out "$OUT/gsm8k_controller.pt" \
  --gates "$GATES" --steps "$STEPS" --lr "$LR"

ntkmirror fit --model "$MODEL" --device "$DEVICE" \
  --train "$OUT/data/mbpp_train.jsonl" \
  --out "$OUT/mbpp_controller.pt" \
  --gates "$GATES" --steps "$STEPS" --lr "$LR"

ntkmirror compose \
  --controllers "$OUT/gsm8k_controller.pt" "$OUT/mbpp_controller.pt" \
  --out "$OUT/composed_controller.pt" \
  --report "$OUT/composition_report.json"

for CTRL in none gsm8k mbpp composed; do
  case "$CTRL" in
    none) CARGS=() ; SUFFIX=base ;;
    gsm8k) CARGS=(--controller "$OUT/gsm8k_controller.pt") ; SUFFIX=gsm8k ;;
    mbpp) CARGS=(--controller "$OUT/mbpp_controller.pt") ; SUFFIX=mbpp ;;
    composed) CARGS=(--controller "$OUT/composed_controller.pt") ; SUFFIX=composed ;;
  esac
  ntkmirror eval --model "$MODEL" --device "$DEVICE" \
    "${CARGS[@]}" --eval "$OUT/data/gsm8k_eval.jsonl" \
    --out "$OUT/eval_gsm8k_${SUFFIX}.json"
  ntkmirror eval --model "$MODEL" --device "$DEVICE" \
    "${CARGS[@]}" --eval "$OUT/data/mbpp_eval.jsonl" \
    --out "$OUT/eval_mbpp_${SUFFIX}.json"
done

python - <<'PY'
import json, pathlib, os
out = pathlib.Path(os.environ.get('OUT', 'runs/disjoint_composition'))
print('\n=== composition report ===')
print((out/'composition_report.json').read_text())
print('\n=== eval files ===')
for p in sorted(out.glob('eval_*.json')):
    obj=json.loads(p.read_text())
    ctrl=obj.get('controller')
    base=obj.get('base')
    print(p.name, 'base_nll=', round(base['nll'],4), 'ctrl_nll=', None if ctrl is None else round(ctrl['nll'],4))
PY
