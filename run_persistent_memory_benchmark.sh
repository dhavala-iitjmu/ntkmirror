#!/usr/bin/env bash
set -euo pipefail

# Single-entry runner for the persistent controller-memory benchmark.
# It prepares public task subsets, fits one controller-memory item per task,
# registers the memory store, and evaluates cross-task NLL plus retrieval recall.
#
# Typical usage:
#   bash run_persistent_memory_benchmark.sh
#
# Useful overrides:
#   MODEL=Qwen/Qwen2.5-7B-Instruct TASKS=gsm8k,mbpp,hellaswag,arc_easy,boolq \
#   TRAIN_PER_TASK=512 EVAL_PER_TASK=200 GATES=50000 RETRIEVAL_METHOD=hybrid MEMORY_TEXT_MODE=descriptor_prompts GPU_LIST="0 1 2 3 4" \
#   bash run_persistent_memory_benchmark.sh

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

OUT=${OUT:-runs/persistent_memory_public}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TASKS=${TASKS:-gsm8k,mbpp,hellaswag,arc_easy,boolq}
TRAIN_PER_TASK=${TRAIN_PER_TASK:-128}
EVAL_PER_TASK=${EVAL_PER_TASK:-64}
SEED=${SEED:-0}
DTYPE=${DTYPE:-bf16}
GATES=${GATES:-21000}
STEPS=${STEPS:-240}
LR=${LR:-5e-3}
MAX_LOG_GATE=${MAX_LOG_GATE:-0.05}
SCORE_BATCHES=${SCORE_BATCHES:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
TOP_K=${TOP_K:-1}
RETRIEVAL_METHOD=${RETRIEVAL_METHOD:-hybrid}
RETRIEVAL_COMPARE_METHODS=${RETRIEVAL_COMPARE_METHODS:-lexical,embedding,hybrid}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-cpu}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-64}
HYBRID_ALPHA=${HYBRID_ALPHA:-0.65}
MEMORY_TEXT_MODE=${MEMORY_TEXT_MODE:-descriptor_prompts}
MEMORY_TRAIN_SNIPPETS=${MEMORY_TRAIN_SNIPPETS:-32}
GPU_LIST=${GPU_LIST:-$(python - <<'PY'
import torch
n=torch.cuda.device_count()
print(' '.join(str(i) for i in range(n)) if n else '')
PY
)}

mkdir -p "$OUT" logs

python scripts/benchmark_persistent_memory_suite.py prepare \
  --out "$OUT" \
  --tasks "$TASKS" \
  --train-per-task "$TRAIN_PER_TASK" \
  --eval-per-task "$EVAL_PER_TASK" \
  --seed "$SEED"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
read -r -a GPU_ARRAY <<< "$GPU_LIST"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  GPU_ARRAY=("")
fi

pids=()
for idx in "${!TASK_ARRAY[@]}"; do
  task="${TASK_ARRAY[$idx]}"
  gpu="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  log="logs/fit_${task}.log"
  echo "[fit] task=$task gpu=${gpu:-cpu} log=$log"
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" MODEL="$MODEL" DTYPE="$DTYPE" GATES="$GATES" STEPS="$STEPS" LR="$LR" \
    MAX_LOG_GATE="$MAX_LOG_GATE" SCORE_BATCHES="$SCORE_BATCHES" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    python scripts/benchmark_persistent_memory_suite.py fit-one --out "$OUT" --task "$task" --quiet --overwrite > "$log" 2>&1 &
  else
    MODEL="$MODEL" DTYPE="$DTYPE" GATES="$GATES" STEPS="$STEPS" LR="$LR" \
    MAX_LOG_GATE="$MAX_LOG_GATE" SCORE_BATCHES="$SCORE_BATCHES" TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    DEVICE=cpu python scripts/benchmark_persistent_memory_suite.py fit-one --out "$OUT" --task "$task" --quiet --overwrite > "$log" 2>&1 &
  fi
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python scripts/benchmark_persistent_memory_suite.py register --out "$OUT" --memory-text-mode "$MEMORY_TEXT_MODE" --memory-train-snippets "$MEMORY_TRAIN_SNIPPETS"

eval_gpu=""
if [[ ${#GPU_ARRAY[@]} -gt 0 ]]; then eval_gpu="${GPU_ARRAY[0]}"; fi
if [[ -n "$eval_gpu" ]]; then
  CUDA_VISIBLE_DEVICES="$eval_gpu" MODEL="$MODEL" DTYPE="$DTYPE" GATES="$GATES" MAX_LOG_GATE="$MAX_LOG_GATE" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" TOP_K="$TOP_K" RETRIEVAL_METHOD="$RETRIEVAL_METHOD" RETRIEVAL_COMPARE_METHODS="$RETRIEVAL_COMPARE_METHODS" \
  EMBEDDING_MODEL="$EMBEDDING_MODEL" EMBEDDING_DEVICE="$EMBEDDING_DEVICE" EMBEDDING_BATCH_SIZE="$EMBEDDING_BATCH_SIZE" HYBRID_ALPHA="$HYBRID_ALPHA" MEMORY_TEXT_MODE="$MEMORY_TEXT_MODE" \
  python scripts/benchmark_persistent_memory_suite.py eval --out "$OUT"
else
  MODEL="$MODEL" DTYPE="$DTYPE" GATES="$GATES" DEVICE=cpu EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" TOP_K="$TOP_K" RETRIEVAL_METHOD="$RETRIEVAL_METHOD" RETRIEVAL_COMPARE_METHODS="$RETRIEVAL_COMPARE_METHODS" \
  EMBEDDING_MODEL="$EMBEDDING_MODEL" EMBEDDING_DEVICE="$EMBEDDING_DEVICE" EMBEDDING_BATCH_SIZE="$EMBEDDING_BATCH_SIZE" HYBRID_ALPHA="$HYBRID_ALPHA" MEMORY_TEXT_MODE="$MEMORY_TEXT_MODE" \
  python scripts/benchmark_persistent_memory_suite.py eval --out "$OUT"
fi

python - <<PY
from pathlib import Path
import pandas as pd
out=Path('$OUT')
print('\n=== summary.csv ===')
print(pd.read_csv(out/'summary.csv').to_string(index=False))
print('\n=== retrieval_recall.csv ===')
print(pd.read_csv(out/'retrieval_recall.csv').to_string(index=False))
print('\n=== cross_task_nll.csv ===')
print(pd.read_csv(out/'cross_task_nll.csv').to_string(index=False))
for extra in ['retrieval_method_comparison.csv','selectivity_summary.csv','retrieval_sweep.csv','controller_diagnostics.csv','controller_pairwise_geometry.csv','leakage_audit.csv','per_example_retrieved_memory.csv']:
    path=out/extra
    if path.exists():
        print(f'\n=== {extra} ===')
        print(pd.read_csv(path).to_string(index=False))
PY

echo "\nDone. Outputs in $OUT"
