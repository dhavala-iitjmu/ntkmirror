# Persistent memory benchmark

Run `bash run_persistent_memory_benchmark.sh` to train one controller-backed memory per task and evaluate base, gold-memory, retrieved-memory, composed-memory, retrieval recall, and cross-task NLL.

Default tasks are GSM8K, MBPP, HellaSwag, ARC-Easy, and BoolQ. For the small paper-style cross-task memory table, run:

```bash
TASKS=arithmetic,gsm8k,mbpp TRAIN_PER_TASK=64 EVAL_PER_TASK=20 bash run_persistent_memory_benchmark.sh
```

## Retrieval modes

The benchmark now separates controller quality from retrieval quality. `MEMORY_TEXT_MODE=descriptor_prompts` stores the task descriptor plus prompt-only training snippets in the memory index. This avoids the lexical-descriptor failure where GSM8K and BoolQ queries do not contain words such as `grade-school` or `yes-no`.

Available retrieval methods:

```bash
RETRIEVAL_METHOD=lexical    # dependency-free TF-IDF cosine
RETRIEVAL_METHOD=embedding  # sentence-transformer cosine
RETRIEVAL_METHOD=hybrid     # weighted embedding + lexical score, default
```

Useful controls:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
HYBRID_ALPHA=0.65
MEMORY_TEXT_MODE=descriptor_prompts
MEMORY_TRAIN_SNIPPETS=32
TOP_K=1
REVISION=<model-commit-sha-or-tag>
TOKENIZER_REVISION=<tokenizer-commit-sha-or-tag>
TRUST_REMOTE_CODE=0
```

The benchmark also writes `retrieval_method_comparison.csv`, so lexical, embedding, and hybrid recall are visible side-by-side.

## Outputs

- `summary.csv`: base/gold/retrieved/composed metrics by task. `retrieved_query_memory` is the realistic per-example retrieval path; `retrieved_descriptor_memory` is a diagnostic only. Multiple-choice `choice_acc` is length-normalized; `choice_acc_sum_loss` is also reported for transparency.
- `cross_task_nll.csv`: every memory controller applied to every evaluation task.
- `retrieval_recall.csv`: top-1/top-k recall for the selected retrieval method.
- `retrieval_method_comparison.csv`: recall comparison for lexical, embedding, and hybrid retrieval.
- `retrieval_sweep.csv`: top-k recall for k=1..5 using the selected retrieval method.
- `per_example_retrieved_memory.csv`: retrieved memory ids per evaluation row.
- `composition_report.json`: gate overlap/cosine diagnostics for all memory controllers.
- `selectivity_summary.csv`: gold-vs-wrong memory margins and worst wrong-memory harm.
- `controller_diagnostics.csv`: gate norm, saturation, and active-gate statistics.
- `controller_pairwise_geometry.csv`: pairwise Jaccard and gate-vector cosine.
- `leakage_audit.csv`: train/eval exact overlap and 5-gram overlap diagnostics.

## Failure modes directly tested

1. **Descriptor lexical failure.** If lexical recall is low but embedding/hybrid recall is high, the memory controller is not the bottleneck; the lexical retriever is.
2. **Wrong-memory harm.** If wrong memories reduce task NLL less than gold or increase NLL, `cross_task_nll.csv` and `selectivity_summary.csv` expose it.
3. **Composition saturation.** If `clip_frac_active` is high, summed memories are saturating the log-gate budget.
4. **Gate collision.** If pairwise gate cosine is high and composed performance drops, the task controllers interfere geometrically.
5. **Leakage.** Exact and 5-gram train/eval overlap are audited before interpreting memory gains.
6. **Choice length bias.** Length-normalized multiple-choice accuracy is the default; summed-loss accuracy is reported separately so length effects are visible.
