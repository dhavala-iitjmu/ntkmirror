# Persistent memory benchmark

Run `bash run_persistent_memory_benchmark.sh` to train one controller-backed memory per task and evaluate base, gold-memory, retrieved-memory, composed-memory, retrieval recall, and cross-task NLL.

Default tasks are GSM8K, MBPP, HellaSwag, ARC-Easy, and BoolQ. For the small paper-style cross-task memory table, run `TASKS=arithmetic,gsm8k,mbpp TRAIN_PER_TASK=64 EVAL_PER_TASK=20 bash run_persistent_memory_benchmark.sh`.

Outputs:
- `summary.csv`: base/gold/retrieved/composed metrics by task.
- `cross_task_nll.csv`: every memory controller applied to every evaluation task.
- `retrieval_recall.csv`: top-1 and top-k retrieval recall using the known gold task id.
- `composition_report.json`: gate overlap/cosine diagnostics for all memory controllers.
- `selectivity_summary.csv`: gold-vs-wrong memory margins and worst wrong-memory harm.
- `retrieval_sweep.csv`: retrieval top-k recall for k=1..5.
- `controller_diagnostics.csv`: gate norm, saturation, and active-gate statistics.
- `controller_pairwise_geometry.csv`: pairwise Jaccard and gate-vector cosine.
- `leakage_audit.csv`: train/eval exact overlap and 5-gram overlap diagnostics.
