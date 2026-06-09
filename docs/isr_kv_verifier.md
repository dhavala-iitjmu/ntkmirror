# ISR verifier gate and KV order-debias

This v2 feature turns the standalone ISR/KV AUC repro into a library and CLI
path for evidence-support verification. It evaluates whether a claim is fully
supported by evidence under three related scores:

```text
q_canon : one verifier pass on the canonical evidence order
q_marg  : mean P(YES) over randomized evidence orderings
isr     : q_marg - dispersion_penalty * std(P(YES) over orderings)
q_kv    : optional one-pass closed-form NTK KV order-debias score
```

`q_marg` is the headline AUROC score. `isr` is reported as the
order-stability gate score: examples whose support probability changes a lot
under evidence reordering are penalized. `q_kv` is optional and requires the
legacy `kv_delta_bayes_ntk` package used by the original KV controller.

## Basic usage

The dependency-light path uses an ordinary Hugging Face causal LM backend and
scores YES/NO choice sequence log-likelihoods:

```bash
pip install -e '.[isr]'
ntkmirror isr-auc \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset vitaminc \
  --n 200 \
  --num-orderings 6 \
  --out runs/isr_vitaminc.json
```

Built-in datasets:

```text
vitaminc   tals/vitaminc test split, binary supported/refuted rows; NEI rows skipped
hotpot     hotpotqa/hotpot_qa validation distractor split, gold-support vs distractor rows
ragtruth   wandb/RAGTruth-processed test split, balanced supported/hallucinated rows
```

Custom JSONL is also supported:

```jsonl
{"id":"row-1","claim":"The answer is Paris.","spans":["Paris is the capital of France."],"supported":true}
{"id":"row-2","claim":"The answer is Rome.","evidence":"Paris is the capital of France.","label":"refuted"}
```

```bash
ntkmirror isr-auc \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data-jsonl claims.jsonl \
  --out runs/isr_custom.json
```

By default raw claims and evidence are not copied into the result JSON. Use
`--include-raw` only for non-private data.

## Optional KV order-debias

The KV path reproduces the prototype's per-claim closed-form NTK ridge
projection. It pre-fills the evidence/claim cache, builds a dense value-cache
controller spec, computes the Jacobian of centered YES/NO event logits with
respect to the controller coordinates, solves

```text
min_q ||Bq - d||² + ridge ||q||²
```

and clips the resulting controller delta by `--kv-max-norm` before re-scoring.

```bash
PYTHONPATH=/path/to/kv_delta_bayes_ntk:$PYTHONPATH \
ntkmirror isr-auc \
  --backend kv-delta-bayes-ntk \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype fp32 \
  --device cuda:0 \
  --dataset vitaminc \
  --n 400 \
  --num-orderings 6 \
  --fit-controller \
  --use-layers 4 \
  --kv-ridge 1e-3 \
  --kv-max-norm 10 \
  --out runs/isr_kv_vitaminc.json
```

Security and reproducibility constraints are intentionally strict. If the
installed legacy backend does not accept requested reproducibility flags such as
`--revision`, `--local-files-only`, or `--cache-dir`, `ntkmirror` raises rather
than silently ignoring them. `--trust-remote-code` remains opt-in.

## Output schema

The output file contains:

```text
schema_version
feature = isr_kv_order_debias_auc
model / revision / backend
dataset or data-jsonl path
query and YES/NO choices
num_orderings, leak, dispersion_penalty
rows[]
summary.scores.{q_canon,q_marg,isr,q_kv}.auroc
summary.scores.*.tpr_at_leak
```

Rows with verifier failures are marked `status="error"` unless `--fail-fast` is
set. KV-controller failures are row-local by default: the row can still
contribute to `q_canon`, `q_marg`, and `isr`, while `kv_status="error"` records
the failure. This protects long sharded runs from losing all non-KV scores when
a single cache layout fails.

## Sharding

For large runs, use deterministic modulo sharding:

```bash
ntkmirror isr-auc --dataset ragtruth --n 1000 --shard 0 --nshards 4 --out shard0.json
ntkmirror isr-auc --dataset ragtruth --n 1000 --shard 1 --nshards 4 --out shard1.json
```

Each shard reports metrics for its own scored subset. Merge row JSONs before
reporting a final headline AUROC.

## Failure modes and guardrails

- `q_marg` is not a factuality proof. It is a verifier score under sampled
evidence orderings.
- ISR penalizes order sensitivity, but a confident wrong verifier can still
score a hallucinated claim highly.
- The HF backend scores choice sequence log-likelihoods. For non-YES/NO labels,
consider `--length-normalize-choices` if choices tokenize to different lengths.
- The KV debias solve is local to the current cache and event-logit projection;
large deltas are norm-clipped and reported with diagnostics.
- Built-in dataset loaders are for reproducible research. For product evals,
prefer explicit JSONL manifests with stable row IDs and dataset hashes.
- Raw claims/evidence can contain private data. The CLI omits them from output
unless `--include-raw` is explicitly set.
