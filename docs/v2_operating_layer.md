# V2 controller operating layer

This document records the v2 boundary introduced by the feature-architecture patch. It is intentionally conservative: controller artifacts are sparse activation interventions, not factual stores, and the serving path assumes PyTorch hooks are model-global unless wrapped by an isolation runtime.

## Runtime isolation

Use `ControllerRuntime` for request-scoped application rather than attaching a controller directly to a shared model. The runtime validates controller shape and identity, serialises hook application by default, and removes hooks in a `finally` block.

```python
from ntkmirror import ControllerRuntime

runtime = ControllerRuntime(model, tokenizer)
with runtime.apply(controller_state):
    out = model(**batch)
```

Heterogeneous per-row controllers are not silently supported. Split batches by controller or use a serving implementation that explicitly implements per-row scaling. This avoids a high-severity failure mode where one user's controller affects another user's request.

## Artifact admission

Controller artifacts should be linted before they enter a memory store or deployment registry:

```bash
ntkmirror lint --controller controller.pt --require-revision --out lint.json
ntkmirror card --controller controller.pt --out controller.card.md
```

The linter checks schema validity, model/tokenizer revision metadata, gate saturation, layer concentration, and presence of evaluation/safety metadata. The controller card is a human-readable artifact summary; it is not a safety proof.

## Model doctor

Before training or loading controllers for a new model family, run:

```bash
ntkmirror doctor --model MODEL --revision REV --out doctor.json
```

The doctor reports the detected decoder layer path, hidden size, tokenizer pad/chat-template status, dtype/device, and supported hook sites.

## Training hygiene

The normal fitting path now supports validation and retain-data objectives:

```bash
ntkmirror fit \
  --model MODEL \
  --revision REV \
  --train train.jsonl \
  --validation validation.jsonl \
  --eval-every 20 \
  --early-stop-patience 3 \
  --retain retain.jsonl \
  --retain-weight 0.2 \
  --kl-to-base 0.05 \
  --out controller.pt
```

`--retain-weight` adds supervised retain-set NLL. `--kl-to-base` penalizes drift from the base model on retain rows. Both require `--retain`.

## Chat data

`load_jsonl_examples` accepts either raw prompt/completion rows or chat rows:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

By default, chat rows train only on the final assistant turn. If a tokenizer with a chat template is supplied, `--chat-template auto` uses it; otherwise the fallback serialization is explicit role-prefixed text. This keeps the raw data contract visible.

## Memory governance

The local memory store now has namespaces, versioned artifacts, soft delete, rollback, atomic index writes, and an audit command:

```bash
ntkmirror memory add --store store --namespace alice --id style --train style.jsonl --overwrite
ntkmirror memory delete --store store --namespace alice --id style --soft
ntkmirror memory rollback --store store --namespace alice --id style --version 1
ntkmirror memory audit --store store --namespace alice --out audit.json
```

Search excludes soft-deleted rows and defaults to a positive retrieval threshold. A no-hit query falls back to the base model in generation/eval paths instead of attaching an arbitrary zero-score controller.

## Composition planning

Before composing multiple controllers, run:

```bash
ntkmirror compose-plan --controllers a.pt b.pt c.pt --out plan.json
```

The plan reports gate overlap, opposing-sign overlap, negative gate-space cosine, and preclip saturation. These are mechanical risk signals only; they do not prove behavioral compatibility.

## Explicitly out of scope for this patch

This patch does not add a full LoRA/PEFT baseline runner, a vector database/RAG stack, a safetensors artifact migration, a signed controller registry, true heterogeneous per-row serving, or a GPU benchmark. Those remain separate v2 workstreams. The runtime and artifact APIs are designed so those pieces can be added without changing controller state semantics.
