# Method note

`ntkmirror` implements the deployable controller, not the full research harness.
The base model is frozen. A sparse set of decoder-layer output channels is
selected by the magnitude of the log-gate derivative

```text
dL/ds_{l,c} = sum_t <dL/dh_{l,t,c}, h_{l,t,c}> .
```

The fitted controller is a shared signed log-mask

```text
h'_{l,t,c} = exp(s_{l,c}) h_{l,t,c},   |s_{l,c}| <= max_log_gate.
```

The same `s` is attached during support scoring, held-out evaluation, and
generation. No LoRA modules or permanent weight edits are inserted.

## Why this interface is small

The public API exposes one class and one CLI:

```python
from ntkmirror import ForwardFineTuner, load_jsonl_examples

tuner = ForwardFineTuner(model, tokenizer, gates=5000)
tuner.fit(load_jsonl_examples("train.jsonl"))
tuner.save("controller.pt")
```

The CLI is the same:

```bash
ntkmirror fit --model Qwen/Qwen2.5-0.5B-Instruct --train train.jsonl --out controller.pt
ntkmirror eval --model Qwen/Qwen2.5-0.5B-Instruct --controller controller.pt --eval eval.jsonl
```

## Failure modes to check before believing a result

1. **Base example selection.** If evaluation contains problems the base model
   already solves, exact-accuracy deltas can be washed out. Report base metrics.

2. **Controller capacity.** A 512-gate controller is a different object from a
   5k or 10k controller. Sweep `--gates` before making a LoRA comparison.

3. **NLL vs exact answer.** The controller can improve NLL without improving
   long-horizon exact-answer accuracy. Report both where possible.

4. **Gate budget.** Very large `--max-log-gate` leaves the local regime and can
   hurt generation. `0.03` to `0.05` is the intended starting range.

5. **Layer hooks.** This package gates decoder-layer outputs. Architectures
   without a standard decoder stack may need a layer-path extension.

6. **Seed manifests.** For benchmark claims, fix train/eval splits and compare
   every arm on the same examples.

7. **Oracle diagnostics.** Fitting a controller to an SGD displacement is useful
   for theorem diagnostics, but it is not a deployable fine-tuning method. The
   deployable path here is support-loss optimisation of the gates.
