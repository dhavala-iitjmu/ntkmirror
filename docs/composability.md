# Controller composability

A fitted controller is stored in signed log-gate coordinates. If controller A
has gate values `s_A` and controller B has `s_B`, their composition is

```text
s_AB = clip(w_A s_A + w_B s_B, -max_log_gate, max_log_gate)
```

and the attached intervention is

```text
h' = exp(s_AB) h.
```

This is not raw parameter addition. It is addition in the controller's
mirror/log-mask coordinates. Since channel scales multiply in activation space,
log-gate addition is the natural composition rule.

## Disjoint-task experiment

The included runner trains one controller on GSM8K-style maths examples and one
controller on MBPP-style code examples, composes them, and evaluates all four
conditions on both eval sets:

```bash
pip install -e '.[datasets]'
bash scripts/run_disjoint_composition.sh
```

Outputs:

```text
runs/disjoint_composition/gsm8k_controller.pt
runs/disjoint_composition/mbpp_controller.pt
runs/disjoint_composition/composed_controller.pt
runs/disjoint_composition/composition_report.json
runs/disjoint_composition/eval_*.json
```

The composition report includes:

- exact gate overlap,
- gate-space cosine,
- L2 norms,
- union size.

Low overlap and low cosine are a proxy that the two tasks occupy different
controller directions. The final test is behavioural: the composed controller
should retain the task-specific NLL improvements on both held-out eval sets.

## Failure hypotheses

1. **Negative transfer.** The composed controller improves task A but hurts task
   B. Try weights `--weights 1,0.5`, `0.5,1`, or a smaller `--max-log-gate`.

2. **Gate collision.** The controllers reuse the same gates with opposite signs.
   Inspect `composition_report.json`: high overlap with negative cosine is a
   warning sign.

3. **Saturation.** The composed `s_AB` clips many values at `max_log_gate`. Lower
   each task weight or raise the gate count instead of raising the clip budget.

4. **Under-capacity.** Each task controller is too small. Sweep `GATES=1024`,
   `5000`, `10000` before interpreting composition.

5. **Prompt/data mismatch.** GSM8K and MBPP prompts have different formatting.
   Evaluate each individual controller on both tasks to detect whether the
   apparent composition failure is actually prompt incompatibility.

6. **Exact-answer mismatch.** Use NLL/token accuracy for early diagnostics. Exact
   answer for code and maths can fail even when local teacher-forced NLL improves.
