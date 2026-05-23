# Persistent memory controllers

A controller can be treated as a small persistent memory item. Instead of
putting every retrieved document, conversation, or user preference into the
prompt, fit one signed log-mask controller per item and store it in a memory
index. At inference time, retrieve relevant controllers, compose them in signed
log-gate space, and attach the composed controller for generation.

```bash
ntkmirror memory add \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --store runs/memory \
  --id gsm8k-style \
  --train examples/math_train.jsonl \
  --text "worked addition arithmetic with carrying" \
  --tags math,arithmetic

ntkmirror memory search \
  --store runs/memory \
  --query "solve an addition problem with carrying"

ntkmirror memory generate \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --store runs/memory \
  --query "addition with carrying" \
  --prompt "Problem: 47 + 36 = ?\nSolution:"
```

## Why composition is simple

The intervention is

```text
h' = exp(s) h.
```

Adding two controller states adds their signed log-gates:

```text
s_AB = clip(w_A s_A + w_B s_B).
```

That is equivalent to multiplying their channel scales. The default memory
retriever uses softmax-normalised retrieval weights, so composing three memories
usually creates an interpolation rather than an uncontrolled sum.

## What persistent controllers are good for

They are most natural for compact procedural, stylistic, preference, or
domain-adaptation memories: a user's formatting preference, a document's local
terminology, a coding style, a narrow math procedure, or a recurring task
pattern. They are not a replacement for factual retrieval when the answer needs
verbatim evidence. For facts, use prompt/RAG provenance; for behaviour, style,
and local adaptation, use controller memory.

## Failure hypotheses to test

1. **Retrieval error.** The selected controllers are irrelevant. Always inspect
   `ntkmirror memory search` before blaming the controller.

2. **Composition interference.** Two relevant controllers may push overlapping
   gates in opposite directions. Use `ntkmirror inspect` on candidate controllers
   and reduce `--top-k` or use softmax weighting when overlap is high.

3. **Saturation.** Composing many controllers can clip signed gates at
   `max_log_gate`. Lower `--top-k`, use retrieval weights, or reduce the clip.

4. **Undertrained memory.** A memory item fitted from too few or noisy examples
   will retrieve correctly but inject a weak or unstable controller.

5. **Prompt/controller mismatch.** The controller changes the forward dynamics;
   it does not insert missing strings into the context. If the answer requires a
   quote or a date, put that evidence in the prompt.

6. **Stale or poisoned memory.** A controller is executable model state. Keep the
   memory store trusted, versioned, and deletable. Do not load untrusted `.pt`
   files.

7. **Task non-orthogonality.** Disjoint controllers compose best when their
   induced directions are weakly interfering. For overlapping skills, evaluate
   individual and composed memories on held-out examples.

## Minimal benchmark

For a memory benchmark, report:

- retrieval top-k accuracy against known relevant memory ids;
- base vs single-memory vs composed-memory NLL/exact accuracy;
- number of controllers composed;
- clipping fraction or max signed-gate magnitude;
- prompt tokens saved relative to putting the retrieved text in context;
- latency for retrieval, composition, and generation.
