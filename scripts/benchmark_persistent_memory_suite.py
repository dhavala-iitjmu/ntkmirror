#!/usr/bin/env python3
"""Persistent controller-memory benchmark.

This script prepares task datasets, fits one signed log-mask controller per task,
registers the controllers as persistent memory items, and evaluates base, gold
memory, retrieved memory, composed memory, and the full cross-task NLL matrix.

It is intentionally built on the small ntkmirror public API:
  - ForwardFineTuner(model, tokenizer, gates=...)
  - SignedLogMaskState / compose_states
  - ControllerMemoryStore

Default tasks are public and common: GSM8K, MBPP, HellaSwag, ARC-Easy, BoolQ.
For reproducing the paper's small memory table use TASKS=arithmetic,gsm8k,mbpp.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from ntkmirror.compose import compose_states, composition_report, save_report, dense_gate_vector, pair_report
from ntkmirror.controller import ForwardFineTuner, SignedLogMaskState
from ntkmirror.data import Example, load_jsonl_examples, save_jsonl_examples, make_batch
from ntkmirror.memory import ControllerMemoryStore, MemoryHit
from ntkmirror.retrieval import MemoryRetriever, build_memory_retriever


@dataclasses.dataclass
class EvalRow:
    prompt: str
    completion: str
    query: str
    choices: list[str] | None = None
    gold_index: int | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TaskPack:
    task_id: str
    descriptor: str
    train: list[Example]
    eval: list[EvalRow]


def _write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rows_to_examples(rows: Sequence[EvalRow]) -> list[Example]:
    return [Example(r.prompt, r.completion) for r in rows]


def _save_eval_rows(path: str | Path, rows: Sequence[EvalRow]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")


def _load_eval_rows(path: str | Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(EvalRow(**obj))
    return rows


def _strip_answer(ans: str) -> str:
    # GSM8K answers often end with "#### 42". Keep the full chain as a
    # completion for NLL, but exact-answer utilities can use this parser.
    m = re.search(r"####\s*([^\n]+)", ans)
    return (m.group(1) if m else ans).strip()


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - import guard for first-run UX
        raise RuntimeError("install with `pip install -e .[datasets]` or `pip install datasets`") from e
    return load_dataset(*args, **kwargs)


def _sample_rows(rows: Sequence[Any], n: int, rng: random.Random) -> list[Any]:
    rows = list(rows)
    if n <= 0 or n >= len(rows):
        return rows
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:n]]


def make_arithmetic(train_n: int, eval_n: int, seed: int) -> TaskPack:
    rng = random.Random(seed)
    def one() -> tuple[str, str]:
        # Force carrying in the ones digit.
        a_t, b_t = rng.randint(1, 8), rng.randint(1, 8)
        a_o = rng.randint(4, 9)
        b_o = rng.randint(max(10 - a_o, 1), 9)
        a = 10 * a_t + a_o
        b = 10 * b_t + b_o
        s = a + b
        ones = a_o + b_o
        prompt = f"Problem: {a} + {b} = ?\nSolution:"
        comp = f" Add ones: {a_o}+{b_o}={ones}, write {ones%10} carry 1. Tens: {a_t}+{b_t}+1={s//10}. Answer: {s}"
        return prompt, comp
    train = [Example(*one()) for _ in range(train_n)]
    eval_rows = [EvalRow(*one(), query="two digit addition with carrying arithmetic") for _ in range(eval_n)]
    return TaskPack("arithmetic", "two digit addition arithmetic with carrying and step-by-step worked solutions", train, eval_rows)


def make_gsm8k(train_n: int, eval_n: int, seed: int) -> TaskPack:
    rng = random.Random(seed)
    ds_train = _load_dataset("openai/gsm8k", "main", split="train")
    ds_test = _load_dataset("openai/gsm8k", "main", split="test")
    tr = _sample_rows(ds_train, train_n, rng)
    ev = _sample_rows(ds_test, eval_n, rng)
    train = [Example(f"Question: {x['question']}\nAnswer:", " " + x["answer"].strip()) for x in tr]
    eval_rows = [
        EvalRow(
            prompt=f"Question: {x['question']}\nAnswer:",
            completion=" " + x["answer"].strip(),
            query=str(x["question"]),
            meta={"answer": _strip_answer(str(x["answer"]))},
        )
        for x in ev
    ]
    return TaskPack("gsm8k", "grade-school math word problems requiring multi-step arithmetic reasoning", train, eval_rows)


def _load_mbpp_dataset():
    # Prefer the canonical HF mirror when available; fall back to Muennighoff.
    for name in ["google-research-datasets/mbpp", "Muennighoff/mbpp"]:
        try:
            return name, _load_dataset(name)
        except Exception:
            continue
    raise RuntimeError("could not load an MBPP dataset mirror")


def make_mbpp(train_n: int, eval_n: int, seed: int) -> TaskPack:
    rng = random.Random(seed)
    _name, ds = _load_mbpp_dataset()
    # Different mirrors use different split names.
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else train_split)
    tr = _sample_rows(ds[train_split], train_n, rng)
    ev = _sample_rows(ds[eval_split], eval_n, rng)

    def fields(x):
        prompt = x.get("text") or x.get("prompt") or x.get("question") or x.get("task") or ""
        code = x.get("code") or x.get("completion") or x.get("target") or x.get("answer") or ""
        tests = x.get("test_list") or x.get("tests") or []
        return str(prompt), str(code), tests

    train: list[Example] = []
    eval_rows: list[EvalRow] = []
    for x in tr:
        prompt, code, _tests = fields(x)
        train.append(Example("Write a Python function for this task.\nTask: " + prompt + "\nSolution:\n", code))
    for x in ev:
        prompt, code, tests = fields(x)
        eval_rows.append(EvalRow(
            prompt="Write a Python function for this task.\nTask: " + prompt + "\nSolution:\n",
            completion=code,
            query="python programming code generation " + prompt,
            meta={"tests": tests},
        ))
    return TaskPack("mbpp", "Python code generation from natural-language programming tasks", train, eval_rows)


def make_hellaswag(train_n: int, eval_n: int, seed: int) -> TaskPack:
    rng = random.Random(seed)
    ds_train = _load_dataset("Rowan/hellaswag", split="train")
    ds_val = _load_dataset("Rowan/hellaswag", split="validation")
    tr = _sample_rows(ds_train, train_n, rng)
    ev = _sample_rows(ds_val, eval_n, rng)

    def row_to_choice(x):
        ctx = str(x.get("ctx") or x.get("ctx_a") or "")
        endings = [str(e) for e in x["endings"]]
        label = int(x["label"])
        prompt = "Context: " + ctx.strip() + "\nBest continuation:"
        return prompt, endings, label

    train = []
    eval_rows = []
    for x in tr:
        prompt, endings, label = row_to_choice(x)
        train.append(Example(prompt, " " + endings[label].strip()))
    for x in ev:
        prompt, endings, label = row_to_choice(x)
        eval_rows.append(EvalRow(prompt, " " + endings[label].strip(), query=prompt, choices=[" " + e.strip() for e in endings], gold_index=label))
    return TaskPack("hellaswag", "commonsense continuation selection from short situations", train, eval_rows)


def make_boolq(train_n: int, eval_n: int, seed: int) -> TaskPack:
    rng = random.Random(seed)
    ds_train = _load_dataset("google/boolq", split="train")
    ds_val = _load_dataset("google/boolq", split="validation")
    tr = _sample_rows(ds_train, train_n, rng)
    ev = _sample_rows(ds_val, eval_n, rng)

    def mk(x):
        prompt = f"Passage: {x['passage']}\nQuestion: {x['question']}\nAnswer yes or no:"
        ans = " yes" if bool(x["answer"]) else " no"
        return prompt, ans
    train = [Example(*mk(x)) for x in tr]
    eval_rows = []
    for x in ev:
        prompt, ans = mk(x)
        gold = 0 if ans.strip() == "yes" else 1
        eval_rows.append(EvalRow(prompt, ans, query=x["question"] + " " + x["passage"][:300], choices=[" yes", " no"], gold_index=gold))
    return TaskPack("boolq", "yes-no question answering over a short passage", train, eval_rows)


def make_arc_easy(train_n: int, eval_n: int, seed: int) -> TaskPack:
    rng = random.Random(seed)
    ds_train = _load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    ds_test = _load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    tr = _sample_rows(ds_train, train_n, rng)
    ev = _sample_rows(ds_test, eval_n, rng)

    def parse(x):
        q = str(x["question"])
        labels = [str(v) for v in x["choices"]["label"]]
        texts = [str(v) for v in x["choices"]["text"]]
        answer = str(x["answerKey"])
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        gold = label_to_idx.get(answer, 0)
        choices_str = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        prompt = f"Question: {q}\nChoices:\n{choices_str}\nAnswer:"
        return prompt, [" " + t for t in texts], gold, labels
    train = []
    eval_rows = []
    for x in tr:
        prompt, choices, gold, labels = parse(x)
        train.append(Example(prompt, choices[gold]))
    for x in ev:
        prompt, choices, gold, labels = parse(x)
        eval_rows.append(EvalRow(prompt, choices[gold], query=prompt, choices=choices, gold_index=gold, meta={"labels": labels}))
    return TaskPack("arc_easy", "science multiple-choice question answering", train, eval_rows)


TASK_BUILDERS = {
    "arithmetic": make_arithmetic,
    "gsm8k": make_gsm8k,
    "mbpp": make_mbpp,
    "hellaswag": make_hellaswag,
    "boolq": make_boolq,
    "arc_easy": make_arc_easy,
}


def _task_ids_arg(s: str) -> list[str]:
    ids = [x.strip() for x in s.split(",") if x.strip()]
    for t in ids:
        if t not in TASK_BUILDERS:
            raise ValueError(f"unknown task {t!r}; options are {sorted(TASK_BUILDERS)}")
    return ids


def cmd_prepare(args) -> None:
    out = Path(args.out)
    tasks_dir = out / "tasks"
    rng_seed = int(args.seed)
    manifest = {"tasks": [], "seed": rng_seed, "train_per_task": args.train_per_task, "eval_per_task": args.eval_per_task}
    for i, task_id in enumerate(_task_ids_arg(args.tasks)):
        pack = TASK_BUILDERS[task_id](args.train_per_task, args.eval_per_task, rng_seed + 1009 * i)
        train_path = tasks_dir / f"{task_id}.train.jsonl"
        eval_path = tasks_dir / f"{task_id}.eval.jsonl"
        save_jsonl_examples(train_path, pack.train)
        _save_eval_rows(eval_path, pack.eval)
        row = {
            "task_id": pack.task_id,
            "descriptor": pack.descriptor,
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "train_n": len(pack.train),
            "eval_n": len(pack.eval),
        }
        manifest["tasks"].append(row)
        print(json.dumps(row), flush=True)
    _write_json(out / "manifest.json", manifest)


def _load_model(
    model_name: str,
    device: str,
    dtype: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_revision = tokenizer_revision or revision
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(trust_remote_code),
        revision=tok_revision,
        local_files_only=bool(local_files_only),
        cache_dir=cache_dir,
    )
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype must be auto/bf16/fp16/fp32")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=bool(trust_remote_code),
        revision=revision,
        local_files_only=bool(local_files_only),
        cache_dir=cache_dir,
    )
    if hasattr(model, "config"):
        setattr(model.config, "_ntkmirror_requested_revision", revision)
    setattr(tok, "_ntkmirror_requested_revision", tok_revision)
    model.to(torch.device(device))
    model.eval()
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tok


def _make_tuner(args) -> ForwardFineTuner:
    model, tok = _load_model(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        local_files_only=getattr(args, "local_files_only", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    return ForwardFineTuner(
        model,
        tok,
        gates=args.gates,
        layers=args.layers,
        max_log_gate=args.max_log_gate,
        model_name=args.model,
        model_revision=getattr(args, "revision", None),
        tokenizer_name=args.model,
        tokenizer_revision=getattr(args, "tokenizer_revision", None) or getattr(args, "revision", None),
    )


def cmd_fit_one(args) -> None:
    out = Path(args.out)
    manifest = _read_json(out / "manifest.json")
    task = next((t for t in manifest["tasks"] if t["task_id"] == args.task), None)
    if task is None:
        raise ValueError(f"task {args.task!r} not in manifest")
    ctrl_path = out / "controllers" / f"{args.task}.pt"
    result_path = out / "fit" / f"{args.task}.json"
    if ctrl_path.exists() and not args.overwrite:
        print(f"controller exists: {ctrl_path}; skipping", flush=True)
        return
    examples = load_jsonl_examples(task["train_path"])
    tuner = _make_tuner(args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = tuner.evaluate_nll(examples, batch_size=args.train_batch_size, max_length=args.max_length, use_controller=False)
    t0 = time.perf_counter()
    fit = tuner.fit(
        examples,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.train_batch_size,
        score_batches=args.score_batches,
        max_length=args.max_length,
        l2=args.l2,
        verbose=not args.quiet,
    )
    adapt_seconds = time.perf_counter() - t0
    after = tuner.evaluate_nll(examples, batch_size=args.train_batch_size, max_length=args.max_length, use_controller=True)
    tuner.save(ctrl_path)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    result = {
        "task_id": args.task,
        "controller_path": str(ctrl_path),
        "before": before,
        "after": after,
        "fit": fit,
        "adapt_seconds_total": adapt_seconds,
        "peak_alloc_gb": peak_gb,
        "gates": args.gates,
        "steps": args.steps,
        "lr": args.lr,
    }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)



def _memory_text_for_task(task: dict[str, Any], *, mode: str, train_snippets: int) -> str:
    """Build retrieval text for one controller-memory item.

    Descriptor-only retrieval is intentionally brittle: GSM8K and BoolQ queries
    often do not contain words like "grade-school" or "yes-no".  The default
    includes a small prompt-only sample from the controller's own training set,
    which is the realistic memory-index analogue of storing document snippets
    beside an activation controller.  Completions are excluded by default so the
    retriever does not index answer text.
    """
    desc = str(task.get("descriptor", ""))
    mode = str(mode or "descriptor_prompts").strip().lower()
    if mode in {"descriptor", "desc"}:
        return desc
    examples = load_jsonl_examples(task["train_path"])
    snippets = []
    for ex in examples[: max(0, int(train_snippets))]:
        if mode in {"descriptor_prompts", "prompts", "descriptor_prompt_examples"}:
            snippets.append(ex.prompt[:1000])
        elif mode in {"descriptor_full", "full", "descriptor_train_examples"}:
            snippets.append((ex.prompt + "\n" + ex.completion)[:1200])
        else:
            raise ValueError("memory_text_mode must be descriptor, descriptor_prompts, or descriptor_full")
    return desc + ("\n\nRepresentative prompts:\n" + "\n---\n".join(snippets) if snippets else "")

def cmd_register(args) -> None:
    out = Path(args.out)
    manifest = _read_json(out / "manifest.json")
    store = ControllerMemoryStore(out / "memory_store")
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        ctrl_path = out / "controllers" / f"{task_id}.pt"
        if not ctrl_path.exists():
            raise FileNotFoundError(ctrl_path)
        retrieval_text = _memory_text_for_task(
            task,
            mode=args.memory_text_mode,
            train_snippets=args.memory_train_snippets,
        )
        store.add_controller(
            memory_id=task_id,
            controller_path=ctrl_path,
            text=retrieval_text,
            tags=[task_id, "benchmark"],
            metadata={
                "task_id": task_id,
                "descriptor": task["descriptor"],
                "retrieval_text_mode": args.memory_text_mode,
                "retrieval_text": retrieval_text[:4000],
            },
            overwrite=True,
        )
    print(json.dumps({"store": str(out / "memory_store"), "items": [t["task_id"] for t in manifest["tasks"]]}, indent=2), flush=True)


def _attach_state(tuner: ForwardFineTuner, state: SignedLogMaskState | None):
    if tuner.controller is not None:
        tuner.controller.remove()
    if state is None:
        tuner.controller = None
    else:
        tmp = Path(os.environ.get("NTKMIRROR_TMP", "/tmp")) / f"ntkmirror_tmp_state_{os.getpid()}_{time.time_ns()}.pt"
        state.save(tmp)
        try:
            tuner.load(tmp)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass


def _evaluate_nll_with_state(tuner: ForwardFineTuner, examples: Sequence[Example], state: SignedLogMaskState | None, *, batch_size: int, max_length: int) -> dict[str, float]:
    _attach_state(tuner, state)
    return tuner.evaluate_nll(examples, batch_size=batch_size, max_length=max_length, use_controller=state is not None)


def _choice_accuracy(tuner: ForwardFineTuner, rows: Sequence[EvalRow], state: SignedLogMaskState | None, *, max_length: int) -> dict[str, float]:
    rows = [r for r in rows if r.choices is not None and r.gold_index is not None]
    if not rows:
        return {"choice_acc": math.nan, "choice_acc_len_norm": math.nan, "choice_acc_sum_loss": math.nan, "choice_n": 0.0}
    _attach_state(tuner, state)
    if tuner.controller is not None:
        tuner.controller.attach()
    correct_norm = 0
    correct_sum = 0
    try:
        for r in rows:
            losses = _choice_losses(tuner, r, max_length=max_length)
            pred_norm = int(min(range(len(losses)), key=lambda i: losses[i]["mean_loss"]))
            pred_sum = int(min(range(len(losses)), key=lambda i: losses[i]["sum_loss"]))
            correct_norm += int(pred_norm == int(r.gold_index))
            correct_sum += int(pred_sum == int(r.gold_index))
    finally:
        if tuner.controller is not None:
            tuner.controller.remove()
    return {
        "choice_acc": correct_norm / max(1, len(rows)),
        "choice_acc_len_norm": correct_norm / max(1, len(rows)),
        "choice_acc_sum_loss": correct_sum / max(1, len(rows)),
        "choice_n": float(len(rows)),
    }


def _choice_losses(tuner: ForwardFineTuner, row: EvalRow, *, max_length: int) -> list[dict[str, float]]:
    losses: list[dict[str, float]] = []
    for choice in row.choices or []:
        batch = make_batch(
            tuner.tokenizer,
            [Example(row.prompt, choice)],
            device=next(tuner.model.parameters()).device,
            max_length=max_length,
        )
        with torch.no_grad():
            out = tuner.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), use_cache=False)
            loss, tokens = _causal_sum_loss_and_tokens(out.logits, batch["labels"])
        losses.append({"sum_loss": float(loss), "tokens": float(tokens), "mean_loss": float(loss) / max(1, int(tokens))})
    if not losses:
        raise ValueError("multiple-choice row contains no choices")
    return losses


def _causal_sum_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss, _tokens = _causal_sum_loss_and_tokens(logits, labels)
    return loss


def _causal_sum_loss_and_tokens(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    if not bool(mask.any()):
        return torch.tensor(0.0, device=logits.device), 0
    tokens = int(mask.sum().item())
    return F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="sum"), tokens


def _retrieval_recall(retriever: MemoryRetriever, task_id: str, rows: Sequence[EvalRow], *, top_k: int) -> dict[str, float]:
    top1 = 0
    topk = 0
    n = 0
    for r in rows:
        hits = retriever.search(r.query, top_k=top_k)
        ids = [h.item.id for h in hits]
        if not ids:
            continue
        n += 1
        top1 += int(ids[0] == task_id)
        topk += int(task_id in ids[:top_k])
    return {"retrieval_n": float(n), "top1_recall": top1 / max(1, n), "topk_recall": topk / max(1, n)}


def _state_from_path(path: str | Path) -> SignedLogMaskState:
    return SignedLogMaskState.load(path, map_location="cpu")




def _token_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)}


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _leakage_audit(out: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        tid = task["task_id"]
        train = load_jsonl_examples(task["train_path"])
        eval_rows = _load_eval_rows(task["eval_path"])
        train_texts = [(ex.prompt + "\n" + ex.completion) for ex in train]
        eval_texts = [(r.prompt + "\n" + r.completion) for r in eval_rows]
        train_hashes = {_stable_text_hash(t) for t in train_texts}
        exact = sum(1 for t in eval_texts if _stable_text_hash(t) in train_hashes)
        train_grams = [_token_ngrams(t) for t in train_texts]
        max_j = []
        for et in eval_texts:
            eg = _token_ngrams(et)
            best = 0.0
            for tg in train_grams:
                denom = len(eg | tg)
                if denom:
                    best = max(best, len(eg & tg) / denom)
            max_j.append(best)
        rows.append({
            "task_id": tid,
            "train_n": len(train),
            "eval_n": len(eval_rows),
            "exact_train_eval_overlaps": exact,
            "max_5gram_jaccard_mean": sum(max_j) / max(1, len(max_j)),
            "max_5gram_jaccard_max": max(max_j) if max_j else 0.0,
        })
    _write_csv(out / "leakage_audit.csv", rows)
    return rows


def _controller_diagnostics(out: Path, states: dict[str, SignedLogMaskState]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid, st in states.items():
        v = dense_gate_vector(st)
        abs_v = v.abs()
        active = abs_v > 0
        rows.append({
            "task_id": tid,
            "n_gates": st.n_gates,
            "dense_dim": int(v.numel()),
            "l2": float(torch.linalg.vector_norm(v).item()),
            "l1": float(abs_v.sum().item()),
            "max_abs": float(abs_v.max().item()) if v.numel() else 0.0,
            "mean_abs_active": float(abs_v[active].mean().item()) if bool(active.any()) else 0.0,
            "clip_frac_active": float((abs_v[active] >= 0.99 * st.max_log_gate).float().mean().item()) if bool(active.any()) else 0.0,
            "max_log_gate": st.max_log_gate,
        })
    # pairwise overlap/cosine
    pair_rows: list[dict[str, Any]] = []
    keys = list(states)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pr = pair_report(states[keys[i]], states[keys[j]])
            pr.update({"task_i": keys[i], "task_j": keys[j]})
            pair_rows.append(pr)
    _write_csv(out / "controller_diagnostics.csv", rows)
    _write_csv(out / "controller_pairwise_geometry.csv", pair_rows)
    return rows


def _retrieval_sweep(out: Path, retriever: MemoryRetriever, task_rows: dict[str, list[EvalRow]], *, max_k: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid, rows_eval in task_rows.items():
        for k in range(1, max_k + 1):
            top1 = topk = n = 0
            scores_gold = []
            for r in rows_eval:
                hits = retriever.search(r.query, top_k=k)
                ids = [h.item.id for h in hits]
                if not ids:
                    continue
                n += 1
                top1 += int(ids[0] == tid)
                topk += int(tid in ids)
                for h in hits:
                    if h.item.id == tid:
                        scores_gold.append(h.score)
            rows.append({
                "task_id": tid,
                "top_k": k,
                "retrieval_n": n,
                "top1_recall": top1 / max(1, n),
                "topk_recall": topk / max(1, n),
                "gold_score_mean_when_retrieved": sum(scores_gold) / max(1, len(scores_gold)),
            })
    _write_csv(out / "retrieval_sweep.csv", rows)
    return rows



def _compose_for_query(
    store: ControllerMemoryStore,
    retriever: MemoryRetriever,
    query: str,
    *,
    top_k: int,
    weighting: str,
    temperature: float,
    max_log_gate: float | None,
) -> tuple[SignedLogMaskState | None, list[MemoryHit]]:
    hits = retriever.search(query, top_k=top_k)
    hits = store.weight_hits(hits, weighting=weighting, temperature=temperature)
    if not hits:
        return None, []
    states = [SignedLogMaskState.load(store.controller_path(h.item), map_location="cpu") for h in hits]
    state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=max_log_gate)
    return state, hits


def _retrieval_signature(hits: Sequence[MemoryHit]) -> tuple[tuple[str, float], ...]:
    return tuple((h.item.id, round(float(h.weight), 6)) for h in hits)


def _evaluate_retrieved_per_example(
    tuner: ForwardFineTuner,
    store: ControllerMemoryStore,
    retriever: MemoryRetriever,
    rows: Sequence[EvalRow],
    *,
    top_k: int,
    weighting: str,
    temperature: float,
    max_log_gate: float,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate with memory retrieved from each individual query.

    Rows that retrieve the same controller signature are grouped so top-1
    retrieval remains cheap.  This is the realistic persistent-memory path; the
    descriptor-query path is kept only as a diagnostic.
    """
    groups: dict[tuple[tuple[str, float], ...], list[tuple[int, EvalRow, list[MemoryHit]]]] = {}
    per_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        raw = retriever.search(row.query, top_k=top_k)
        hits = store.weight_hits(raw, weighting=weighting, temperature=temperature)
        sig = _retrieval_signature(hits)
        groups.setdefault(sig, []).append((i, row, hits))

    total_loss = 0.0
    total_tokens = 0.0
    weighted_token_acc = 0.0
    for sig, items in groups.items():
        hits = items[0][2]
        examples = [Example(r.prompt, r.completion) for _, r, _ in items]
        state = None
        if hits:
            states = [SignedLogMaskState.load(store.controller_path(h.item), map_location="cpu") for h in hits]
            state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=max_log_gate)
        metrics = _evaluate_nll_with_state(tuner, examples, state, batch_size=batch_size, max_length=max_length)
        tokens = float(metrics.get("tokens", 0.0))
        total_loss += float(metrics.get("nll", 0.0)) * tokens
        total_tokens += tokens
        weighted_token_acc += float(metrics.get("token_acc", 0.0)) * tokens
        ids = [h.item.id for h in hits]
        weights = [h.weight for h in hits]
        for idx, row, _ in items:
            per_rows.append({
                "row": idx,
                "retrieved_ids": ";".join(ids),
                "retrieved_weights": ";".join(f"{w:.4f}" for w in weights),
            })
    return {
        "nll": total_loss / max(1.0, total_tokens),
        "token_acc": weighted_token_acc / max(1.0, total_tokens),
        "tokens": total_tokens,
    }, sorted(per_rows, key=lambda r: int(r["row"]))


def _retrieval_method_comparison(out: Path, store: ControllerMemoryStore, args, task_rows: dict[str, list[EvalRow]]) -> list[dict[str, Any]]:
    methods = [x.strip() for x in str(args.retrieval_compare_methods).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for method in methods:
        try:
            retr = build_memory_retriever(
                store,
                method=method,
                embedding_model=args.embedding_model,
                embedding_device=args.embedding_device,
                hybrid_alpha=args.hybrid_alpha,
                batch_size=args.embedding_batch_size,
            )
        except Exception as e:
            rows.append({"retriever": method, "error": str(e)})
            continue
        for tid, eval_rows in task_rows.items():
            rec = _retrieval_recall(retr, tid, eval_rows, top_k=args.top_k)
            rows.append({"retriever": method, "task_id": tid, **rec})
    _write_csv(out / "retrieval_method_comparison.csv", rows)
    return rows

def _selectivity_summary(out: Path, cross_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in cross_rows:
        by_task.setdefault(str(r["eval_task"]), []).append(r)
    rows: list[dict[str, Any]] = []
    for task, rs in by_task.items():
        gold = next((r for r in rs if int(r.get("is_gold", 0)) == 1), None)
        wrong = [r for r in rs if int(r.get("is_gold", 0)) == 0]
        best_wrong = min(wrong, key=lambda r: float(r["memory_nll"])) if wrong else None
        worst_wrong = max(wrong, key=lambda r: float(r["memory_nll"])) if wrong else None
        base = float(gold["base_nll"]) if gold else (float(rs[0]["base_nll"]) if rs else math.nan)
        gold_nll = float(gold["memory_nll"]) if gold else math.nan
        bw_nll = float(best_wrong["memory_nll"]) if best_wrong else math.nan
        ww_nll = float(worst_wrong["memory_nll"]) if worst_wrong else math.nan
        rows.append({
            "eval_task": task,
            "base_nll": base,
            "gold_memory_nll": gold_nll,
            "gold_delta": gold_nll - base,
            "best_wrong_memory": best_wrong["memory_id"] if best_wrong else "",
            "best_wrong_nll": bw_nll,
            "best_wrong_delta": bw_nll - base if best_wrong else math.nan,
            "worst_wrong_memory": worst_wrong["memory_id"] if worst_wrong else "",
            "worst_wrong_nll": ww_nll,
            "worst_wrong_delta": ww_nll - base if worst_wrong else math.nan,
            "gold_vs_best_wrong_margin": bw_nll - gold_nll if best_wrong else math.nan,
        })
    _write_csv(out / "selectivity_summary.csv", rows)
    return rows

def _choice_accuracy_retrieved_per_example(
    tuner: ForwardFineTuner,
    store: ControllerMemoryStore,
    retriever: MemoryRetriever,
    rows: Sequence[EvalRow],
    *,
    top_k: int,
    weighting: str,
    temperature: float,
    max_log_gate: float,
    max_length: int,
) -> dict[str, float]:
    rows = [r for r in rows if r.choices is not None and r.gold_index is not None]
    if not rows:
        return {"choice_acc": math.nan, "choice_acc_len_norm": math.nan, "choice_acc_sum_loss": math.nan, "choice_n": 0.0}
    correct_norm = 0
    correct_sum = 0
    for r in rows:
        raw = retriever.search(r.query, top_k=top_k)
        hits = store.weight_hits(raw, weighting=weighting, temperature=temperature)
        state = None
        if hits:
            states = [SignedLogMaskState.load(store.controller_path(h.item), map_location="cpu") for h in hits]
            state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=max_log_gate)
        _attach_state(tuner, state)
        if tuner.controller is not None:
            tuner.controller.attach()
        try:
            losses = _choice_losses(tuner, r, max_length=max_length)
            pred_norm = int(min(range(len(losses)), key=lambda i: losses[i]["mean_loss"]))
            pred_sum = int(min(range(len(losses)), key=lambda i: losses[i]["sum_loss"]))
            correct_norm += int(pred_norm == int(r.gold_index))
            correct_sum += int(pred_sum == int(r.gold_index))
        finally:
            if tuner.controller is not None:
                tuner.controller.remove()
    return {
        "choice_acc": correct_norm / max(1, len(rows)),
        "choice_acc_len_norm": correct_norm / max(1, len(rows)),
        "choice_acc_sum_loss": correct_sum / max(1, len(rows)),
        "choice_n": float(len(rows)),
    }


def cmd_eval(args) -> None:
    out = Path(args.out)
    manifest = _read_json(out / "manifest.json")
    store = ControllerMemoryStore(out / "memory_store")
    retriever = build_memory_retriever(
        store,
        method=args.retrieval_method,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        hybrid_alpha=args.hybrid_alpha,
        batch_size=args.embedding_batch_size,
    )
    model, tok = _load_model(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        local_files_only=getattr(args, "local_files_only", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    tuner = ForwardFineTuner(
        model,
        tok,
        gates=args.gates,
        layers=args.layers,
        max_log_gate=args.max_log_gate,
        model_name=args.model,
        model_revision=getattr(args, "revision", None),
        tokenizer_name=args.model,
        tokenizer_revision=getattr(args, "tokenizer_revision", None) or getattr(args, "revision", None),
    )

    states: dict[str, SignedLogMaskState] = {}
    task_rows: dict[str, list[EvalRow]] = {}
    task_examples: dict[str, list[Example]] = {}
    task_desc: dict[str, str] = {}
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        task_desc[task_id] = task["descriptor"]
        rows = _load_eval_rows(task["eval_path"])
        task_rows[task_id] = rows
        task_examples[task_id] = _rows_to_examples(rows)
        states[task_id] = _state_from_path(out / "controllers" / f"{task_id}.pt")

    _leakage_audit(out, manifest)
    _controller_diagnostics(out, states)

    all_state = compose_states(list(states.values()), weights=[1.0] * len(states), max_log_gate=args.max_log_gate)
    all_path = out / "memory_store" / "composed_all.pt"
    all_state.save(all_path)
    save_report(out / "composition_report.json", composition_report([str(out / "controllers" / f"{k}.pt") for k in states], list(states.values())))

    _retrieval_sweep(out, retriever, task_rows, max_k=min(5, max(1, len(states))))
    _retrieval_method_comparison(out, store, args, task_rows)

    summary_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    per_example_rows_all: list[dict[str, Any]] = []

    for task_id, rows in task_rows.items():
        examples = task_examples[task_id]
        base = _evaluate_nll_with_state(tuner, examples, None, batch_size=args.eval_batch_size, max_length=args.max_length)
        base_choice = _choice_accuracy(tuner, rows, None, max_length=args.max_length) if args.choice_acc else {}
        retrieval = _retrieval_recall(retriever, task_id, rows, top_k=args.top_k)
        retrieval_rows.append({"task_id": task_id, "retriever": args.retrieval_method, **retrieval})

        gold = _evaluate_nll_with_state(tuner, examples, states[task_id], batch_size=args.eval_batch_size, max_length=args.max_length)
        gold_choice = _choice_accuracy(tuner, rows, states[task_id], max_length=args.max_length) if args.choice_acc else {}

        # Diagnostic: retrieval from task descriptor. This is not the real per-query path.
        desc_state, desc_hits = _compose_for_query(
            store, retriever, task_desc[task_id],
            top_k=args.top_k, weighting=args.weighting, temperature=args.temperature, max_log_gate=args.max_log_gate,
        )
        retrieved_desc = _evaluate_nll_with_state(tuner, examples, desc_state, batch_size=args.eval_batch_size, max_length=args.max_length)
        retrieved_desc_choice = _choice_accuracy(tuner, rows, desc_state, max_length=args.max_length) if args.choice_acc else {}

        # Realistic persistent-memory path: retrieve memory from each individual query/prompt.
        retrieved_query, per_rows = _evaluate_retrieved_per_example(
            tuner, store, retriever, rows,
            top_k=args.top_k, weighting=args.weighting, temperature=args.temperature,
            max_log_gate=args.max_log_gate, batch_size=args.eval_batch_size, max_length=args.max_length,
        )
        retrieved_query_choice = _choice_accuracy_retrieved_per_example(
            tuner, store, retriever, rows,
            top_k=args.top_k, weighting=args.weighting, temperature=args.temperature,
            max_log_gate=args.max_log_gate, max_length=args.max_length,
        ) if args.choice_acc else {}
        for rr in per_rows:
            rr.update({"task_id": task_id, "gold_retrieved": int(task_id in str(rr.get("retrieved_ids", "")).split(";"))})
        per_example_rows_all.extend(per_rows)

        composed = _evaluate_nll_with_state(tuner, examples, all_state, batch_size=args.eval_batch_size, max_length=args.max_length)
        composed_choice = _choice_accuracy(tuner, rows, all_state, max_length=args.max_length) if args.choice_acc else {}

        condition_rows = [
            ("base", base, base_choice, "", ""),
            ("gold_memory", gold, gold_choice, "", ""),
            ("retrieved_descriptor_memory", retrieved_desc, retrieved_desc_choice, ";".join(h.item.id for h in desc_hits), ";".join(f"{h.weight:.4f}" for h in desc_hits)),
            ("retrieved_query_memory", retrieved_query, retrieved_query_choice, "per-example", "per-example"),
            ("composed_all_memories", composed, composed_choice, "", ""),
        ]
        for condition, metrics, cmetrics, hit_ids, hit_weights in condition_rows:
            summary_rows.append({
                "task_id": task_id,
                "condition": condition,
                "retriever": args.retrieval_method,
                "retrieval_text_mode": args.memory_text_mode,
                "nll": metrics.get("nll"),
                "token_acc": metrics.get("token_acc"),
                "tokens": metrics.get("tokens"),
                "choice_acc": cmetrics.get("choice_acc", math.nan),
                "choice_acc_len_norm": cmetrics.get("choice_acc_len_norm", math.nan),
                "choice_acc_sum_loss": cmetrics.get("choice_acc_sum_loss", math.nan),
                "choice_n": cmetrics.get("choice_n", 0.0),
                "nll_delta_vs_base": metrics.get("nll") - base.get("nll"),
                "retrieved_ids": hit_ids,
                "retrieved_weights": hit_weights,
                **retrieval,
            })

        for mem_id, st in states.items():
            m = _evaluate_nll_with_state(tuner, examples, st, batch_size=args.eval_batch_size, max_length=args.max_length)
            cross_rows.append({
                "eval_task": task_id,
                "memory_id": mem_id,
                "base_nll": base["nll"],
                "memory_nll": m["nll"],
                "delta_vs_base": m["nll"] - base["nll"],
                "is_gold": int(mem_id == task_id),
            })

    _write_csv(out / "summary.csv", summary_rows)
    _write_csv(out / "cross_task_nll.csv", cross_rows)
    _write_csv(out / "retrieval_recall.csv", retrieval_rows)
    _write_csv(out / "per_example_retrieved_memory.csv", per_example_rows_all)
    _selectivity_summary(out, cross_rows)
    print(json.dumps({
        "summary": str(out / "summary.csv"),
        "cross_task_nll": str(out / "cross_task_nll.csv"),
        "retrieval_recall": str(out / "retrieval_recall.csv"),
        "retrieval_method_comparison": str(out / "retrieval_method_comparison.csv"),
        "per_example_retrieved_memory": str(out / "per_example_retrieved_memory.csv"),
    }, indent=2), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Persistent NTK-mirror controller memory benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_model(q):
        q.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
        q.add_argument("--revision", default=os.environ.get("REVISION") or None)
        q.add_argument("--tokenizer-revision", default=os.environ.get("TOKENIZER_REVISION") or None)
        q.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR") or None)
        q.add_argument("--local-files-only", action="store_true", default=os.environ.get("LOCAL_FILES_ONLY", "0") == "1")
        q.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("TRUST_REMOTE_CODE", "0") == "1")
        q.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        q.add_argument("--dtype", default=os.environ.get("DTYPE", "bf16"), choices=["auto", "bf16", "fp16", "fp32"])
        q.add_argument("--gates", type=int, default=int(os.environ.get("GATES", "21000")))
        q.add_argument("--layers", default=os.environ.get("LAYERS", "all"))
        q.add_argument("--max-log-gate", type=float, default=float(os.environ.get("MAX_LOG_GATE", "0.05")))
        q.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "1024")))

    prep = sub.add_parser("prepare")
    prep.add_argument("--out", required=True)
    prep.add_argument("--tasks", default=os.environ.get("TASKS", "gsm8k,mbpp,hellaswag,arc_easy,boolq"))
    prep.add_argument("--train-per-task", type=int, default=int(os.environ.get("TRAIN_PER_TASK", "128")))
    prep.add_argument("--eval-per-task", type=int, default=int(os.environ.get("EVAL_PER_TASK", "64")))
    prep.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    prep.set_defaults(func=cmd_prepare)

    fit = sub.add_parser("fit-one")
    common_model(fit)
    fit.add_argument("--out", required=True)
    fit.add_argument("--task", required=True)
    fit.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "240")))
    fit.add_argument("--lr", type=float, default=float(os.environ.get("LR", "5e-3")))
    fit.add_argument("--l2", type=float, default=float(os.environ.get("L2", "1e-5")))
    fit.add_argument("--score-batches", type=int, default=int(os.environ.get("SCORE_BATCHES", "16")))
    fit.add_argument("--train-batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "8")))
    fit.add_argument("--quiet", action="store_true")
    fit.add_argument("--overwrite", action="store_true")
    fit.set_defaults(func=cmd_fit_one)

    reg = sub.add_parser("register")
    reg.add_argument("--out", required=True)
    reg.add_argument("--memory-text-mode", default=os.environ.get("MEMORY_TEXT_MODE", "descriptor_prompts"), choices=["descriptor", "descriptor_prompts", "descriptor_full"])
    reg.add_argument("--memory-train-snippets", type=int, default=int(os.environ.get("MEMORY_TRAIN_SNIPPETS", "32")))
    reg.set_defaults(func=cmd_register)

    ev = sub.add_parser("eval")
    common_model(ev)
    ev.add_argument("--out", required=True)
    ev.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "8")))
    ev.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "1")))
    ev.add_argument("--weighting", default=os.environ.get("WEIGHTING", "softmax"), choices=["softmax", "score", "uniform"])
    ev.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.25")))
    ev.add_argument("--choice-acc", action="store_true", default=os.environ.get("CHOICE_ACC", "1") != "0")
    ev.add_argument("--retrieval-method", default=os.environ.get("RETRIEVAL_METHOD", "hybrid"), choices=["lexical", "embedding", "hybrid"])
    ev.add_argument("--retrieval-compare-methods", default=os.environ.get("RETRIEVAL_COMPARE_METHODS", "lexical,embedding,hybrid"))
    ev.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ev.add_argument("--embedding-device", default=os.environ.get("EMBEDDING_DEVICE", "cpu"))
    ev.add_argument("--embedding-batch-size", type=int, default=int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")))
    ev.add_argument("--hybrid-alpha", type=float, default=float(os.environ.get("HYBRID_ALPHA", "0.65")))
    ev.add_argument("--memory-text-mode", default=os.environ.get("MEMORY_TEXT_MODE", "descriptor_prompts"))
    ev.set_defaults(func=cmd_eval)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
