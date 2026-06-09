from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import torch

from .artifacts import doctor_model, lint_controller, write_controller_card
from .compose import compose_states, composition_plan, composition_report, save_report
from .controller import ForwardFineTuner, SignedLogMaskState
from .data import load_jsonl_examples, save_jsonl_examples, tiny_arithmetic_eval, tiny_arithmetic_train
from .memory import DEFAULT_MIN_SCORE, ControllerMemoryStore
from .isr import (
    HFChoiceVerifierBackend,
    KVDeltaBayesNTKBackendAdapter,
    SUPPORT_QUERY,
    YESNO,
    atomic_write_json,
    load_evidence_claims_jsonl,
    load_isr_dataset,
    run_isr_auc,
)


def _set_seed(seed: int | None, *, deterministic: bool = False) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_hf(
    model_name: str,
    *,
    device: str,
    dtype: str,
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
    else:  # pragma: no cover - argparse guards this
        raise ValueError("dtype must be auto, bf16, fp16, or fp32")

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
    return model, tok


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision", default=None, help="model revision/tag/commit to load")
    p.add_argument("--tokenizer-revision", default=None, help="tokenizer revision; defaults to --revision")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true", help="opt in to executing remote model/tokenizer code")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--torch-deterministic", action="store_true")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chat-template", default="auto", choices=["auto", "none"], help="how to render JSONL messages rows")
    p.add_argument("--loss-on", default="assistant", choices=["assistant", "all"], help="chat loss mask for messages rows")


def _add_controller_load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="load a controller even if saved model/tokenizer identity differs; use only after manual compatibility checks",
    )


def _load_controller_into_tuner(tuner: ForwardFineTuner, args, path: str | Path) -> None:
    tuner.load(path, allow_model_mismatch=getattr(args, "allow_model_mismatch", False))


def _make_tuner(args) -> ForwardFineTuner:
    _set_seed(getattr(args, "seed", None), deterministic=getattr(args, "torch_deterministic", False))
    model, tok = _load_hf(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        local_files_only=getattr(args, "local_files_only", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    return ForwardFineTuner(
        model,
        tok,
        gates=getattr(args, "gates", 5000),
        layers=getattr(args, "layers", "all"),
        max_log_gate=getattr(args, "max_log_gate", 0.05),
        hook_site=getattr(args, "hook_site", "layer_output"),
        model_name=args.model,
        model_revision=getattr(args, "revision", None),
        tokenizer_name=args.model,
        tokenizer_revision=getattr(args, "tokenizer_revision", None) or getattr(args, "revision", None),
    )


def _load_examples_for_args(path: str | Path, args, tokenizer=None):
    return load_jsonl_examples(
        path,
        tokenizer=tokenizer,
        chat_template=getattr(args, "chat_template", "auto"),
        loss_on=getattr(args, "loss_on", "assistant"),
    )


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_weights(s: str | None, n: int) -> list[float]:
    if s is None or not s.strip():
        return [1.0] * n
    vals = [float(x) for x in s.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"expected {n} weights, got {len(vals)}")
    return vals


def _parse_param_filter(s: str | None) -> list[str] | None:
    if s is None or not str(s).strip():
        return None
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    return vals or None


def _memory_text_from_examples(examples) -> str:
    parts = []
    for ex in examples[:8]:
        parts.append((ex.prompt + " " + ex.completion).strip())
    return "\n".join(parts)


def _write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _read_optional_json(path: str | None) -> dict | None:
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def _args_manifest(args, *, exclude: set[str] | None = None) -> dict[str, Any]:
    exclude = set(exclude or set())
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in exclude or key == "func":
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = str(value)
    return out


def _training_kwargs(args) -> dict[str, Any]:
    patience = int(getattr(args, "early_stop_patience", 0) or 0)
    legacy_kl = getattr(args, "retain_kl_weight", None)
    kl_to_base = float(legacy_kl if legacy_kl is not None else getattr(args, "kl_to_base", 0.0))
    return {
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "score_batches": args.score_batches,
        "max_length": args.max_length,
        "weight_decay": getattr(args, "weight_decay", 0.0),
        "l2": getattr(args, "l2", 1e-5),
        "validation_interval": getattr(args, "eval_every", None),
        "early_stop_patience": patience,
        "select_best_on_validation": not getattr(args, "no_select_best", False),
        "retain_weight": getattr(args, "retain_weight", 0.0),
        "kl_to_base": kl_to_base,
        "verbose": not getattr(args, "quiet", False),
    }


@torch.no_grad()
def _generate_base_model(model, tokenizer, prompt: str, *, max_new_tokens: int, return_full_text: bool) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(encoded["input_ids"].shape[-1])
    model.eval()
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        do_sample=False,
    )
    ids = output_ids[0] if return_full_text else output_ids[0, prompt_len:]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------- normal controller commands ----------


def cmd_fit(args) -> None:
    tuner = _make_tuner(args)
    examples = _load_examples_for_args(args.train, args, tuner.tokenizer)
    validation = _load_examples_for_args(args.validation, args, tuner.tokenizer) if args.validation else None
    retain = _load_examples_for_args(args.retain, args, tuner.tokenizer) if args.retain else None
    before = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    validation_before = None
    if validation:
        validation_before = tuner.evaluate_nll(validation, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit(
        examples,
        validation_examples=validation,
        retain_examples=retain,
        **_training_kwargs(args),
    )
    after = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    validation_after = None
    if validation:
        validation_after = tuner.evaluate_nll(validation, batch_size=args.batch_size, max_length=args.max_length)
    metadata = {
        "train": {"path": args.train, "examples": len(examples), "before": before, "after": after},
        "validation": {"path": args.validation, "examples": len(validation or []), "before": validation_before, "after": validation_after},
        "retain": {"path": args.retain, "examples": len(retain or []), "retain_weight": getattr(args, "retain_weight", 0.0), "kl_to_base": getattr(args, "retain_kl_weight", None) if getattr(args, "retain_kl_weight", None) is not None else getattr(args, "kl_to_base", 0.0)},
        "fit": fit_stats,
        "validation_history": getattr(tuner, "_last_validation_history", []),
        "data_format": {"chat_template": args.chat_template, "loss_on": args.loss_on},
    }
    tuner.save(args.out, metadata=metadata)
    manifest_path = str(Path(args.out).with_suffix(".manifest.json"))
    tuner.write_manifest(manifest_path)
    card_path = str(Path(args.out).with_suffix(".card.md"))
    write_controller_card(
        args.out,
        card_path,
        eval_report={"train": metadata["train"], "validation": metadata["validation"], "fit": fit_stats},
        intended_use="Training run artifact generated by ntkmirror fit.",
    )
    print(
        json.dumps(
            {
                "controller": args.out,
                "manifest": manifest_path,
                "card": card_path,
                "train_examples": len(examples),
                "validation_examples": len(validation or []),
                "retain_examples": len(retain or []),
                "before": before,
                "after": after,
                "validation_before": validation_before,
                "validation_after": validation_after,
                "fit": fit_stats,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def cmd_fit_dual(args) -> None:
    tuner = _make_tuner(args)
    examples = _load_examples_for_args(args.train, args, tuner.tokenizer)
    before = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit_dual(
        examples,
        steps=args.steps,
        target_step_size=args.target_step_size,
        apply_scale=args.apply_scale,
        batch_size=args.batch_size,
        score_batches=args.score_batches,
        max_length=args.max_length,
        projection=args.projection,
        top_k=args.top_k,
        ridge=args.ridge,
        cg_iters=args.cg_iters,
        cg_tol=args.cg_tol,
        fd_eps=args.fd_eps,
        metric=args.metric,
        metric_eps=args.metric_eps,
        jvp_mode=args.jvp_mode,
        param_name_substrings=_parse_param_filter(args.param_filter),
        max_target_parameters=args.max_target_parameters,
        verbose=not args.quiet,
    )
    after = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    tuner.save(args.out, metadata={"train": {"path": args.train, "examples": len(examples), "before": before, "after": after}, "fit_dual": fit_stats})
    manifest_path = str(Path(args.out).with_suffix(".manifest.json"))
    tuner.write_manifest(manifest_path)
    result = {"controller": args.out, "manifest": manifest_path, "train_examples": len(examples), "before": before, "after": after, "fit_dual": fit_stats}
    if args.report:
        _write_json(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def cmd_secant(args) -> None:
    tuner = _make_tuner(args)
    examples = _load_examples_for_args(args.eval, args, tuner.tokenizer)
    _load_controller_into_tuner(tuner, args, args.controller)
    report = tuner.secant_diagnostics(
        examples,
        eps=args.fd_eps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        projection=args.projection,
        top_k=args.top_k,
    )
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_dual_diagnose(args) -> None:
    tuner = _make_tuner(args)
    support = _load_examples_for_args(args.support, args, tuner.tokenizer)
    calibration = _load_examples_for_args(args.calibration, args, tuner.tokenizer) if args.calibration else None
    if args.controller:
        _load_controller_into_tuner(tuner, args, args.controller)
        init = {"source": args.controller}
    else:
        init = tuner.initialize_controller(support, score_batches=args.score_batches, batch_size=args.batch_size, max_length=args.max_length)
    report = tuner.dual_projection_diagnostics(
        support,
        calibration,
        batch_size=args.batch_size,
        max_length=args.max_length,
        projection=args.projection,
        top_k=args.top_k,
        target_step_size=args.target_step_size,
        ridge=args.ridge,
        cg_iters=args.cg_iters,
        cg_tol=args.cg_tol,
        fd_eps=args.fd_eps,
        metric=args.metric,
        metric_eps=args.metric_eps,
        jvp_mode=args.jvp_mode,
        param_name_substrings=_parse_param_filter(args.param_filter),
        max_target_parameters=args.max_target_parameters,
    )
    report["basis"] = init
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_eval(args) -> None:
    tuner = _make_tuner(args)
    examples = _load_examples_for_args(args.eval, args, tuner.tokenizer)
    base = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    ctrl = None
    if args.controller:
        _load_controller_into_tuner(tuner, args, args.controller)
        ctrl = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    result = {"examples": len(examples), "base": base, "controller": ctrl}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def cmd_generate(args) -> None:
    tuner = _make_tuner(args)
    _load_controller_into_tuner(tuner, args, args.controller)
    print(
        tuner.generate(args.prompt, max_new_tokens=args.max_new_tokens, return_full_text=not args.new_text_only, do_sample=False),
        flush=True,
    )


def cmd_demo(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "math_train.jsonl"
    eval_path = out_dir / "math_eval.jsonl"
    ctrl_path = out_dir / "controller.pt"
    save_jsonl_examples(train_path, tiny_arithmetic_train())
    save_jsonl_examples(eval_path, tiny_arithmetic_eval())
    print(f"wrote {train_path} and {eval_path}", flush=True)

    tuner = _make_tuner(args)
    train = _load_examples_for_args(train_path, args, tuner.tokenizer)
    eval_examples = _load_examples_for_args(eval_path, args, tuner.tokenizer)
    base_train = tuner.evaluate_nll(train, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    base_eval = tuner.evaluate_nll(eval_examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit(train, steps=args.steps, lr=args.lr, batch_size=args.batch_size, score_batches=args.score_batches, max_length=args.max_length, verbose=True)
    ctrl_train = tuner.evaluate_nll(train, batch_size=args.batch_size, max_length=args.max_length)
    ctrl_eval = tuner.evaluate_nll(eval_examples, batch_size=args.batch_size, max_length=args.max_length)
    tuner.save(ctrl_path, metadata={"demo": True, "fit": fit_stats})
    print(json.dumps({"controller": str(ctrl_path), "train_base": base_train, "train_controller": ctrl_train, "eval_base": base_eval, "eval_controller": ctrl_eval, "fit": fit_stats}, indent=2, sort_keys=True), flush=True)


def cmd_compose(args) -> None:
    states = [SignedLogMaskState.load(path, map_location="cpu") for path in args.controllers]
    weights = _parse_weights(args.weights, len(states))
    composed = compose_states(states, weights=weights, max_log_gate=args.max_log_gate)
    composed.save(args.out)
    report = composition_plan(args.controllers, states, weights=weights, max_log_gate=args.max_log_gate)
    report["composition"] = {"out": args.out, "weights": weights, "n_gates": composed.n_gates, "max_log_gate": composed.max_log_gate}
    if args.report:
        save_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_compose_plan(args) -> None:
    states = [SignedLogMaskState.load(path, map_location="cpu") for path in args.controllers]
    weights = _parse_weights(args.weights, len(states))
    report = composition_plan(args.controllers, states, weights=weights, max_log_gate=args.max_log_gate)
    if args.out:
        save_report(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_inspect(args) -> None:
    states = [SignedLogMaskState.load(path, map_location="cpu") for path in args.controllers]
    report = composition_report(args.controllers, states)
    if args.out:
        save_report(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_lint(args) -> None:
    report = lint_controller(
        args.controller,
        require_revision=args.require_revision,
        max_gates=args.max_gates or None,
    )
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_card(args) -> None:
    metrics = _read_optional_json(args.metrics)
    report = write_controller_card(
        args.controller,
        args.out,
        eval_report=metrics,
        intended_use=args.intended_use,
        limitations=args.limitations,
    )
    report = {"controller": args.controller, **report}
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_doctor(args) -> None:
    model, tok = _load_hf(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        local_files_only=getattr(args, "local_files_only", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    report = doctor_model(model, tok, layers=getattr(args, "layers", "all"))
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


# ---------- persistent memory commands ----------


def cmd_memory_add(args) -> None:
    store = ControllerMemoryStore(args.store)
    tags = _split_csv(args.tags)
    meta = {"source": args.train or args.controller}
    if args.controller:
        if not args.text:
            raise ValueError("--text is required when registering an existing --controller")
        item = store.add_controller(
            memory_id=args.id,
            controller_path=args.controller,
            text=args.text,
            tags=tags,
            metadata=meta,
            overwrite=args.overwrite,
            namespace=args.namespace,
        )
        print(json.dumps({"added": item.id, "item": item.__dict__}, indent=2, sort_keys=True), flush=True)
        return

    if not args.train:
        raise ValueError("memory add requires either --train or --controller")
    tuner = _make_tuner(args)
    examples = _load_examples_for_args(args.train, args, tuner.tokenizer)
    validation = _load_examples_for_args(args.validation, args, tuner.tokenizer) if args.validation else None
    retain = _load_examples_for_args(args.retain, args, tuner.tokenizer) if args.retain else None
    text = args.text or _memory_text_from_examples(examples)
    before = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    validation_before = tuner.evaluate_nll(validation, batch_size=args.batch_size, max_length=args.max_length, use_controller=False) if validation else None
    fit_stats = tuner.fit(examples, validation_examples=validation, retain_examples=retain, **_training_kwargs(args))
    after = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    validation_after = tuner.evaluate_nll(validation, batch_size=args.batch_size, max_length=args.max_length) if validation else None
    with tempfile.NamedTemporaryFile(suffix=".pt", prefix=".tmp-memory-add-", dir=store.root, delete=False) as f:
        tmp_controller = Path(f.name)
    fit_meta = {
        "train": {"path": args.train, "examples": len(examples), "before": before, "after": after},
        "validation": {"path": args.validation, "examples": len(validation or []), "before": validation_before, "after": validation_after},
        "retain": {"path": args.retain, "examples": len(retain or []), "retain_weight": getattr(args, "retain_weight", 0.0), "kl_to_base": getattr(args, "retain_kl_weight", None) if getattr(args, "retain_kl_weight", None) is not None else getattr(args, "kl_to_base", 0.0)},
        "fit": fit_stats,
        "validation_history": getattr(tuner, "_last_validation_history", []),
    }
    tuner.save(tmp_controller, metadata=fit_meta)
    meta.update({"before": before, "after": after, "validation_before": validation_before, "validation_after": validation_after, "fit": fit_stats})
    try:
        item = store.add_controller(
            memory_id=args.id,
            controller_path=tmp_controller,
            text=text,
            tags=tags,
            metadata=meta,
            overwrite=args.overwrite,
            namespace=args.namespace,
        )
    finally:
        try:
            tmp_controller.unlink()
        except OSError:
            pass
    print(json.dumps({"added": item.id, "namespace": item.namespace, "version": item.version, "before": before, "after": after, "validation_before": validation_before, "validation_after": validation_after, "fit": fit_stats}, indent=2, sort_keys=True), flush=True)


def cmd_memory_list(args) -> None:
    store = ControllerMemoryStore(args.store)
    items = store.list_items(namespace=args.namespace, include_deleted=args.include_deleted)
    print(json.dumps({"items": [x.__dict__ for x in items]}, indent=2, sort_keys=True), flush=True)


def cmd_memory_delete(args) -> None:
    store = ControllerMemoryStore(args.store)
    store.delete(args.id, namespace=args.namespace, hard=not args.soft)
    print(json.dumps({"deleted": args.id, "namespace": args.namespace or "default", "hard": not args.soft}, indent=2, sort_keys=True), flush=True)


def cmd_memory_rollback(args) -> None:
    store = ControllerMemoryStore(args.store)
    item = store.rollback(args.id, namespace=args.namespace, version=args.version)
    print(json.dumps({"rolled_back": item.__dict__}, indent=2, sort_keys=True), flush=True)


def cmd_memory_audit(args) -> None:
    store = ControllerMemoryStore(args.store)
    report = store.audit(namespace=args.namespace)
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_memory_search(args) -> None:
    store = ControllerMemoryStore(args.store)
    hits = store.search(args.query, top_k=args.top_k, tag=args.tag, namespace=args.namespace, min_score=args.min_score)
    hits = store.weight_hits(hits, weighting=args.weighting, temperature=args.temperature)
    print(json.dumps({"query": args.query, "hits": [h.to_dict() for h in hits]}, indent=2, sort_keys=True), flush=True)


def cmd_memory_compose(args) -> None:
    store = ControllerMemoryStore(args.store)
    state, hits = store.compose_for_query(
        args.query,
        top_k=args.top_k,
        weighting=args.weighting,
        temperature=args.temperature,
        max_log_gate=args.max_log_gate,
        tag=args.tag,
        namespace=args.namespace,
        min_score=args.min_score,
    )
    state.save(args.out)
    report = {"query": args.query, "out": args.out, "n_gates": state.n_gates, "max_log_gate": state.max_log_gate, "hits": [h.to_dict() for h in hits]}
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def cmd_memory_generate(args) -> None:
    store = ControllerMemoryStore(args.store)
    raw_hits = store.search(args.query, top_k=args.top_k, tag=args.tag, namespace=args.namespace, min_score=args.min_score)
    hits = store.weight_hits(raw_hits, weighting=args.weighting, temperature=args.temperature)
    tuner = _make_tuner(args)
    used_controller = False
    if hits:
        states = [SignedLogMaskState.load(store.controller_path(h.item), map_location="cpu") for h in hits]
        state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=args.compose_max_log_gate)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = Path(f.name)
        try:
            state.save(tmp)
            _load_controller_into_tuner(tuner, args, tmp)
            text = tuner.generate(args.prompt or args.query, max_new_tokens=args.max_new_tokens, return_full_text=not args.new_text_only, do_sample=False)
            used_controller = True
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    else:
        text = _generate_base_model(tuner.model, tuner.tokenizer, args.prompt or args.query, max_new_tokens=args.max_new_tokens, return_full_text=not args.new_text_only)
    print(json.dumps({"query": args.query, "hits": [h.to_dict() for h in hits], "used_controller": used_controller, "text": text}, indent=2, sort_keys=True), flush=True)


def cmd_memory_eval(args) -> None:
    store = ControllerMemoryStore(args.store)
    tuner = _make_tuner(args)
    examples = _load_examples_for_args(args.eval, args, tuner.tokenizer)
    raw_hits = store.search(args.query, top_k=args.top_k, tag=args.tag, namespace=args.namespace, min_score=args.min_score)
    hits = store.weight_hits(raw_hits, weighting=args.weighting, temperature=args.temperature)
    base = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    used_controller = False
    if hits:
        states = [SignedLogMaskState.load(store.controller_path(h.item), map_location="cpu") for h in hits]
        state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=args.compose_max_log_gate)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = Path(f.name)
        try:
            state.save(tmp)
            _load_controller_into_tuner(tuner, args, tmp)
            ctrl = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
            used_controller = True
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    else:
        ctrl = dict(base)
    result = {"query": args.query, "hits": [h.to_dict() for h in hits], "used_controller": used_controller, "base": base, "controller": ctrl}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


# ---------- ISR verifier / KV order-debias benchmark ----------


def cmd_isr_auc(args) -> None:
    _set_seed(getattr(args, "seed", None), deterministic=getattr(args, "torch_deterministic", False))
    if args.data_jsonl:
        examples = load_evidence_claims_jsonl(args.data_jsonl, cap_spans=args.cap_spans, max_chars=args.max_chars)
        dataset_name = str(args.data_jsonl)
    else:
        examples = load_isr_dataset(args.dataset, args.n, cap_spans=args.cap_spans, max_chars=args.max_chars)
        dataset_name = args.dataset

    if args.backend == "auto":
        backend_name = "kv-delta-bayes-ntk" if args.fit_controller else "hf"
    else:
        backend_name = args.backend
    if args.fit_controller and backend_name != "kv-delta-bayes-ntk":
        raise ValueError("--fit-controller requires --backend kv-delta-bayes-ntk or --backend auto")

    if backend_name == "hf":
        model, tok = _load_hf(
            args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            local_files_only=getattr(args, "local_files_only", False),
            cache_dir=getattr(args, "cache_dir", None),
        )
        backend = HFChoiceVerifierBackend(
            model,
            tok,
            add_special_tokens=args.add_special_tokens,
            length_normalize_choices=args.length_normalize_choices,
        )
    else:
        backend = KVDeltaBayesNTKBackendAdapter.from_pretrained(
            args.model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            revision=getattr(args, "revision", None),
            local_files_only=getattr(args, "local_files_only", False),
            cache_dir=getattr(args, "cache_dir", None),
        )

    choices = (args.yes_choice, args.no_choice)

    def progress(done: int, total: int, elapsed: float) -> None:
        if args.progress_every > 0 and (done % args.progress_every == 0 or done == total):
            print(f"isr-auc {dataset_name} {done}/{total} ({elapsed:.0f}s)", flush=True)

    result = run_isr_auc(
        backend,
        examples,
        seed=args.seed or 0,
        num_orderings=args.num_orderings,
        dispersion_penalty=args.dispersion_penalty,
        leak=args.leak,
        shard=args.shard,
        nshards=args.nshards,
        query=args.query,
        choices=choices,
        fit_controller=args.fit_controller,
        use_layers=args.use_layers,
        kv_ridge=args.kv_ridge,
        kv_max_norm=args.kv_max_norm,
        include_ordering_scores=args.include_ordering_scores,
        include_orderings=args.include_orderings,
        include_raw=args.include_raw,
        fail_fast=args.fail_fast,
        progress_callback=progress,
    )
    result["dataset"] = dataset_name
    result["model"] = args.model
    result["model_revision"] = getattr(args, "revision", None)
    result["backend"] = backend_name
    result["args"] = _args_manifest(args, exclude={"out"})
    atomic_write_json(args.out, result)
    print(json.dumps({"out": args.out, "dataset": dataset_name, "backend": backend_name, "summary": result["summary"]}, indent=2, sort_keys=True), flush=True)


# ---------- parser ----------


def _add_tuner_args(p: argparse.ArgumentParser, *, demo: bool = False) -> None:
    p.add_argument("--gates", type=int, default=1024 if demo else 5000)
    p.add_argument("--steps", type=int, default=80 if demo else 240)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=2 if demo else 8)
    p.add_argument("--score-batches", type=int, default=4 if demo else 16)
    p.add_argument("--layers", default="all")
    p.add_argument("--max-log-gate", type=float, default=0.05)
    p.add_argument("--hook-site", default="layer_output", choices=["layer_output", "layer_input"])
    p.add_argument("--max-length", type=int, default=512 if demo else 1024)


def _add_training_hygiene_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--validation", default=None, help="optional validation JSONL for early stopping / best-controller selection")
    p.add_argument("--eval-every", type=int, default=None, help="validation cadence in optimizer steps; defaults to a coarse schedule")
    p.add_argument("--early-stop-patience", type=int, default=0, help="number of stale validation checks before stopping; 0 disables")
    p.add_argument("--no-select-best", action="store_true", help="do not restore the best validation checkpoint after training")
    p.add_argument("--retain", default=None, help="optional retain JSONL for behavior preservation")
    p.add_argument("--retain-weight", type=float, default=0.0, help="supervised retain-set NLL weight")
    p.add_argument("--kl-to-base", type=float, default=0.0, help="KL(base || controller) retain-set weight")
    p.add_argument("--retain-kl-weight", type=float, default=None, help=argparse.SUPPRESS)  # backward-compatible alias for --kl-to-base
    p.add_argument("--l2", type=float, default=1e-5, help="L2 penalty on signed log gates")
    p.add_argument("--weight-decay", type=float, default=0.0)


def _add_memory_retrieval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", required=True)
    p.add_argument("--namespace", default=None, help="memory namespace; defaults to 'default'")
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--tag", default=None)
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    p.add_argument("--weighting", default="softmax", choices=["softmax", "score", "uniform"])
    p.add_argument("--temperature", type=float, default=0.25)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ntkmirror", description="LoRA-free signed log-mask forward fine-tuning for Hugging Face causal LMs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="fit a signed log-mask controller on JSONL examples")
    _add_common_model_args(fit)
    _add_data_args(fit)
    fit.add_argument("--train", required=True)
    fit.add_argument("--out", required=True)
    _add_tuner_args(fit)
    _add_training_hygiene_args(fit)
    fit.add_argument("--quiet", action="store_true")
    fit.set_defaults(func=cmd_fit)

    fit_dual = sub.add_parser("fit-dual", help="experimental pathwise activation-control NTK fit against full-weight SGD fields")
    _add_common_model_args(fit_dual)
    _add_data_args(fit_dual)
    fit_dual.add_argument("--train", required=True)
    fit_dual.add_argument("--out", required=True)
    fit_dual.add_argument("--report", default=None)
    _add_tuner_args(fit_dual)
    fit_dual.add_argument("--projection", default="target", choices=["target", "topk", "full"])
    fit_dual.add_argument("--top-k", type=int, default=32)
    fit_dual.add_argument("--target-step-size", type=float, default=1e-5)
    fit_dual.add_argument("--apply-scale", type=float, default=1.0)
    fit_dual.add_argument("--ridge", type=float, default=1e-4)
    fit_dual.add_argument("--cg-iters", type=int, default=16)
    fit_dual.add_argument("--cg-tol", type=float, default=1e-5)
    fit_dual.add_argument("--fd-eps", type=float, default=1e-3, help="legacy finite-difference epsilon; only used with --jvp-mode fd")
    fit_dual.add_argument("--jvp-mode", default="exact", choices=["exact", "fd"], help="exact uses autograd JVP; fd is a legacy diagnostic fallback")
    fit_dual.add_argument("--metric", default="identity", choices=["identity", "activation"], help="gate metric M in BM^-1B^T")
    fit_dual.add_argument("--metric-eps", type=float, default=1e-6)
    fit_dual.add_argument("--param-filter", default=None, help="comma substrings of model parameter names to include in the full-weight target")
    fit_dual.add_argument("--max-target-parameters", type=int, default=0, help="0 disables the guard; otherwise fail above this many differentiated parameters")
    fit_dual.add_argument("--quiet", action="store_true")
    fit_dual.set_defaults(func=cmd_fit_dual)

    sec = sub.add_parser("secant", help="diagnose finite controller departure from the initial gate tangent")
    _add_common_model_args(sec)
    _add_data_args(sec)
    _add_controller_load_args(sec)
    sec.add_argument("--controller", required=True)
    sec.add_argument("--eval", required=True)
    sec.add_argument("--out", default=None)
    sec.add_argument("--batch-size", type=int, default=1)
    sec.add_argument("--max-length", type=int, default=1024)
    sec.add_argument("--projection", default="target", choices=["target", "topk", "full"])
    sec.add_argument("--top-k", type=int, default=32)
    sec.add_argument("--fd-eps", type=float, default=1e-3)
    sec.add_argument("--gates", type=int, default=5000)
    sec.add_argument("--layers", default="all")
    sec.add_argument("--max-log-gate", type=float, default=0.05)
    sec.add_argument("--hook-site", default="layer_output", choices=["layer_output", "layer_input"])
    sec.set_defaults(func=cmd_secant)

    diag = sub.add_parser("dual-diagnose", help="test whether the full-weight SGD field lies in the selected gate tangent range")
    _add_common_model_args(diag)
    _add_data_args(diag)
    _add_controller_load_args(diag)
    diag.add_argument("--support", required=True)
    diag.add_argument("--calibration", default=None)
    diag.add_argument("--controller", default=None, help="existing controller basis; if omitted, score a fresh zero basis")
    diag.add_argument("--out", default=None)
    diag.add_argument("--gates", type=int, default=5000)
    diag.add_argument("--layers", default="all")
    diag.add_argument("--max-log-gate", type=float, default=0.05)
    diag.add_argument("--hook-site", default="layer_output", choices=["layer_output", "layer_input"])
    diag.add_argument("--score-batches", type=int, default=16)
    diag.add_argument("--batch-size", type=int, default=1)
    diag.add_argument("--max-length", type=int, default=1024)
    diag.add_argument("--projection", default="target", choices=["target", "topk", "full"])
    diag.add_argument("--top-k", type=int, default=32)
    diag.add_argument("--target-step-size", type=float, default=1e-5)
    diag.add_argument("--ridge", type=float, default=1e-4)
    diag.add_argument("--cg-iters", type=int, default=16)
    diag.add_argument("--cg-tol", type=float, default=1e-5)
    diag.add_argument("--fd-eps", type=float, default=1e-3, help="legacy finite-difference epsilon; only used with --jvp-mode fd")
    diag.add_argument("--jvp-mode", default="exact", choices=["exact", "fd"], help="exact uses autograd JVP; fd is a legacy diagnostic fallback")
    diag.add_argument("--metric", default="identity", choices=["identity", "activation"], help="gate metric M in BM^-1B^T")
    diag.add_argument("--metric-eps", type=float, default=1e-6)
    diag.add_argument("--param-filter", default=None, help="comma substrings of model parameter names to include in the full-weight target")
    diag.add_argument("--max-target-parameters", type=int, default=0, help="0 disables the guard; otherwise fail above this many differentiated parameters")
    diag.set_defaults(func=cmd_dual_diagnose)

    ev = sub.add_parser("eval", help="evaluate base and optionally controller NLL/token accuracy")
    _add_common_model_args(ev)
    _add_data_args(ev)
    _add_controller_load_args(ev)
    ev.add_argument("--eval", required=True)
    ev.add_argument("--controller", default=None)
    ev.add_argument("--out", default=None)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--max-length", type=int, default=1024)
    ev.set_defaults(func=cmd_eval)

    gen = sub.add_parser("generate", help="generate with a fitted controller attached")
    _add_common_model_args(gen)
    _add_controller_load_args(gen)
    gen.add_argument("--controller", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--new-text-only", action="store_true", help="decode only generated tokens, not the prompt")
    gen.add_argument("--gates", type=int, default=5000)
    gen.add_argument("--layers", default="all")
    gen.add_argument("--max-log-gate", type=float, default=0.05)
    gen.set_defaults(func=cmd_generate)

    demo = sub.add_parser("demo", help="write and run a tiny arithmetic demo")
    _add_common_model_args(demo)
    _add_data_args(demo)
    demo.add_argument("--out-dir", default="runs/demo")
    _add_tuner_args(demo, demo=True)
    demo.set_defaults(func=cmd_demo)

    comp = sub.add_parser("compose", help="compose controllers by adding signed log-gates")
    comp.add_argument("--controllers", nargs="+", required=True)
    comp.add_argument("--out", required=True)
    comp.add_argument("--weights", default=None)
    comp.add_argument("--max-log-gate", type=float, default=None)
    comp.add_argument("--report", default=None)
    comp.set_defaults(func=cmd_compose)

    plan = sub.add_parser("compose-plan", help="pre-flight composition conflict/saturation report without writing a controller")
    plan.add_argument("--controllers", nargs="+", required=True)
    plan.add_argument("--weights", default=None)
    plan.add_argument("--max-log-gate", type=float, default=None)
    plan.add_argument("--out", default=None)
    plan.set_defaults(func=cmd_compose_plan)

    ins = sub.add_parser("inspect", help="inspect controller size, overlap, and gate-space cosine")
    ins.add_argument("--controllers", nargs="+", required=True)
    ins.add_argument("--out", default=None)
    ins.set_defaults(func=cmd_inspect)

    lint = sub.add_parser("lint", help="lint one controller artifact for v2 admission")
    lint.add_argument("--controller", required=True)
    lint.add_argument("--require-revision", action="store_true")
    lint.add_argument("--max-gates", type=int, default=0)
    lint.add_argument("--out", default=None)
    lint.set_defaults(func=cmd_lint)

    card = sub.add_parser("card", help="write a Markdown controller card")
    card.add_argument("--controller", required=True)
    card.add_argument("--out", required=True)
    card.add_argument("--metrics", default=None, help="optional JSON object with eval/train metrics")
    card.add_argument("--intended-use", default=None)
    card.add_argument("--limitations", default=None)
    card.set_defaults(func=cmd_card)

    doctor = sub.add_parser("doctor", help="inspect model/tokenizer compatibility for ntkmirror controllers")
    _add_common_model_args(doctor)
    doctor.add_argument("--layers", default="all", help="layer path to inspect, or all for automatic discovery")
    doctor.add_argument("--out", default=None)
    doctor.set_defaults(func=cmd_doctor)

    isr = sub.add_parser("isr-auc", help="run ISR verifier AUROC with optional KV order-debias")
    _add_common_model_args(isr)
    source = isr.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=["vitaminc", "hotpot", "ragtruth"], default=None)
    source.add_argument("--data-jsonl", default=None, help="custom JSONL with claim/spans/supported rows")
    isr.add_argument("--out", default="halu_auc.json")
    isr.add_argument("--backend", default="auto", choices=["auto", "hf", "kv-delta-bayes-ntk"], help="auto uses HF unless --fit-controller is set")
    isr.add_argument("--n", type=int, default=200, help="number of built-in dataset rows to load")
    isr.add_argument("--num-orderings", type=int, default=6)
    isr.add_argument("--cap-spans", type=int, default=5)
    isr.add_argument("--max-chars", type=int, default=240, help="max characters per loaded evidence span")
    isr.add_argument("--fit-controller", action="store_true", help="also compute closed-form NTK KV-debiased q_kv scores")
    isr.add_argument("--use-layers", type=int, default=4, help="number of KV cache layers to expose to the debias solve")
    isr.add_argument("--kv-ridge", type=float, default=1e-3)
    isr.add_argument("--kv-max-norm", type=float, default=10.0)
    isr.add_argument("--dispersion-penalty", type=float, default=1.0)
    isr.add_argument("--leak", type=float, default=0.1, help="false-positive leak budget for TPR@leak")
    isr.add_argument("--shard", type=int, default=0)
    isr.add_argument("--nshards", type=int, default=1)
    isr.add_argument("--query", default=SUPPORT_QUERY)
    isr.add_argument("--yes-choice", default=YESNO[0])
    isr.add_argument("--no-choice", default=YESNO[1])
    isr.add_argument("--length-normalize-choices", action="store_true", help="length-normalize HF choice log-likelihoods")
    isr.add_argument("--add-special-tokens", action="store_true", help="ask the tokenizer to add special tokens in HF verifier scoring")
    isr.add_argument("--include-ordering-scores", action="store_true")
    isr.add_argument("--include-orderings", action="store_true")
    isr.add_argument("--include-raw", action="store_true", help="include raw claims/evidence in the output JSON; avoid for private data")
    isr.add_argument("--progress-every", type=int, default=20)
    isr.add_argument("--fail-fast", action="store_true")
    isr.set_defaults(func=cmd_isr_auc)

    mem = sub.add_parser("memory", help="persistent controller memory")
    mem_sub = mem.add_subparsers(dest="memory_cmd", required=True)

    mem_add = mem_sub.add_parser("add", help="fit or register one controller memory item")
    _add_common_model_args(mem_add)
    _add_data_args(mem_add)
    mem_add.add_argument("--store", required=True)
    mem_add.add_argument("--namespace", default=None, help="memory namespace; defaults to 'default'")
    mem_add.add_argument("--id", required=True)
    mem_add.add_argument("--train", default=None, help="JSONL used to fit a new controller")
    mem_add.add_argument("--controller", default=None, help="existing controller .pt to register")
    mem_add.add_argument("--text", default=None, help="retrieval text; defaults to first training examples")
    mem_add.add_argument("--tags", default=None, help="comma tags, e.g. math,user-pref")
    mem_add.add_argument("--overwrite", action="store_true")
    _add_tuner_args(mem_add)
    _add_training_hygiene_args(mem_add)
    mem_add.add_argument("--quiet", action="store_true")
    mem_add.set_defaults(func=cmd_memory_add)

    mem_list = mem_sub.add_parser("list", help="list memory items")
    mem_list.add_argument("--store", required=True)
    mem_list.add_argument("--namespace", default=None)
    mem_list.add_argument("--include-deleted", action="store_true")
    mem_list.set_defaults(func=cmd_memory_list)

    mem_del = mem_sub.add_parser("delete", help="delete a memory item")
    mem_del.add_argument("--store", required=True)
    mem_del.add_argument("--namespace", default=None)
    mem_del.add_argument("--id", required=True)
    mem_del.add_argument("--soft", action="store_true", help="tombstone the item but keep the artifact for rollback/audit")
    mem_del.set_defaults(func=cmd_memory_delete)

    mem_rollback = mem_sub.add_parser("rollback", help="promote an older memory version to a new active version")
    mem_rollback.add_argument("--store", required=True)
    mem_rollback.add_argument("--namespace", default=None)
    mem_rollback.add_argument("--id", required=True)
    mem_rollback.add_argument("--version", type=int, required=True)
    mem_rollback.set_defaults(func=cmd_memory_rollback)

    mem_audit = mem_sub.add_parser("audit", help="audit memory index/artifact integrity")
    mem_audit.add_argument("--store", required=True)
    mem_audit.add_argument("--namespace", default=None)
    mem_audit.add_argument("--out", default=None)
    mem_audit.set_defaults(func=cmd_memory_audit)

    mem_search = mem_sub.add_parser("search", help="retrieve candidate controller memories")
    _add_memory_retrieval_args(mem_search)
    mem_search.set_defaults(func=cmd_memory_search)

    mem_comp = mem_sub.add_parser("compose", help="retrieve and compose controllers for a query")
    _add_memory_retrieval_args(mem_comp)
    mem_comp.add_argument("--out", required=True)
    mem_comp.add_argument("--max-log-gate", type=float, default=None)
    mem_comp.add_argument("--report", default=None)
    mem_comp.set_defaults(func=cmd_memory_compose)

    mem_gen = mem_sub.add_parser("generate", help="retrieve, compose, and generate with memory controllers")
    _add_common_model_args(mem_gen)
    _add_controller_load_args(mem_gen)
    _add_memory_retrieval_args(mem_gen)
    mem_gen.add_argument("--prompt", default=None, help="generation prompt; defaults to query")
    mem_gen.add_argument("--max-new-tokens", type=int, default=128)
    mem_gen.add_argument("--new-text-only", action="store_true", help="decode only generated tokens, not the prompt")
    mem_gen.add_argument("--compose-max-log-gate", type=float, default=None)
    mem_gen.add_argument("--gates", type=int, default=5000)
    mem_gen.add_argument("--layers", default="all")
    mem_gen.add_argument("--max-log-gate", type=float, default=0.05)
    mem_gen.set_defaults(func=cmd_memory_generate)

    mem_eval = mem_sub.add_parser("eval", help="retrieve, compose, and evaluate NLL on JSONL examples")
    _add_common_model_args(mem_eval)
    _add_data_args(mem_eval)
    _add_controller_load_args(mem_eval)
    _add_memory_retrieval_args(mem_eval)
    mem_eval.add_argument("--eval", required=True)
    mem_eval.add_argument("--compose-max-log-gate", type=float, default=None)
    mem_eval.add_argument("--batch-size", type=int, default=8)
    mem_eval.add_argument("--max-length", type=int, default=1024)
    mem_eval.add_argument("--gates", type=int, default=5000)
    mem_eval.add_argument("--layers", default="all")
    mem_eval.add_argument("--max-log-gate", type=float, default=0.05)
    mem_eval.add_argument("--out", default=None)
    mem_eval.set_defaults(func=cmd_memory_eval)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
