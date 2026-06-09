from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .controller import SignedLogMaskState
from .layers import find_decoder_layers, infer_hidden_size


@dataclass(frozen=True)
class LintIssue:
    """One controller/store lint finding."""

    severity: str
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _signed_values(state: SignedLogMaskState) -> torch.Tensor:
    return state.max_log_gate * torch.tanh(state.raw.detach().float())


def controller_summary(state: SignedLogMaskState) -> dict[str, Any]:
    """Return an inspectable, JSON-serialisable controller summary."""

    s = _signed_values(state)
    abs_s = s.abs()
    layer_hist: dict[str, int] = {}
    for layer in state.layer_indices.tolist():
        key = str(int(layer))
        layer_hist[key] = layer_hist.get(key, 0) + 1
    return {
        "schema_version": 2,
        "n_gates": state.n_gates,
        "layer_path": state.layer_path,
        "n_layers": state.n_layers,
        "hidden_size": state.hidden_size,
        "hook_site": state.hook_site,
        "theory_version": state.theory_version,
        "max_log_gate": state.max_log_gate,
        "model_name": state.model_name,
        "model_revision": state.model_revision,
        "tokenizer_name": state.tokenizer_name,
        "tokenizer_revision": state.tokenizer_revision,
        "created_at": state.created_at,
        "signed_log_gate": {
            "l1": float(abs_s.sum().item()),
            "l2": float(torch.linalg.vector_norm(s).item()),
            "mean_abs": float(abs_s.mean().item()) if s.numel() else 0.0,
            "max_abs": float(abs_s.max().item()) if s.numel() else 0.0,
            "saturation_fraction_98pct": float((abs_s >= 0.98 * state.max_log_gate).float().mean().item()) if s.numel() else 0.0,
            "positive": int((s > 0).sum().item()),
            "negative": int((s < 0).sum().item()),
            "zero": int((s == 0).sum().item()),
        },
        "layer_histogram": layer_hist,
        "metadata": dict(state.metadata),
    }


def lint_state(
    state: SignedLogMaskState,
    *,
    path: str | Path | None = None,
    require_revision: bool = False,
    max_saturation_fraction: float = 0.25,
    max_layer_fraction: float = 0.80,
) -> list[LintIssue]:
    """Run admission-policy checks on an already-loaded controller."""

    p = None if path is None else str(path)
    issues: list[LintIssue] = []
    try:
        state.validate()
    except Exception as exc:
        return [LintIssue("error", "invalid-state", str(exc), p)]
    for field in ("model_revision", "tokenizer_revision"):
        if getattr(state, field) in (None, ""):
            sev = "error" if require_revision else "warning"
            issues.append(LintIssue(sev, f"missing-{field.replace('_', '-')}", f"{field} is missing; exact checkpoint compatibility cannot be proven", p))
    if state.model_name is None:
        issues.append(LintIssue("warning", "missing-model-name", "model_name is missing", p))
    if state.tokenizer_name is None:
        issues.append(LintIssue("warning", "missing-tokenizer-name", "tokenizer_name is missing", p))
    summary = controller_summary(state)
    sat = float(summary["signed_log_gate"]["saturation_fraction_98pct"])
    if sat > float(max_saturation_fraction):
        issues.append(LintIssue("warning", "high-saturation", f"{sat:.1%} of gates are within 98% of max_log_gate; composition may clip or become brittle", p))
    hist = summary["layer_histogram"]
    if hist:
        layer, count = max(hist.items(), key=lambda kv: kv[1])
        frac = count / max(1, state.n_gates)
        if frac > float(max_layer_fraction) and state.n_gates > 1:
            issues.append(LintIssue("info", "layer-concentrated", f"layer {layer} contains {frac:.1%} of gates; check whether this is expected", p))
    meta = state.metadata or {}
    if "eval" not in meta and "eval_report" not in meta:
        issues.append(LintIssue("info", "missing-eval-report", "controller metadata has no eval report", p))
    if "safety" not in meta and "safety_report" not in meta:
        issues.append(LintIssue("info", "missing-safety-report", "controller metadata has no safety/retention report", p))
    return issues


def lint_controller(
    path: str | Path,
    *,
    require_revision: bool = False,
    unsafe_legacy_load: bool = False,
    max_gates: int | None = None,
) -> dict[str, Any]:
    """Load and lint one controller artifact."""

    p = Path(path)
    try:
        state = SignedLogMaskState.load(p, map_location="cpu", unsafe_legacy_load=unsafe_legacy_load, max_gates=max_gates)
        issues = lint_state(state, path=p, require_revision=require_revision)
        summary = controller_summary(state)
    except Exception as exc:
        issues = [LintIssue("error", "load-failed", f"failed to load controller: {exc}", str(p))]
        summary = None
    status = "fail" if any(i.severity == "error" for i in issues) else "pass"
    return {"path": str(p), "status": status, "summary": summary, "issues": [i.to_dict() for i in issues]}


def write_controller_card(
    controller_path: str | Path,
    out: str | Path,
    *,
    eval_report: Mapping[str, Any] | None = None,
    safety_report: Mapping[str, Any] | None = None,
    intended_use: str | None = None,
    limitations: str | None = None,
) -> dict[str, Any]:
    """Write a Markdown controller card."""

    state = SignedLogMaskState.load(controller_path, map_location="cpu")
    summary = controller_summary(state)
    issues = lint_state(state, path=controller_path)
    intended_use = intended_use or "Not specified. Treat as experimental until an owner supplies intended-use metadata."
    limitations = limitations or (
        "Forward-pass activation controllers bias behavior but do not provide factual provenance. "
        "Use retrieval/RAG for facts that require citation or freshness."
    )
    lines = [
        "# ntkmirror controller card",
        "",
        f"- Controller: `{controller_path}`",
        f"- Created: `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(state.created_at or time.time()))}`",
        f"- Model: `{state.model_name or 'unknown'}`",
        f"- Model revision: `{state.model_revision or 'unknown'}`",
        f"- Tokenizer: `{state.tokenizer_name or 'unknown'}`",
        f"- Tokenizer revision: `{state.tokenizer_revision or 'unknown'}`",
        f"- Hook site: `{state.hook_site}`",
        f"- Gates: `{state.n_gates}`",
        f"- Max log gate: `{state.max_log_gate}`",
        "",
        "## Intended use",
        "",
        intended_use,
        "",
        "## Limitations and trust boundary",
        "",
        limitations,
        "",
        "## Gate statistics",
        "",
        "```json",
        json.dumps(summary["signed_log_gate"], indent=2, sort_keys=True),
        "```",
        "",
        "## Layer histogram",
        "",
        "```json",
        json.dumps(summary["layer_histogram"], indent=2, sort_keys=True),
        "```",
        "",
        "## Lint findings",
        "",
    ]
    if issues:
        lines.extend(f"- **{issue.severity}** `{issue.code}`: {issue.message}" for issue in issues)
    else:
        lines.append("No lint findings.")
    if eval_report is not None:
        lines.extend(["", "## Evaluation report", "", "```json", json.dumps(dict(eval_report), indent=2, sort_keys=True), "```"])
    if safety_report is not None:
        lines.extend(["", "## Safety / retention report", "", "```json", json.dumps(dict(safety_report), indent=2, sort_keys=True), "```"])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"card": str(out_path), "summary": summary, "issues": [i.to_dict() for i in issues]}


def _tokenizer_report(tokenizer) -> dict[str, Any]:
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", None),
        "revision": getattr(tokenizer, "_commit_hash", None) or getattr(tokenizer, "_ntkmirror_requested_revision", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
    }


def doctor_model(model, tokenizer=None, *, layers: str = "all") -> dict[str, Any]:
    """Inspect instantiated model/tokenizer objects for ntkmirror compatibility."""

    issues: list[LintIssue] = []
    try:
        layer_path, decoder_layers = find_decoder_layers(model)
    except Exception as exc:
        return {"status": "fail", "issues": [LintIssue("error", "decoder-layers-not-found", str(exc)).to_dict()]}
    try:
        hidden_size = infer_hidden_size(model)
    except Exception as exc:
        return {"status": "fail", "layer_path": layer_path, "n_layers": len(decoder_layers), "issues": [LintIssue("error", "hidden-size-not-found", str(exc)).to_dict()]}
    tok = _tokenizer_report(tokenizer) if tokenizer is not None else None
    if tok is not None:
        if tok["pad_token_id"] is None and tok["eos_token_id"] is None:
            issues.append(LintIssue("warning", "tokenizer-no-pad-or-eos", "tokenizer has neither pad_token_id nor eos_token_id; batching will fail until a pad token is configured"))
        if tok["has_chat_template"]:
            issues.append(LintIssue("info", "chat-template-present", "raw prompt/completion training does not apply chat templates unless the caller does so explicitly"))
    params = list(model.parameters())
    device = str(params[0].device) if params else "unknown"
    dtype = str(params[0].dtype) if params else "unknown"
    cfg = getattr(model, "config", None)
    report = {
        "status": "fail" if any(i.severity == "error" for i in issues) else "pass",
        "architecture": cfg.__class__.__name__ if cfg is not None else model.__class__.__name__,
        "model_name": getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None),
        "model_revision": getattr(cfg, "_commit_hash", None) or getattr(cfg, "_ntkmirror_requested_revision", None),
        "layer_path": layer_path,
        "n_layers": len(decoder_layers),
        "hidden_size": hidden_size,
        "requested_layers": layers,
        "supported_hook_sites": sorted(SignedLogMaskState.VALID_HOOK_SITES),
        "device": device,
        "dtype": dtype,
        "tokenizer": tok,
        "issues": [i.to_dict() for i in issues],
    }
    for key, value in list(report.items()):
        if isinstance(value, float) and not math.isfinite(value):
            report[key] = None
    return report
