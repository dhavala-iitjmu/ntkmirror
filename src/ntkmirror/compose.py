from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

import torch

from .controller import SignedLogMaskState


GateKey = tuple[int, int]


def gate_values(state: SignedLogMaskState) -> dict[GateKey, float]:
    """Return a sparse map (layer, channel) -> signed log-gate value s."""
    state.validate()
    s = state.max_log_gate * torch.tanh(state.raw.float())
    out: dict[GateKey, float] = {}
    for layer, channel, value in zip(
        state.layer_indices.tolist(), state.channel_indices.tolist(), s.tolist(), strict=True
    ):
        key = (int(layer), int(channel))
        out[key] = out.get(key, 0.0) + float(value)
    return out


def _atanh_clamped(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def state_from_values(
    values: dict[GateKey, float],
    *,
    template: SignedLogMaskState,
    max_log_gate: float,
    model_name: str | None = None,
    model_revision: str | None = None,
    tokenizer_name: str | None = None,
    tokenizer_revision: str | None = None,
    metadata: dict | None = None,
) -> SignedLogMaskState:
    template.validate()
    if not values:
        raise ValueError("cannot create a controller from zero gate values")
    if not math.isfinite(float(max_log_gate)) or float(max_log_gate) <= 0.0:
        raise ValueError("max_log_gate must be finite and positive")
    items = sorted(values.items())
    layer_indices = torch.tensor([k[0] for k, _ in items], dtype=torch.long)
    channel_indices = torch.tensor([k[1] for k, _ in items], dtype=torch.long)
    signed = torch.tensor([v for _, v in items], dtype=torch.float32)
    signed = torch.clamp(signed, -float(max_log_gate), float(max_log_gate))
    raw = _atanh_clamped(signed / float(max_log_gate))
    return SignedLogMaskState(
        layer_path=template.layer_path,
        n_layers=template.n_layers,
        hidden_size=template.hidden_size,
        layer_indices=layer_indices,
        channel_indices=channel_indices,
        raw=raw,
        max_log_gate=float(max_log_gate),
        model_name=model_name if model_name is not None else template.model_name,
        model_revision=model_revision if model_revision is not None else template.model_revision,
        tokenizer_name=tokenizer_name if tokenizer_name is not None else template.tokenizer_name,
        tokenizer_revision=tokenizer_revision if tokenizer_revision is not None else template.tokenizer_revision,
        hook_site=template.hook_site,
        theory_version=template.theory_version,
        created_at=time.time(),
        metadata={**dict(template.metadata or {}), **dict(metadata or {})},
    )


def _common_optional_identity(states: Sequence[SignedLogMaskState], attr: str) -> str | None:
    values = [(i, getattr(st, attr, None)) for i, st in enumerate(states)]
    known = [(i, val) for i, val in values if val is not None and str(val)]
    if not known:
        return None
    missing = [i for i, val in values if val is None or not str(val)]
    if missing:
        raise ValueError(
            f"cannot compose controllers with partial {attr} identity; "
            f"states {missing} are missing {attr}"
        )
    first_i, first = known[0]
    for i, val in known[1:]:
        if str(val) != str(first):
            raise ValueError(f"state {i} {attr}={val!r} != state {first_i} {attr}={first!r}")
    return str(first)


def assert_compatible(states: Sequence[SignedLogMaskState]) -> None:
    if not states:
        raise ValueError("no controller states provided")
    first = states[0]
    first.validate()
    for i, st in enumerate(states[1:], start=1):
        st.validate()
        if st.hidden_size != first.hidden_size:
            raise ValueError(f"state {i} hidden_size={st.hidden_size} != {first.hidden_size}")
        if st.n_layers != first.n_layers:
            raise ValueError(f"state {i} n_layers={st.n_layers} != {first.n_layers}")
        if st.layer_path != first.layer_path:
            raise ValueError(f"state {i} layer_path={st.layer_path!r} != {first.layer_path!r}")
        if st.hook_site != first.hook_site:
            raise ValueError(f"state {i} hook_site={st.hook_site!r} != {first.hook_site!r}")
        if st.theory_version != first.theory_version:
            raise ValueError(
                f"state {i} theory_version={st.theory_version!r} != {first.theory_version!r}"
            )
    for attr in ("model_name", "model_revision", "tokenizer_name", "tokenizer_revision"):
        _common_optional_identity(states, attr)


def compose_states(
    states: Sequence[SignedLogMaskState],
    *,
    weights: Sequence[float] | None = None,
    max_log_gate: float | None = None,
) -> SignedLogMaskState:
    """Compose controllers by adding their signed log-gate displacements.

    Because the intervention is h' = exp(s) h, adding s-values corresponds to
    multiplying the underlying channel scales. This is the natural composition
    rule for signed log-mask controllers. Values are clipped to max_log_gate.
    """
    assert_compatible(states)
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError("number of weights must match number of controllers")
    weights = [float(w) for w in weights]
    if not all(math.isfinite(w) for w in weights):
        raise ValueError("weights must be finite")
    if max_log_gate is None:
        max_log_gate = max(float(st.max_log_gate) for st in states)
    if not math.isfinite(float(max_log_gate)) or float(max_log_gate) <= 0.0:
        raise ValueError("max_log_gate must be finite and positive")
    values: dict[GateKey, float] = {}
    for weight, state in zip(weights, states, strict=True):
        for key, value in gate_values(state).items():
            values[key] = values.get(key, 0.0) + float(weight) * float(value)
    metadata = {
        "composition": {
            "n_inputs": len(states),
            "weights": [float(w) for w in weights],
            "input_gate_counts": [int(st.n_gates) for st in states],
        }
    }
    return state_from_values(
        values,
        template=states[0],
        max_log_gate=float(max_log_gate),
        model_name=_common_optional_identity(states, "model_name"),
        model_revision=_common_optional_identity(states, "model_revision"),
        tokenizer_name=_common_optional_identity(states, "tokenizer_name"),
        tokenizer_revision=_common_optional_identity(states, "tokenizer_revision"),
        metadata=metadata,
    )


def dense_gate_vector(state: SignedLogMaskState) -> torch.Tensor:
    """Dense [n_layers, hidden_size] vector of signed log-gate values."""
    v = torch.zeros((state.n_layers, state.hidden_size), dtype=torch.float32)
    for (layer, channel), value in gate_values(state).items():
        v[layer, channel] += float(value)
    return v.reshape(-1)


def pair_report(a: SignedLogMaskState, b: SignedLogMaskState) -> dict[str, float]:
    va = dense_gate_vector(a)
    vb = dense_gate_vector(b)
    dot = float(torch.dot(va, vb).item())
    na = float(torch.linalg.vector_norm(va).item())
    nb = float(torch.linalg.vector_norm(vb).item())
    cosine = dot / max(1e-30, na * nb)
    ga = gate_values(a)
    gb = gate_values(b)
    ka = set(ga.keys())
    kb = set(gb.keys())
    inter_keys = ka & kb
    inter = len(inter_keys)
    union = len(ka | kb)
    opposing = sum(1 for key in inter_keys if ga[key] * gb[key] < 0.0)
    return {
        "a_gates": float(len(ka)),
        "b_gates": float(len(kb)),
        "overlap_gates": float(inter),
        "union_gates": float(union),
        "jaccard": float(inter / union) if union else 0.0,
        "opposing_overlap_gates": float(opposing),
        "opposing_overlap_fraction": float(opposing / inter) if inter else 0.0,
        "gate_l2_a": na,
        "gate_l2_b": nb,
        "gate_dot": dot,
        "gate_cosine": cosine,
    }


def composition_report(paths: Sequence[str | Path], states: Sequence[SignedLogMaskState]) -> dict:
    assert_compatible(states)
    report = {
        "controllers": [
            {
                "path": str(path),
                "n_gates": st.n_gates,
                "max_log_gate": st.max_log_gate,
                "hook_site": st.hook_site,
                "theory_version": st.theory_version,
                "model_name": st.model_name,
                "model_revision": st.model_revision,
                "tokenizer_name": st.tokenizer_name,
                "tokenizer_revision": st.tokenizer_revision,
                "l2": float(torch.linalg.vector_norm(dense_gate_vector(st)).item()),
            }
            for path, st in zip(paths, states, strict=True)
        ],
        "pairs": [],
    }
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            row = {"i": i, "j": j, "path_i": str(paths[i]), "path_j": str(paths[j])}
            row.update(pair_report(states[i], states[j]))
            report["pairs"].append(row)
    return report


def composition_plan(
    paths: Sequence[str | Path],
    states: Sequence[SignedLogMaskState],
    *,
    weights: Sequence[float] | None = None,
    max_log_gate: float | None = None,
    saturation_warn_fraction: float = 0.02,
    opposing_overlap_warn_fraction: float = 0.5,
    cosine_warn_below: float = -0.1,
) -> dict:
    """Create a conservative pre-flight plan for controller composition.

    The plan does not prove behavioural safety; it catches mechanical risk
    signals that should be visible before a composed controller is attached: box
    clipping, high overlap with opposite signs, and negative gate-space cosine.
    """

    assert_compatible(states)
    if len(paths) != len(states):
        raise ValueError("paths and states must have the same length")
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError("number of weights must match number of controllers")
    weights = [float(w) for w in weights]
    if not all(math.isfinite(w) for w in weights):
        raise ValueError("weights must be finite")
    if max_log_gate is None:
        max_log_gate = max(float(st.max_log_gate) for st in states)
    if not math.isfinite(float(max_log_gate)) or float(max_log_gate) <= 0.0:
        raise ValueError("max_log_gate must be finite and positive")

    report = composition_report(paths, states)
    weighted_values: dict[GateKey, float] = {}
    unclipped = 0
    for weight, state in zip(weights, states, strict=True):
        for key, value in gate_values(state).items():
            weighted_values[key] = weighted_values.get(key, 0.0) + float(weight) * float(value)
    for value in weighted_values.values():
        if abs(float(value)) > float(max_log_gate):
            unclipped += 1
    n_union = max(1, len(weighted_values))
    saturation_fraction = unclipped / n_union
    warnings: list[dict[str, str]] = []
    if saturation_fraction > float(saturation_warn_fraction):
        warnings.append({
            "code": "composition-clips",
            "message": f"{saturation_fraction:.2%} of composed gates exceed max_log_gate before clipping",
        })
    for pair in report["pairs"]:
        if pair["opposing_overlap_fraction"] > float(opposing_overlap_warn_fraction):
            warnings.append({
                "code": "opposing-overlap",
                "message": (
                    f"controllers {pair['i']} and {pair['j']} have "
                    f"{pair['opposing_overlap_fraction']:.2%} opposing overlap on shared gates"
                ),
            })
        if pair["gate_cosine"] < float(cosine_warn_below):
            warnings.append({
                "code": "negative-gate-cosine",
                "message": f"controllers {pair['i']} and {pair['j']} have gate cosine {pair['gate_cosine']:.3f}",
            })
    report["composition_plan"] = {
        "weights": weights,
        "max_log_gate": float(max_log_gate),
        "union_gates": len(weighted_values),
        "preclip_exceeding_gates": unclipped,
        "preclip_saturation_fraction": float(saturation_fraction),
        "warnings": warnings,
        "ok": not warnings,
    }
    return report


def save_report(path: str | Path, report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
