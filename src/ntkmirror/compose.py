from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from .controller import SignedLogMaskState


GateKey = tuple[int, int]


def gate_values(state: SignedLogMaskState) -> dict[GateKey, float]:
    """Return a sparse map (layer, channel) -> signed log-gate value s."""
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
) -> SignedLogMaskState:
    if not values:
        raise ValueError("cannot create a controller from zero gate values")
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
    )


def assert_compatible(states: Sequence[SignedLogMaskState]) -> None:
    if not states:
        raise ValueError("no controller states provided")
    first = states[0]
    for i, st in enumerate(states[1:], start=1):
        if st.hidden_size != first.hidden_size:
            raise ValueError(f"state {i} hidden_size={st.hidden_size} != {first.hidden_size}")
        if st.n_layers != first.n_layers:
            raise ValueError(f"state {i} n_layers={st.n_layers} != {first.n_layers}")
        if st.layer_path != first.layer_path:
            raise ValueError(f"state {i} layer_path={st.layer_path!r} != {first.layer_path!r}")


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
    if max_log_gate is None:
        max_log_gate = max(float(st.max_log_gate) for st in states)
    values: dict[GateKey, float] = {}
    for weight, state in zip(weights, states, strict=True):
        for key, value in gate_values(state).items():
            values[key] = values.get(key, 0.0) + float(weight) * float(value)
    return state_from_values(values, template=states[0], max_log_gate=float(max_log_gate))


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
    ka = set(gate_values(a).keys())
    kb = set(gate_values(b).keys())
    inter = len(ka & kb)
    union = len(ka | kb)
    return {
        "a_gates": float(len(ka)),
        "b_gates": float(len(kb)),
        "overlap_gates": float(inter),
        "union_gates": float(union),
        "jaccard": float(inter / union) if union else 0.0,
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


def save_report(path: str | Path, report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
