import torch

from ntkmirror.controller import SignedLogMaskState
from ntkmirror.compose import compose_states, gate_values, pair_report


def _state(layers, channels, raw, max_log_gate=0.1):
    return SignedLogMaskState(
        layer_path="model.layers",
        n_layers=2,
        hidden_size=4,
        layer_indices=torch.tensor(layers, dtype=torch.long),
        channel_indices=torch.tensor(channels, dtype=torch.long),
        raw=torch.tensor(raw, dtype=torch.float32),
        max_log_gate=max_log_gate,
    )


def test_compose_sums_signed_log_gates():
    a = _state([0], [1], [0.2])
    b = _state([0, 1], [1, 2], [0.3, -0.1])
    c = compose_states([a, b], weights=[1.0, 1.0], max_log_gate=0.1)
    vals = gate_values(c)
    expected = 0.1 * torch.tanh(torch.tensor(0.2)) + 0.1 * torch.tanh(torch.tensor(0.3))
    assert abs(vals[(0, 1)] - float(expected)) < 1e-5
    assert (1, 2) in vals


def test_pair_report_has_overlap():
    a = _state([0], [1], [0.2])
    b = _state([0, 1], [1, 2], [0.3, -0.1])
    r = pair_report(a, b)
    assert r["overlap_gates"] == 1.0
    assert r["union_gates"] == 2.0
