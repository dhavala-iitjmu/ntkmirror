import pytest
import torch

from ntkmirror.controller import SignedLogMaskState
from ntkmirror.compose import compose_states, gate_values, pair_report


def _state(
    layers,
    channels,
    raw,
    max_log_gate=0.1,
    *,
    hook_site="layer_output",
    theory_version="activation_control",
    model_name=None,
    model_revision=None,
    tokenizer_name=None,
    tokenizer_revision=None,
):
    return SignedLogMaskState(
        layer_path="model.layers",
        n_layers=2,
        hidden_size=4,
        layer_indices=torch.tensor(layers, dtype=torch.long),
        channel_indices=torch.tensor(channels, dtype=torch.long),
        raw=torch.tensor(raw, dtype=torch.float32),
        max_log_gate=max_log_gate,
        hook_site=hook_site,
        theory_version=theory_version,
        model_name=model_name,
        model_revision=model_revision,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
    )


def test_compose_sums_signed_log_gates():
    a = _state([0], [1], [0.2])
    b = _state([0, 1], [1, 2], [0.3, -0.1])
    c = compose_states([a, b], weights=[1.0, 1.0], max_log_gate=0.1)
    vals = gate_values(c)
    expected = 0.1 * torch.tanh(torch.tensor(0.2)) + 0.1 * torch.tanh(torch.tensor(0.3))
    assert abs(vals[(0, 1)] - float(expected)) < 1e-5
    assert (1, 2) in vals


def test_compose_preserves_hook_site_theory_and_identity():
    st = _state(
        [0],
        [1],
        [0.2],
        hook_site="layer_input",
        model_name="base-model",
        model_revision="abc123",
        tokenizer_name="base-tokenizer",
        tokenizer_revision="tok123",
    )
    c = compose_states([st], max_log_gate=0.2)
    assert c.hook_site == "layer_input"
    assert c.theory_version == "activation_control"
    assert c.model_name == "base-model"
    assert c.model_revision == "abc123"
    assert c.tokenizer_name == "base-tokenizer"
    assert c.tokenizer_revision == "tok123"
    assert c.metadata["composition"]["n_inputs"] == 1


def test_compose_rejects_mixed_hook_sites():
    a = _state([0], [1], [0.2], hook_site="layer_input")
    b = _state([1], [2], [0.3], hook_site="layer_output")
    with pytest.raises(ValueError, match="hook_site"):
        compose_states([a, b])


def test_compose_rejects_mismatched_optional_identity_even_if_first_is_missing():
    a = _state([0], [1], [0.2], model_name=None)
    b = _state([1], [2], [0.3], model_name="model-a")
    c = _state([0], [3], [0.1], model_name="model-b")
    with pytest.raises(ValueError, match="model_name"):
        compose_states([a, b, c])


def test_pair_report_has_overlap():
    a = _state([0], [1], [0.2])
    b = _state([0, 1], [1, 2], [0.3, -0.1])
    r = pair_report(a, b)
    assert r["overlap_gates"] == 1.0
    assert r["union_gates"] == 2.0
