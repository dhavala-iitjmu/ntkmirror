import pytest
import torch
import torch.nn as nn

from ntkmirror.controller import SignedLogMaskState
from ntkmirror.runtime import ControllerRuntime, RuntimePolicy


class _Cfg:
    hidden_size = 4
    _name_or_path = "toy"
    _commit_hash = "rev1"


class _ToyOut:
    def __init__(self, logits):
        self.logits = logits


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.emb = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.head = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.emb(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return _ToyOut(self.head(h))


def _state(**kw):
    base = dict(
        layer_path="model.layers",
        n_layers=2,
        hidden_size=4,
        layer_indices=torch.tensor([0], dtype=torch.long),
        channel_indices=torch.tensor([1], dtype=torch.long),
        raw=torch.tensor([0.0], dtype=torch.float32),
        max_log_gate=0.1,
        model_name="toy",
        model_revision="rev1",
        tokenizer_name=None,
        tokenizer_revision=None,
    )
    base.update(kw)
    return SignedLogMaskState(**base)


def test_runtime_apply_removes_hooks_after_exception():
    model = _ToyModel()
    runtime = ControllerRuntime(model, policy=RuntimePolicy(require_model_identity=True))
    layer = model.model.layers[0]
    assert len(layer._forward_hooks) == 0
    with pytest.raises(RuntimeError):
        with runtime.apply(_state()):
            assert len(layer._forward_hooks) == 1
            raise RuntimeError("boom")
    assert len(layer._forward_hooks) == 0


def test_runtime_rejects_identity_mismatch():
    model = _ToyModel()
    runtime = ControllerRuntime(model, policy=RuntimePolicy(require_model_identity=True))
    with pytest.raises(ValueError, match="model_revision"):
        runtime.validate_state(_state(model_revision="other"))


def test_runtime_rejects_too_many_controllers_and_heterogeneous_batch():
    model = _ToyModel()
    runtime = ControllerRuntime(model, policy=RuntimePolicy(max_controllers=1))
    a = _state(layer_indices=torch.tensor([0]), channel_indices=torch.tensor([0]))
    b = _state(layer_indices=torch.tensor([1]), channel_indices=torch.tensor([1]))
    with pytest.raises(ValueError, match="at most"):
        runtime.compose([a, b])
    with pytest.raises(NotImplementedError, match="heterogeneous"):
        runtime.generate_batch(["a", "b"], [a, b])
