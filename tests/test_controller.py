import pytest
import torch
import torch.nn as nn

from ntkmirror.controller import ForwardFineTuner, SignedLogMaskState, _SignedLogMaskModule
from ntkmirror.data import Example


def _state(**kwargs):
    base = dict(
        layer_path="model.layers",
        n_layers=2,
        hidden_size=4,
        layer_indices=torch.tensor([0], dtype=torch.long),
        channel_indices=torch.tensor([1], dtype=torch.long),
        raw=torch.tensor([0.0], dtype=torch.float32),
        max_log_gate=0.1,
    )
    base.update(kwargs)
    return SignedLogMaskState(**base)


def test_controller_does_not_register_base_layers_as_children():
    layers = nn.ModuleList([nn.Linear(3, 3)])
    module = _SignedLogMaskModule(
        layers,
        torch.tensor([0]),
        torch.tensor([1]),
        hidden_size=3,
        max_log_gate=0.1,
    )
    assert [name for name, _ in module.named_parameters()] == ["raw"]
    assert set(module.state_dict()) == {"raw", "layer_indices", "channel_indices"}


def test_state_validation_rejects_bad_indices_nan_and_duplicates():
    with pytest.raises(ValueError, match="channel_indices"):
        _state(channel_indices=torch.tensor([4]))
    with pytest.raises(ValueError, match="raw contains"):
        _state(raw=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="duplicate"):
        _state(
            layer_indices=torch.tensor([0, 0]),
            channel_indices=torch.tensor([1, 1]),
            raw=torch.tensor([0.0, 0.1]),
        )


def test_state_validation_rejects_non_rank_one_payload():
    with pytest.raises(ValueError, match="rank-1"):
        _state(layer_indices=torch.tensor([[0]]))


def test_load_rejects_unsupported_schema(tmp_path):
    path = tmp_path / "bad_schema.pt"
    torch.save({"schema_version": 999}, path)
    with pytest.raises(ValueError, match="unsupported controller schema"):
        SignedLogMaskState.load(path)


class _TinyTok:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        table = {"A": 0, "B": 1}
        return [table.get(ch, 0) for ch in text]


class _TinyConfig:
    hidden_size = 4


class _TinyOut:
    def __init__(self, logits):
        self.logits = logits


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _TinyConfig()
        self.emb = nn.Embedding(2, 4)
        with torch.no_grad():
            self.emb.weight.zero_()
            self.emb.weight[0, 0] = 1.0
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.head = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1, 0] = 1.0

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.emb(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return _TinyOut(self.head(h))


def test_evaluate_nll_use_controller_false_detaches_outer_controller():
    model = _TinyModel()
    tuner = ForwardFineTuner(model, _TinyTok(), gates=1, max_log_gate=1.0)
    tuner.controller = _SignedLogMaskModule(
        tuner.decoder_layers,
        torch.tensor([0]),
        torch.tensor([0]),
        hidden_size=4,
        max_log_gate=1.0,
        raw_init=torch.tensor([5.0]),
    )
    examples = [Example("", "AB")]
    with tuner.controller.attached():
        controller_nll = tuner.evaluate_nll(examples, batch_size=1, max_length=8, use_controller=True)["nll"]
        base_nll = tuner.evaluate_nll(examples, batch_size=1, max_length=8, use_controller=False)["nll"]
        assert tuner.controller.is_attached
    assert controller_nll != base_nll
