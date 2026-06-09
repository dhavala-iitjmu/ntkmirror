import torch
import torch.nn as nn

from ntkmirror.artifacts import controller_summary, doctor_model, lint_controller, write_controller_card
from ntkmirror.controller import SignedLogMaskState


def _state(**kw):
    base = dict(
        layer_path="model.layers",
        n_layers=2,
        hidden_size=4,
        layer_indices=torch.tensor([0, 0], dtype=torch.long),
        channel_indices=torch.tensor([1, 2], dtype=torch.long),
        raw=torch.tensor([5.0, -5.0], dtype=torch.float32),
        max_log_gate=0.1,
        model_name="toy",
    )
    base.update(kw)
    return SignedLogMaskState(**base)


class _Cfg:
    hidden_size = 4
    _name_or_path = "toy"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.emb = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def get_input_embeddings(self):
        return self.emb


class _Tok:
    name_or_path = "tok"
    pad_token_id = None
    eos_token_id = 0
    chat_template = "{{ messages }}"


def test_controller_lint_summary_and_card(tmp_path):
    path = tmp_path / "ctrl.pt"
    _state().save(path)
    report = lint_controller(path)
    assert report["status"] == "pass"
    codes = {x["code"] for x in report["issues"]}
    assert "missing-model-revision" in codes
    assert "missing-tokenizer-revision" in codes
    assert "high-saturation" in codes
    summary = controller_summary(_state())
    assert summary["n_gates"] == 2
    card = tmp_path / "card.md"
    out = write_controller_card(path, card, intended_use="unit test")
    assert out["card"] == str(card)
    text = card.read_text(encoding="utf-8")
    assert "ntkmirror controller card" in text
    assert "unit test" in text


def test_doctor_model_reports_layer_stack_and_chat_template():
    report = doctor_model(_Model(), _Tok())
    assert report["status"] == "pass"
    assert report["layer_path"] == "model.layers"
    assert report["n_layers"] == 2
    codes = {x["code"] for x in report["issues"]}
    assert "chat-template-present" in codes
