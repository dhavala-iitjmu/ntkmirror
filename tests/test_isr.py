import json
from pathlib import Path

import pytest
import torch

from ntkmirror.cli import build_parser
from ntkmirror.isr import (
    EvidenceClaim,
    auroc,
    load_evidence_claims_jsonl,
    run_isr_auc,
    score_claim,
    score_kv_debias_probability,
    solve_ridge_projection,
    tpr_at_leak,
    verbalize,
)


class OrderingBackend:
    def p_yes(self, text, *, query=None, choices=None):
        a = text.find("- a")
        b = text.find("- b")
        return 0.75 if a >= 0 and b >= 0 and a < b else 0.25


def test_auroc_uses_average_ranks_for_ties():
    # Pairwise interpretation: positives 0.5, 0.9 vs negatives 0.5, 0.1 =>
    # tie + win + two wins = 3.5 / 4.
    assert auroc([0.5, 0.5, 0.1, 0.9], [1, 0, 0, 1]) == pytest.approx(0.875)
    assert auroc([0.1, 0.2], [1, 1]) is None


def test_tpr_at_leak_handles_boundaries():
    scores = [0.1, 0.3, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert tpr_at_leak(scores, labels, leak=0.0) == 1.0
    assert tpr_at_leak(scores, labels, leak=1.0) == 1.0
    with pytest.raises(ValueError):
        tpr_at_leak(scores, labels, leak=-0.1)


def test_load_custom_evidence_claims_jsonl(tmp_path: Path):
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"id": "a", "claim": "A is true", "spans": ["e1", "e2"], "supported": True})
        + "\n"
        + json.dumps({"claim": "B is false", "evidence": "No. Really no.", "label": "refuted"})
        + "\n",
        encoding="utf-8",
    )
    rows = load_evidence_claims_jsonl(path, cap_spans=2)
    assert rows[0].id == "a"
    assert rows[0].supported is True
    assert rows[1].supported is False
    assert len(rows[1].spans) == 2


def test_score_claim_canonical_and_summary_are_deterministic():
    ex = EvidenceClaim("claim", ("a", "b"), True, id="x")
    rec = score_claim(OrderingBackend(), ex, row_index=0, num_orderings=1, include_ordering_scores=True)
    assert rec["q_canon"] == pytest.approx(0.75)
    assert rec["q_marg"] == pytest.approx(0.75)
    assert rec["ordering_scores"] == [pytest.approx(0.75)]
    result = run_isr_auc(OrderingBackend(), [ex, EvidenceClaim("claim", ("b", "a"), False)], num_orderings=1)
    assert result["summary"]["n_ok"] == 2
    assert "q_marg" in result["summary"]["scores"]


def test_run_isr_auc_row_errors_do_not_abort_by_default():
    class BadBackend:
        def p_yes(self, text, *, query=None, choices=None):
            raise RuntimeError("boom")

    out = run_isr_auc(BadBackend(), [EvidenceClaim("claim", ("a",), True)], fail_fast=False)
    assert out["rows"][0]["status"] == "error"
    with pytest.raises(RuntimeError):
        run_isr_auc(BadBackend(), [EvidenceClaim("claim", ("a",), True)], fail_fast=True)


def test_solve_ridge_projection_matches_small_identity_system():
    B = torch.eye(2)
    d = torch.tensor([1.0, -2.0])
    q, stats = solve_ridge_projection(B, d, ridge=1e-6)
    assert q == pytest.approx(d, rel=1e-5, abs=1e-5)
    assert stats["event_dim"] == 2
    assert stats["controller_dim"] == 2


class FakeLayout:
    def zeros_flat(self, requires_grad=False):
        return torch.zeros(2, requires_grad=requires_grad)


class FakeKVBackend:
    device = torch.device("cpu")
    dtype = torch.float32

    def prefill_cache(self, text):
        tensor = torch.zeros(1, 1, 3, 1)
        return None, [(tensor, tensor)]

    def make_layout_from_cache(self, pkv, spec):
        return FakeLayout()

    def event_logits_from_cache(self, pkv, query, choices, layout=None, q=None):
        if q is None:
            return torch.tensor([0.0, 0.0])
        return q


def test_score_kv_debias_probability_with_fake_backend():
    result = score_kv_debias_probability(
        FakeKVBackend(),
        "Evidence:\n- a\nClaim: c",
        0.8,
        make_dense_spec_fn=lambda **kwargs: kwargs,
    )
    assert result.q_kv > 0.5
    assert result.event_dim == 2
    assert result.controller_dim == 2
    assert result.delta_clipped is False


def test_verbalize_rejects_empty_evidence():
    with pytest.raises(ValueError):
        verbalize([], "claim")


def test_cli_accepts_isr_auc_command(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "isr-auc",
            "--data-jsonl",
            str(tmp_path / "claims.jsonl"),
            "--out",
            str(tmp_path / "out.json"),
            "--backend",
            "hf",
            "--num-orderings",
            "2",
        ]
    )
    assert args.cmd == "isr-auc"
    assert args.backend == "hf"
    assert args.num_orderings == 2
