import json
from pathlib import Path

import pytest
import torch

from ntkmirror.controller import SignedLogMaskState
from ntkmirror.memory import ControllerMemoryStore


def _state(raw):
    return SignedLogMaskState(
        layer_path="model.layers",
        n_layers=2,
        hidden_size=4,
        layer_indices=torch.tensor([0, 1], dtype=torch.long),
        channel_indices=torch.tensor([1, 2], dtype=torch.long),
        raw=torch.tensor(raw, dtype=torch.float32),
        max_log_gate=0.1,
    )


def test_memory_retrieve_and_compose(tmp_path: Path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    _state([0.2, 0.0]).save(a)
    _state([0.0, -0.2]).save(b)

    store = ControllerMemoryStore(tmp_path / "store")
    store.add_controller(memory_id="math", text="addition carrying arithmetic", controller_path=a, tags=["math"])
    store.add_controller(memory_id="code", text="python list comprehension", controller_path=b, tags=["code"])

    rows = store.search("arithmetic addition problem", top_k=1)
    assert len(rows) == 1
    assert rows[0].item.id == "math"

    composed, hits = store.compose_for_query("addition with carrying", top_k=1)
    assert composed.n_gates == 2
    assert hits[0].item.id == "math"


def test_memory_zero_overlap_returns_no_hits_by_default(tmp_path: Path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    _state([0.2, 0.0]).save(a)
    _state([0.0, -0.2]).save(b)

    store = ControllerMemoryStore(tmp_path / "store")
    store.add_controller(memory_id="math", text="addition carrying arithmetic", controller_path=a, tags=["math"])
    store.add_controller(memory_id="code", text="python list comprehension", controller_path=b, tags=["code"])

    assert store.search("zebra unrelated", top_k=3) == []
    # Zero-score retrieval remains available as an explicit opt-in for diagnostics.
    assert len(store.search("zebra unrelated", top_k=3, min_score=0.0)) == 2
    with pytest.raises(ValueError, match="no controllers"):
        store.compose_for_query("zebra unrelated", top_k=3)


def test_memory_overwrite_requires_explicit_flag(tmp_path: Path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    _state([0.2, 0.0]).save(a)
    _state([0.0, -0.2]).save(b)

    store = ControllerMemoryStore(tmp_path / "store")
    store.add_controller(memory_id="math", text="first", controller_path=a, tags=["math"])
    with pytest.raises(FileExistsError):
        store.add_controller(memory_id="math", text="second", controller_path=b, tags=["math"])
    store.add_controller(memory_id="math", text="second", controller_path=b, tags=["math"], overwrite=True)
    assert store.get("math").text == "second"


def test_memory_index_rejects_path_traversal(tmp_path: Path):
    store = ControllerMemoryStore(tmp_path / "store")
    payload = {
        "version": 1,
        "items": [
            {
                "id": "evil",
                "text": "evil",
                "controller": "../outside.pt",
                "tags": [],
                "created_at": 0.0,
                "metadata": {},
            }
        ],
    }
    store.index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes memory store"):
        store.list_items()


def test_memory_tag_filter(tmp_path: Path):
    a = tmp_path / "a.pt"
    _state([0.2, 0.0]).save(a)
    store = ControllerMemoryStore(tmp_path / "store")
    store.add_controller(memory_id="math", text="addition carrying arithmetic", controller_path=a, tags=["math"])
    assert store.search("addition", tag="code") == []


def test_memory_namespaces_soft_delete_rollback_and_audit(tmp_path: Path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    _state([0.2, 0.0]).save(a)
    _state([0.0, -0.2]).save(b)
    store = ControllerMemoryStore(tmp_path / "store")
    first = store.add_controller(memory_id="style", namespace="alice", text="formal concise style", controller_path=a)
    second = store.add_controller(memory_id="style", namespace="alice", text="casual warm style", controller_path=b, overwrite=True)
    assert first.version == 1
    assert second.version == 2
    assert store.get("style", namespace="alice").text == "casual warm style"
    assert len(store.list_items(namespace="alice", include_deleted=True)) == 2
    assert len(store.search("formal", namespace="alice", min_score=0.0)) == 1  # only active rows searched
    restored = store.rollback("style", namespace="alice", version=1)
    assert restored.version == 3
    assert store.get("style", namespace="alice").metadata["rolled_back_from_version"] == 1
    store.delete("style", namespace="alice", hard=False)
    assert store.search("formal", namespace="alice", min_score=0.0) == []
    assert len(store.list_items(namespace="alice", include_deleted=True)) == 3
    audit = store.audit(namespace="alice")
    assert audit["status"] == "pass"


def test_memory_audit_detects_orphan_controller(tmp_path: Path):
    a = tmp_path / "a.pt"
    _state([0.2, 0.0]).save(a)
    store = ControllerMemoryStore(tmp_path / "store")
    store.add_controller(memory_id="math", text="addition", controller_path=a)
    orphan = store.controllers_dir / "default" / "orphan.pt"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    _state([0.0, 0.0]).save(orphan)
    audit = store.audit()
    assert audit["status"] == "pass"
    assert any(x["code"] == "orphan-controller" for x in audit["issues"])
