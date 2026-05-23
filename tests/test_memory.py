from pathlib import Path

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


def test_memory_tag_filter(tmp_path: Path):
    a = tmp_path / "a.pt"
    _state([0.2, 0.0]).save(a)
    store = ControllerMemoryStore(tmp_path / "store")
    store.add_controller(memory_id="math", text="addition carrying arithmetic", controller_path=a, tags=["math"])
    assert store.search("addition", tag="code") == []
