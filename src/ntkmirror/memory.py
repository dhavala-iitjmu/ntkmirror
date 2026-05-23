from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from .compose import compose_states
from .controller import SignedLogMaskState

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "when", "where", "who", "why", "with", "you", "your",
}


def lexical_tokens(text: str) -> list[str]:
    """Small dependency-free tokenizer used by the default retriever."""
    toks = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    return [t for t in toks if len(t) > 1 and t not in _STOP]


def _slug(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s).strip()).strip("-._")
    if not out:
        raise ValueError("memory id becomes empty after sanitisation")
    return out[:160]


@dataclass
class MemoryItem:
    """One persistent controller memory entry.

    `text` is used for retrieval. `controller` is the relative path to the
    controller state inside the memory store. `metadata` is deliberately loose so
    callers can record provenance, source document ids, train/eval manifests, or
    user preference versioning.
    """

    id: str
    text: str
    controller: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryHit:
    item: MemoryItem
    score: float
    weight: float = 1.0

    def to_dict(self) -> dict:
        obj = asdict(self.item)
        obj.update({"score": float(self.score), "weight": float(self.weight)})
        return obj


class ControllerMemoryStore:
    """A tiny persistent memory store for signed log-mask controllers.

    Layout:
        root/index.json
        root/controllers/<id>.pt

    The default retriever is lexical TF-IDF cosine. This keeps the package
    installation-light and makes the most important failure mode visible: if the
    wrong controller is retrieved, composition will be wrong even if the
    controller itself is good.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.controllers_dir = self.root / "controllers"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.controllers_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_items([])

    def _read_items(self) -> list[MemoryItem]:
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = raw.get("items", [])
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError(f"invalid memory index at {self.index_path}")
        return [MemoryItem(**row) for row in rows]

    def _write_items(self, items: Sequence[MemoryItem]) -> None:
        payload = {"version": 1, "items": [asdict(x) for x in items]}
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def list_items(self) -> list[MemoryItem]:
        return self._read_items()

    def get(self, memory_id: str) -> MemoryItem:
        target = _slug(memory_id)
        for item in self._read_items():
            if item.id == target:
                return item
        raise KeyError(f"memory item not found: {memory_id}")

    def controller_path(self, item: MemoryItem | str) -> Path:
        if isinstance(item, MemoryItem):
            rel = item.controller
            return self.root / rel
        return self.controllers_dir / f"{_slug(item)}.pt"

    def add_controller(
        self,
        *,
        memory_id: str,
        controller_path: str | Path,
        text: str,
        tags: Sequence[str] = (),
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> MemoryItem:
        mid = _slug(memory_id)
        controller_path = Path(controller_path)
        if not controller_path.exists():
            raise FileNotFoundError(controller_path)
        # Validate that the payload is a SignedLogMaskState-compatible object.
        st = SignedLogMaskState.load(controller_path, map_location="cpu")
        dst = self.controllers_dir / f"{mid}.pt"
        if dst.exists() and not overwrite:
            raise FileExistsError(f"controller already exists for memory id {mid}; pass overwrite=True")
        shutil.copy2(controller_path, dst)
        items = [x for x in self._read_items() if x.id != mid]
        meta = dict(metadata or {})
        meta.setdefault("n_gates", st.n_gates)
        meta.setdefault("model_name", st.model_name)
        meta.setdefault("max_log_gate", st.max_log_gate)
        item = MemoryItem(
            id=mid,
            text=str(text),
            controller=str(dst.relative_to(self.root)),
            tags=[str(t) for t in tags if str(t).strip()],
            metadata=meta,
        )
        items.append(item)
        self._write_items(sorted(items, key=lambda x: x.id))
        return item

    def delete(self, memory_id: str) -> None:
        mid = _slug(memory_id)
        kept: list[MemoryItem] = []
        removed: MemoryItem | None = None
        for item in self._read_items():
            if item.id == mid:
                removed = item
            else:
                kept.append(item)
        if removed is None:
            raise KeyError(f"memory item not found: {memory_id}")
        path = self.controller_path(removed)
        if path.exists():
            path.unlink()
        self._write_items(kept)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        tag: str | None = None,
        min_score: float = 0.0,
    ) -> list[MemoryHit]:
        items = self._read_items()
        if tag:
            items = [x for x in items if tag in set(x.tags)]
        if not items:
            return []

        docs = [Counter(lexical_tokens(" ".join([x.text, " ".join(x.tags)]))) for x in items]
        q = Counter(lexical_tokens(query))
        if not q:
            return []

        df = Counter()
        for doc in docs:
            for term in doc:
                df[term] += 1
        n = len(docs)

        def idf(term: str) -> float:
            return math.log((1.0 + n) / (1.0 + df.get(term, 0))) + 1.0

        qv = {t: c * idf(t) for t, c in q.items()}
        qn = math.sqrt(sum(v * v for v in qv.values()))
        hits: list[MemoryHit] = []
        for item, doc in zip(items, docs, strict=True):
            dv = {t: c * idf(t) for t, c in doc.items()}
            dn = math.sqrt(sum(v * v for v in dv.values()))
            dot = sum(qv.get(t, 0.0) * dv.get(t, 0.0) for t in qv)
            score = dot / max(1e-12, qn * dn)
            if score >= float(min_score):
                hits.append(MemoryHit(item=item, score=float(score), weight=1.0))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(0, int(top_k))]

    def weight_hits(
        self,
        hits: Sequence[MemoryHit],
        *,
        weighting: str = "softmax",
        temperature: float = 0.25,
    ) -> list[MemoryHit]:
        if not hits:
            return []
        if weighting == "uniform":
            weights = [1.0] * len(hits)
        elif weighting == "score":
            weights = [max(0.0, h.score) for h in hits]
            s = sum(weights)
            weights = [w / s for w in weights] if s > 0 else [1.0 / len(hits)] * len(hits)
        elif weighting == "softmax":
            t = max(1e-6, float(temperature))
            m = max(h.score for h in hits)
            vals = [math.exp((h.score - m) / t) for h in hits]
            z = sum(vals)
            weights = [v / z for v in vals]
        else:
            raise ValueError("weighting must be one of: softmax, score, uniform")
        return [MemoryHit(item=h.item, score=h.score, weight=float(w)) for h, w in zip(hits, weights, strict=True)]

    def compose_for_query(
        self,
        query: str,
        *,
        top_k: int = 3,
        weighting: str = "softmax",
        temperature: float = 0.25,
        max_log_gate: float | None = None,
        tag: str | None = None,
        min_score: float = 0.0,
    ) -> tuple[SignedLogMaskState, list[MemoryHit]]:
        hits = self.search(query, top_k=top_k, tag=tag, min_score=min_score)
        hits = self.weight_hits(hits, weighting=weighting, temperature=temperature)
        if not hits:
            raise ValueError("memory search returned no controllers")
        states = [SignedLogMaskState.load(self.controller_path(h.item), map_location="cpu") for h in hits]
        state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=max_log_gate)
        return state, hits
