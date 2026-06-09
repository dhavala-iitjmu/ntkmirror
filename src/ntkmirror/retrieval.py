from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch

from .memory import DEFAULT_MIN_SCORE, ControllerMemoryStore, MemoryHit, MemoryItem, lexical_tokens


@dataclass
class RetrievalConfig:
    method: str = "hybrid"  # lexical | embedding | hybrid
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    hybrid_alpha: float = 0.65  # weight on embedding score; lexical gets 1-alpha
    batch_size: int = 64


def _normalise01(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo + 1e-12:
        # Preserve a meaningful singleton/all-tied positive signal. Returning all
        # zeros here causes hybrid retrieval with one clearly matching memory to
        # be filtered as a no-hit by the default positive min_score.
        val = 1.0 if hi > 0.0 else 0.0
        return {k: val for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class MemoryRetriever:
    """Retriever for controller memories.

    The default persistent-memory store deliberately uses a light lexical scorer.
    This class adds embedding and hybrid retrieval for the benchmark, so the
    benchmark can separate controller quality from retrieval quality.
    """

    def __init__(self, store: ControllerMemoryStore, config: RetrievalConfig):
        self.store = store
        self.config = config
        method = config.method.lower().strip()
        if method not in {"lexical", "embedding", "hybrid"}:
            raise ValueError("retrieval method must be lexical, embedding, or hybrid")
        self.method = method
        self.items = store.list_items()
        self._emb_model = None
        self._doc_embeddings = None
        if self.method in {"embedding", "hybrid"}:
            self._init_embeddings()

    def refresh(self) -> None:
        self.items = self.store.list_items()
        self._doc_embeddings = None
        if self.method in {"embedding", "hybrid"}:
            self._init_embeddings()

    @staticmethod
    def _item_text(item: MemoryItem) -> str:
        pieces = [item.id, item.text, " ".join(item.tags)]
        # Include metadata descriptors if present. This keeps retrieval robust
        # when item.text is a compact descriptor but the register command stored
        # richer retrieval text under metadata.
        meta = item.metadata or {}
        for key in ("descriptor", "task_id"):
            val = meta.get(key)
            if val:
                pieces.append(str(val))
        retrieval_text = str(meta.get("retrieval_text", ""))
        if retrieval_text and retrieval_text not in str(item.text):
            pieces.append(retrieval_text)
        return "\n".join(pieces)

    def _init_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover - first-run UX guard
            raise RuntimeError(
                "Embedding retrieval requires sentence-transformers. Install with: "
                "pip install sentence-transformers"
            ) from e
        self._emb_model = SentenceTransformer(self.config.embedding_model, device=self.config.embedding_device)
        docs = [self._item_text(x) for x in self.items]
        if docs:
            self._doc_embeddings = self._emb_model.encode(
                docs,
                batch_size=int(self.config.batch_size),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self._doc_embeddings = torch.empty((0, 1), dtype=torch.float32)

    def _lexical_scores(self, query: str, items: Sequence[MemoryItem]) -> dict[str, float]:
        docs = [Counter(lexical_tokens(self._item_text(x))) for x in items]
        q = Counter(lexical_tokens(query))
        if not q:
            return {x.id: 0.0 for x in items}
        df = Counter()
        for doc in docs:
            for term in doc:
                df[term] += 1
        n = len(docs)

        def idf(term: str) -> float:
            return math.log((1.0 + n) / (1.0 + df.get(term, 0))) + 1.0

        qv = {t: c * idf(t) for t, c in q.items()}
        qn = math.sqrt(sum(v * v for v in qv.values()))
        out: dict[str, float] = {}
        for item, doc in zip(items, docs, strict=True):
            dv = {t: c * idf(t) for t, c in doc.items()}
            dn = math.sqrt(sum(v * v for v in dv.values()))
            dot = sum(qv.get(t, 0.0) * dv.get(t, 0.0) for t in qv)
            out[item.id] = dot / max(1e-12, qn * dn)
        return out

    def _embedding_scores(self, query: str, items: Sequence[MemoryItem]) -> dict[str, float]:
        if self._emb_model is None or self._doc_embeddings is None:
            self._init_embeddings()
        if not items:
            return {}
        # The item order can be filtered by tag, so map from global item id to embedding row.
        id_to_row = {item.id: i for i, item in enumerate(self.items)}
        q = self._emb_model.encode(
            [query],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        out = {}
        for item in items:
            row = id_to_row[item.id]
            out[item.id] = float(torch.dot(q, self._doc_embeddings[row]).item())
        return out

    def search(self, query: str, *, top_k: int = 3, tag: str | None = None, min_score: float = DEFAULT_MIN_SCORE) -> list[MemoryHit]:
        if int(top_k) <= 0:
            return []
        items = self.items
        if tag:
            items = [x for x in items if tag in set(x.tags)]
        if not items:
            return []

        lex = self._lexical_scores(query, items) if self.method in {"lexical", "hybrid"} else {}
        emb = self._embedding_scores(query, items) if self.method in {"embedding", "hybrid"} else {}
        if self.method == "lexical":
            scores = lex
        elif self.method == "embedding":
            scores = emb
        else:
            lex_n = _normalise01(lex)
            emb_n = _normalise01(emb)
            a = float(self.config.hybrid_alpha)
            scores = {item.id: a * emb_n.get(item.id, 0.0) + (1.0 - a) * lex_n.get(item.id, 0.0) for item in items}

        by_id = {item.id: item for item in items}
        hits = [MemoryHit(item=by_id[mid], score=float(score), weight=1.0) for mid, score in scores.items() if score >= float(min_score)]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(0, int(top_k))]


def build_memory_retriever(
    store: ControllerMemoryStore,
    *,
    method: str = "hybrid",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "cpu",
    hybrid_alpha: float = 0.65,
    batch_size: int = 64,
) -> MemoryRetriever:
    return MemoryRetriever(
        store,
        RetrievalConfig(
            method=method,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            hybrid_alpha=hybrid_alpha,
            batch_size=batch_size,
        ),
    )
