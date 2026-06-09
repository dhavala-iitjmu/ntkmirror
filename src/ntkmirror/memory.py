from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .compose import compose_states
from .controller import SignedLogMaskState


DEFAULT_MIN_SCORE = 1e-8
DEFAULT_NAMESPACE = "default"

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


def _namespace(s: str | None) -> str:
    return _slug(s or DEFAULT_NAMESPACE)


@dataclass
class MemoryItem:
    """One persistent controller memory entry.

    ``text`` is used for retrieval. ``controller`` is the relative path to the
    controller state inside the memory store. ``namespace``, ``version``, and
    ``deleted_at`` provide basic lifecycle semantics while keeping the default
    local JSON store auditable.
    """

    id: str
    text: str
    controller: str
    tags: list[str] = field(default_factory=list)
    namespace: str = DEFAULT_NAMESPACE
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float | None = None
    deleted_at: float | None = None
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
    """Persistent memory store for signed log-mask controllers.

    Layout:
        root/index.json
        root/controllers/<namespace>/<id>.v<version>.pt

    Stores created by earlier versions remain readable because each index row
    carries its relative controller path. Search excludes soft-deleted rows by
    default and never returns zero-overlap rows unless ``min_score=0`` is an
    explicit diagnostic opt-in.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.controllers_dir = self.root / "controllers"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.controllers_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_items([])

    def _read_items(self, *, include_deleted: bool = False) -> list[MemoryItem]:
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = raw.get("items", [])
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError(f"invalid memory index at {self.index_path}")
        out: list[MemoryItem] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"invalid memory item row in {self.index_path}")
            payload = dict(row)
            payload.setdefault("namespace", DEFAULT_NAMESPACE)
            payload.setdefault("version", 1)
            payload.setdefault("updated_at", None)
            payload.setdefault("deleted_at", None)
            payload.setdefault("metadata", {})
            payload.setdefault("tags", [])
            item = MemoryItem(**payload)
            item.id = _slug(item.id)
            item.namespace = _namespace(item.namespace)
            item.version = int(item.version)
            if item.version <= 0:
                raise ValueError(f"invalid memory version for {item.namespace}/{item.id}")
            if item.deleted_at is not None:
                item.deleted_at = float(item.deleted_at)
            if item.updated_at is not None:
                item.updated_at = float(item.updated_at)
            item.tags = [str(t) for t in item.tags if str(t).strip()]
            item.metadata = dict(item.metadata or {})
            # Validate relative controller paths when reading the index so a
            # tampered store fails before load/delete can escape the store root.
            self.controller_path(item)
            if include_deleted or item.deleted_at is None:
                out.append(item)
        return out

    def _write_items(self, items: Sequence[MemoryItem]) -> None:
        payload = {"version": 2, "items": [asdict(x) for x in items]}
        tmp = self.index_path.with_name(f"{self.index_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def list_items(self, *, namespace: str | None = None, include_deleted: bool = False) -> list[MemoryItem]:
        items = self._read_items(include_deleted=include_deleted)
        if namespace is not None:
            ns = _namespace(namespace)
            items = [x for x in items if x.namespace == ns]
        return items

    def get(self, memory_id: str, *, namespace: str | None = None, include_deleted: bool = False) -> MemoryItem:
        target = _slug(memory_id)
        ns = _namespace(namespace)
        for item in self._read_items(include_deleted=include_deleted):
            if item.id == target and item.namespace == ns:
                return item
        raise KeyError(f"memory item not found: {ns}/{target}")

    def _controller_dst_path(self, memory_id: str, *, namespace: str, version: int) -> Path:
        ns = _namespace(namespace)
        mid = _slug(memory_id)
        dst_dir = self.controllers_dir / ns
        dst_dir.mkdir(parents=True, exist_ok=True)
        return (dst_dir / f"{mid}.v{int(version)}.pt").resolve()

    def controller_path(self, item: MemoryItem | str) -> Path:
        if isinstance(item, MemoryItem):
            rel = Path(item.controller)
            if rel.is_absolute():
                raise ValueError(f"controller path for memory {item.namespace}/{item.id!r} must be relative")
            root = self.root.resolve()
            controllers_root = self.controllers_dir.resolve()
            path = (self.root / rel).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"controller path for memory {item.namespace}/{item.id!r} escapes memory store")
            if not path.is_relative_to(controllers_root):
                raise ValueError(f"controller path for memory {item.namespace}/{item.id!r} must stay under controllers/")
            return path
        # Backward-compatible convenience for callers that only have an id.
        return (self.controllers_dir / f"{_slug(item)}.pt").resolve()

    def add_controller(
        self,
        *,
        memory_id: str,
        controller_path: str | Path,
        text: str,
        tags: Sequence[str] = (),
        metadata: dict | None = None,
        namespace: str | None = None,
        overwrite: bool = False,
        admission: dict | None = None,
    ) -> MemoryItem:
        mid = _slug(memory_id)
        ns = _namespace(namespace)
        src = Path(controller_path)
        if not src.exists():
            raise FileNotFoundError(src)
        # Validate that the payload is a SignedLogMaskState-compatible object.
        st = SignedLogMaskState.load(src, map_location="cpu")
        existing = self._read_items(include_deleted=True)
        active = [x for x in existing if x.id == mid and x.namespace == ns and x.deleted_at is None]
        if active and not overwrite:
            raise FileExistsError(f"memory id {ns}/{mid} already exists; pass overwrite=True")
        next_version = max([x.version for x in existing if x.id == mid and x.namespace == ns] or [0]) + 1
        dst = self._controller_dst_path(mid, namespace=ns, version=next_version)
        if dst.exists() and not overwrite:
            raise FileExistsError(f"controller artifact already exists for {ns}/{mid} v{next_version}; pass overwrite=True")
        if src.resolve() != dst.resolve():
            tmp_dst = dst.with_name(f".{dst.name}.tmp.{os.getpid()}.{time.time_ns()}")
            try:
                shutil.copy2(src, tmp_dst)
                os.replace(tmp_dst, dst)
            finally:
                try:
                    tmp_dst.unlink()
                except FileNotFoundError:
                    pass
        now = time.time()
        kept: list[MemoryItem] = []
        retired: list[MemoryItem] = []
        for item in existing:
            if item.id == mid and item.namespace == ns and item.deleted_at is None:
                item.deleted_at = now
                item.updated_at = now
                item.metadata = dict(item.metadata or {})
                item.metadata.setdefault("retired_by_version", next_version)
                retired.append(item)
            else:
                kept.append(item)
        meta = dict(metadata or {})
        meta.setdefault("n_gates", st.n_gates)
        meta.setdefault("model_name", st.model_name)
        meta.setdefault("model_revision", st.model_revision)
        meta.setdefault("tokenizer_name", st.tokenizer_name)
        meta.setdefault("tokenizer_revision", st.tokenizer_revision)
        meta.setdefault("max_log_gate", st.max_log_gate)
        if admission is not None:
            meta.setdefault("admission", dict(admission))
        item = MemoryItem(
            id=mid,
            text=str(text),
            controller=str(dst.relative_to(self.root.resolve())),
            tags=[str(t) for t in tags if str(t).strip()],
            namespace=ns,
            version=next_version,
            created_at=now,
            updated_at=now,
            deleted_at=None,
            metadata=meta,
        )
        kept.extend(retired)
        kept.append(item)
        self._write_items(sorted(kept, key=lambda x: (x.namespace, x.id, x.version, x.deleted_at is not None)))
        return item

    def delete(self, memory_id: str, *, namespace: str | None = None, hard: bool = True) -> None:
        mid = _slug(memory_id)
        ns = _namespace(namespace)
        kept: list[MemoryItem] = []
        removed: list[MemoryItem] = []
        for item in self._read_items(include_deleted=True):
            if item.id == mid and item.namespace == ns and item.deleted_at is None:
                removed.append(item)
            else:
                kept.append(item)
        if not removed:
            raise KeyError(f"memory item not found: {ns}/{mid}")
        if hard:
            for item in removed:
                path = self.controller_path(item)
                if path.exists():
                    path.unlink()
            self._write_items(kept)
        else:
            now = time.time()
            for item in removed:
                item.deleted_at = now
                item.updated_at = now
            kept.extend(removed)
            self._write_items(sorted(kept, key=lambda x: (x.namespace, x.id, x.version, x.deleted_at is not None)))

    def rollback(self, memory_id: str, *, namespace: str | None = None, version: int) -> MemoryItem:
        mid = _slug(memory_id)
        ns = _namespace(namespace)
        version = int(version)
        if version <= 0:
            raise ValueError("version must be positive")
        items = self._read_items(include_deleted=True)
        target = next((x for x in items if x.id == mid and x.namespace == ns and x.version == version), None)
        if target is None:
            raise KeyError(f"memory version not found: {ns}/{mid} v{version}")
        src_path = self.controller_path(target)
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        now = time.time()
        for item in items:
            if item.id == mid and item.namespace == ns and item.deleted_at is None:
                item.deleted_at = now
                item.updated_at = now
        next_version = max(x.version for x in items if x.id == mid and x.namespace == ns) + 1
        dst = self._controller_dst_path(mid, namespace=ns, version=next_version)
        shutil.copy2(src_path, dst)
        restored = MemoryItem(
            id=mid,
            namespace=ns,
            version=next_version,
            text=target.text,
            controller=str(dst.relative_to(self.root.resolve())),
            tags=list(target.tags),
            created_at=now,
            updated_at=now,
            metadata={**dict(target.metadata or {}), "rolled_back_from_version": version},
        )
        items.append(restored)
        self._write_items(sorted(items, key=lambda x: (x.namespace, x.id, x.version, x.deleted_at is not None)))
        return restored

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        tag: str | None = None,
        namespace: str | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[MemoryHit]:
        if int(top_k) <= 0:
            return []
        items = self.list_items(namespace=namespace, include_deleted=False)
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
        namespace: str | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> tuple[SignedLogMaskState, list[MemoryHit]]:
        hits = self.search(query, top_k=top_k, tag=tag, namespace=namespace, min_score=min_score)
        hits = self.weight_hits(hits, weighting=weighting, temperature=temperature)
        if not hits:
            raise ValueError("memory search returned no controllers")
        states = [SignedLogMaskState.load(self.controller_path(h.item), map_location="cpu") for h in hits]
        state = compose_states(states, weights=[h.weight for h in hits], max_log_gate=max_log_gate)
        return state, hits

    def audit(self, *, namespace: str | None = None) -> dict:
        """Return integrity and governance checks for a memory store."""

        issues: list[dict] = []
        try:
            items = self.list_items(namespace=namespace, include_deleted=True)
        except Exception as exc:
            return {"status": "fail", "issues": [{"severity": "error", "code": "index-invalid", "message": str(exc)}]}
        active = [x for x in items if x.deleted_at is None]
        seen: set[tuple[str, str]] = set()
        identities: set[tuple[str | None, str | None, str | None, str | None]] = set()
        referenced: set[Path] = set()
        for item in items:
            key = (item.namespace, item.id)
            if item.deleted_at is None:
                if key in seen:
                    issues.append({"severity": "error", "code": "duplicate-active-id", "message": f"duplicate active memory {item.namespace}/{item.id}"})
                seen.add(key)
            path = self.controller_path(item)
            referenced.add(path.resolve())
            if not path.exists() and item.deleted_at is None:
                issues.append({"severity": "error", "code": "missing-controller", "message": f"active memory {item.namespace}/{item.id} points to missing {item.controller}"})
                continue
            if path.exists():
                try:
                    st = SignedLogMaskState.load(path, map_location="cpu")
                except Exception as exc:
                    severity = "error" if item.deleted_at is None else "warning"
                    issues.append({"severity": severity, "code": "controller-load-failed", "message": f"{item.namespace}/{item.id}: {exc}"})
                    continue
                if item.deleted_at is None:
                    identities.add((st.model_name, st.model_revision, st.tokenizer_name, st.tokenizer_revision))
        if len(identities) > 1:
            issues.append({"severity": "warning", "code": "mixed-model-identities", "message": "active store contains controllers for multiple model/tokenizer identities"})
        for path in self.controllers_dir.rglob("*.pt"):
            if path.resolve() not in referenced:
                issues.append({"severity": "warning", "code": "orphan-controller", "message": str(path.relative_to(self.root))})
        return {
            "status": "fail" if any(x["severity"] == "error" for x in issues) else "pass",
            "active_items": len(active),
            "total_items_including_deleted": len(items),
            "issues": issues,
        }
