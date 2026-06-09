from __future__ import annotations

import inspect
import json
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

YESNO: tuple[str, str] = (" YES", " NO")
SUPPORT_QUERY = "\nIs the claim fully supported by the evidence? Answer YES or NO:"
ISR_SCHEMA_VERSION = 1


class ISROptionalDependencyError(ImportError):
    """Raised when an optional ISR backend dependency is requested but missing."""


class YesProbabilityBackend(Protocol):
    """Minimal verifier backend used by the ISR order-marginal score."""

    def p_yes(
        self,
        text: str,
        *,
        query: str = SUPPORT_QUERY,
        choices: Sequence[str] = YESNO,
    ) -> float:
        ...


@dataclass
class EvidenceClaim:
    """One binary evidence-support example for ISR verifier evaluation."""

    claim: str
    spans: tuple[str, ...]
    supported: bool
    id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.claim = str(self.claim).strip()
        self.spans = tuple(str(s).strip() for s in self.spans if str(s).strip())
        self.supported = bool(self.supported)
        self.metadata = dict(self.metadata or {})
        if not self.claim:
            raise ValueError("EvidenceClaim.claim must be non-empty")
        if not self.spans:
            raise ValueError("EvidenceClaim.spans must contain at least one non-empty span")


@dataclass(frozen=True)
class KVDebiasResult:
    """Diagnostics for the closed-form NTK KV order-debias projection."""

    q_kv: float
    delta_norm: float
    delta_clipped: bool
    event_dim: int
    controller_dim: int
    ridge: float
    max_delta_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HFChoiceVerifierBackend:
    """Standard Hugging Face verifier backend for canonical/marginal ISR scores.

    It computes the probability of the first choice, normally ``" YES"``, by
    normalising the sequence log-likelihood of each supplied choice. This is
    intentionally independent of the legacy KV controller package so the ISR
    gate can be used for ordinary verifier benchmarking even when the optional
    KV-debias backend is unavailable.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        add_special_tokens: bool = False,
        length_normalize_choices: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.add_special_tokens = bool(add_special_tokens)
        self.length_normalize_choices = bool(length_normalize_choices)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover - unusual empty module
            return torch.device("cpu")

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=self.add_special_tokens)
        return [int(x) for x in ids]

    @torch.no_grad()
    def choice_log_probs(
        self,
        text: str,
        *,
        query: str = SUPPORT_QUERY,
        choices: Sequence[str] = YESNO,
    ) -> torch.Tensor:
        choices = _validate_choices(choices)
        prompt_ids = self._encode(str(text) + str(query))
        if not prompt_ids:
            raise ValueError("verifier prompt tokenized to zero tokens")
        scores: list[torch.Tensor] = []
        for choice in choices:
            choice_ids = self._encode(choice)
            if not choice_ids:
                raise ValueError(f"choice {choice!r} tokenized to zero tokens")
            ids = torch.tensor([prompt_ids + choice_ids], dtype=torch.long, device=self.device)
            logits = self.model(ids).logits
            if logits.ndim != 3 or logits.shape[1] < len(prompt_ids) + len(choice_ids) - 1:
                raise ValueError("model returned logits with an unexpected shape")
            # Token at absolute index t is predicted by logits[t-1].
            start = len(prompt_ids) - 1
            stop = len(prompt_ids) + len(choice_ids) - 1
            lp = logits[0, start:stop, :].float().log_softmax(dim=-1)
            target = torch.tensor(choice_ids, dtype=torch.long, device=lp.device)
            val = lp.gather(1, target.view(-1, 1)).sum()
            if self.length_normalize_choices:
                val = val / max(1, len(choice_ids))
            scores.append(val.detach().cpu())
        return torch.stack(scores, dim=0)

    def p_yes(
        self,
        text: str,
        *,
        query: str = SUPPORT_QUERY,
        choices: Sequence[str] = YESNO,
    ) -> float:
        logps = self.choice_log_probs(text, query=query, choices=choices).double()
        p = torch.softmax(logps, dim=-1)[0]
        return _validate_probability(float(p.item()), "p_yes")

    @torch.no_grad()
    def p_yes_batch(
        self,
        texts: Sequence[str],
        *,
        query: str = SUPPORT_QUERY,
        choices: Sequence[str] = YESNO,
        batch_size: int | None = None,
    ) -> list[float]:
        """Fused exact permutation scoring.

        Scores every (text, choice) branch in batched forwards instead of one
        ``model()`` call per branch. For a causal LM with right padding and an
        attention mask, the real-position logits are identical to the unpadded
        single-sequence forward, so the returned probabilities match looping
        :meth:`p_yes` to floating-point tolerance (bit-identical on CPU).
        """
        choices = _validate_choices(choices)
        texts = [str(t) for t in texts]
        if not texts:
            return []
        choice_ids_list: list[list[int]] = []
        for choice in choices:
            cids = self._encode(choice)
            if not cids:
                raise ValueError(f"choice {choice!r} tokenized to zero tokens")
            choice_ids_list.append(cids)
        seqs: list[tuple[int, int, int, int, list[int]]] = []
        for ti, text in enumerate(texts):
            prompt_ids = self._encode(str(text) + str(query))
            if not prompt_ids:
                raise ValueError("verifier prompt tokenized to zero tokens")
            for ci, cids in enumerate(choice_ids_list):
                seqs.append((ti, ci, len(prompt_ids), len(cids), prompt_ids + cids))
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
        n = len(seqs)
        bs = int(batch_size) if batch_size else n
        if bs <= 0:
            raise ValueError("batch_size must be positive")
        logps: list[list[Any]] = [[None] * len(choices) for _ in texts]
        for b0 in range(0, n, bs):
            batch = seqs[b0 : b0 + bs]
            maxlen = max(len(srow[4]) for srow in batch)
            ids = torch.full((len(batch), maxlen), int(pad_id), dtype=torch.long, device=self.device)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long, device=self.device)
            for r, (_, _, _, _, srow) in enumerate(batch):
                ids[r, : len(srow)] = torch.tensor(srow, dtype=torch.long, device=self.device)
                mask[r, : len(srow)] = 1
            logits = self.model(ids, attention_mask=mask).logits
            if logits.ndim != 3:
                raise ValueError("model returned logits with an unexpected shape")
            for r, (ti, ci, plen, clen, _) in enumerate(batch):
                start = plen - 1
                stop = plen + clen - 1
                lp = logits[r, start:stop, :].float().log_softmax(dim=-1)
                target = torch.tensor(choice_ids_list[ci], dtype=torch.long, device=lp.device)
                val = lp.gather(1, target.view(-1, 1)).sum()
                if self.length_normalize_choices:
                    val = val / max(1, clen)
                logps[ti][ci] = val.detach().cpu()
        out: list[float] = []
        for ti in range(len(texts)):
            lp = torch.stack(logps[ti], dim=0).double()
            p = torch.softmax(lp, dim=-1)[0]
            out.append(_validate_probability(float(p.item()), "p_yes"))
        return out


class KVDeltaBayesNTKBackendAdapter:
    """Adapter for the legacy ``kv_delta_bayes_ntk`` backend used by the prototype.

    The adapter is intentionally thin: it preserves the prototype's cache-based
    event-logit path, while moving dependency loading, reproducibility checks,
    and closed-form ridge solving into ntkmirror. The optional package must be
    importable for this backend to be constructed.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._make_dense_spec = _load_make_dense_spec()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: str = "cuda:0",
        dtype: str = "float32",
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        extra_backend_kwargs: Mapping[str, Any] | None = None,
    ) -> "KVDeltaBayesNTKBackendAdapter":
        HFBackend = _load_legacy_hf_backend()
        kwargs: dict[str, Any] = {
            "device": device,
            "dtype": _normalise_legacy_dtype(dtype),
        }
        optional_kwargs = {
            "trust_remote_code": bool(trust_remote_code),
            "revision": revision,
            "local_files_only": bool(local_files_only),
            "cache_dir": cache_dir,
        }
        kwargs.update(_filter_backend_kwargs(HFBackend, optional_kwargs))
        if extra_backend_kwargs:
            kwargs.update(dict(extra_backend_kwargs))
        backend = HFBackend(model_name, **kwargs)
        return cls(backend)

    @property
    def device(self):
        return getattr(self.backend, "device", torch.device("cpu"))

    @property
    def dtype(self):
        return getattr(self.backend, "dtype", torch.float32)

    def p_yes(
        self,
        text: str,
        *,
        query: str = SUPPORT_QUERY,
        choices: Sequence[str] = YESNO,
    ) -> float:
        choices = _validate_choices(choices)
        _, pkv = self.backend.prefill_cache(text)
        logits = self.backend.event_logits_from_cache(pkv, query, choices)
        return _softmax_first_probability(logits)

    def kv_debias_probability(
        self,
        text: str,
        q_bar: float,
        *,
        query: str = SUPPORT_QUERY,
        choices: Sequence[str] = YESNO,
        use_layers: int = 4,
        ridge: float = 1e-3,
        max_delta_norm: float = 10.0,
        kinds: Sequence[str] = ("v",),
        position_policy: str = "all",
    ) -> KVDebiasResult:
        return score_kv_debias_probability(
            self.backend,
            text,
            q_bar,
            query=query,
            choices=choices,
            use_layers=use_layers,
            ridge=ridge,
            max_delta_norm=max_delta_norm,
            kinds=kinds,
            position_policy=position_policy,
            make_dense_spec_fn=self._make_dense_spec,
        )


def _load_legacy_hf_backend():
    try:
        from kv_delta_bayes_ntk.hf_backend import HFBackend
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ISROptionalDependencyError(
            "The ISR KV-debias backend requires the optional kv_delta_bayes_ntk package. "
            "Place that package on PYTHONPATH or run ISR without --fit-controller."
        ) from exc
    return HFBackend


def _load_make_dense_spec():
    try:
        from kv_delta_bayes_ntk.controller import make_dense_spec
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ISROptionalDependencyError(
            "The ISR KV-debias path requires kv_delta_bayes_ntk.controller.make_dense_spec."
        ) from exc
    return make_dense_spec


def _normalise_legacy_dtype(dtype: str) -> str:
    table = {
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    key = str(dtype).lower().strip()
    if key == "auto":
        raise ValueError("the legacy KV ISR backend does not support dtype='auto'; use fp32, fp16, or bf16")
    if key not in table:
        raise ValueError("dtype must be one of auto, fp32, fp16, bf16, float32, float16, bfloat16")
    return table[key]


def _filter_backend_kwargs(cls_or_fn: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Pass supported reproducibility kwargs, refusing to silently ignore them."""

    sig = inspect.signature(cls_or_fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    out: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        if has_varkw or key in params:
            out[key] = value
        else:
            ignored.append(key)
    if ignored:
        raise ValueError(
            "Installed kv_delta_bayes_ntk.HFBackend does not accept reproducibility/security "
            f"argument(s) {ignored}; refusing to silently ignore them."
        )
    return out


def _validate_choices(choices: Sequence[str]) -> tuple[str, str]:
    vals = tuple(str(x) for x in choices)
    if len(vals) != 2 or not vals[0] or not vals[1]:
        raise ValueError("ISR verifier choices must contain exactly two non-empty strings")
    return vals[0], vals[1]


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    # Numerical softmax can very rarely produce tiny excursions after conversion.
    if value < -1e-7 or value > 1.0 + 1e-7:
        raise ValueError(f"{name}={value} is outside [0, 1]")
    return min(1.0, max(0.0, value))


def _finite_float(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _softmax_first_probability(logits: torch.Tensor | Sequence[float]) -> float:
    lg = torch.as_tensor(logits).reshape(-1)
    if lg.numel() != 2:
        raise ValueError(f"expected two verifier event logits, got {int(lg.numel())}")
    if not bool(torch.isfinite(lg).all()):
        raise ValueError("verifier event logits contain NaN or Inf")
    return _validate_probability(float(torch.softmax(lg.double(), dim=-1)[0].detach().cpu().item()), "p_yes")


def centered(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    return x - x.mean(dim=-1, keepdim=True)


def centered_log_probs_from_probs(probs: torch.Tensor | Sequence[float], *, eps: float = 1e-6) -> torch.Tensor:
    p = torch.as_tensor(probs)
    if p.ndim != 1:
        raise ValueError("probs must be a rank-1 vector")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("probs contain NaN or Inf")
    p = p.clamp(float(eps), 1.0 - float(eps))
    p = p / p.sum().clamp_min(float(eps))
    lp = torch.log(p)
    return lp - lp.mean(dim=-1, keepdim=True)


def solve_ridge_projection(B: torch.Tensor, d: torch.Tensor, *, ridge: float = 1e-3) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve ``min_q ||Bq - d||² + ridge ||q||²`` robustly.

    The legacy prototype imported this solve from ``kv_delta_bayes_ntk``. V2
    keeps the solver local so diagnostics and failure modes are testable even
    when the optional backend is absent.
    """

    ridge = _finite_float(ridge, "ridge")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    B = torch.as_tensor(B)
    d = torch.as_tensor(d, device=B.device, dtype=B.dtype).reshape(-1)
    if B.ndim != 2:
        raise ValueError("B must be rank-2 [event_dim, controller_dim]")
    if d.numel() != B.shape[0]:
        raise ValueError("d length must equal B.shape[0]")
    if not bool(torch.isfinite(B).all()) or not bool(torch.isfinite(d).all()):
        raise ValueError("B and d must be finite")
    event_dim, controller_dim = int(B.shape[0]), int(B.shape[1])
    if event_dim == 0 or controller_dim == 0:
        raise ValueError("B must have non-zero event and controller dimensions")

    work_dtype = torch.float64 if B.dtype in {torch.float16, torch.bfloat16, torch.float32} else B.dtype
    Bw = B.to(dtype=work_dtype)
    dw = d.to(dtype=work_dtype)
    eye_event = torch.eye(event_dim, dtype=work_dtype, device=B.device)
    try:
        alpha = torch.linalg.solve(Bw @ Bw.T + ridge * eye_event, dw)
    except RuntimeError:  # pragma: no cover - ridge should make this rare
        alpha = torch.linalg.lstsq(Bw @ Bw.T + ridge * eye_event, dw.unsqueeze(1)).solution.squeeze(1)
    q = Bw.T @ alpha
    realized = Bw @ q
    residual = torch.linalg.vector_norm(realized - dw).item()
    target_norm = max(float(torch.linalg.vector_norm(dw).item()), 1e-30)
    stats = {
        "event_dim": event_dim,
        "controller_dim": controller_dim,
        "ridge": float(ridge),
        "residual_norm": float(residual),
        "relative_residual": float(residual / target_norm),
    }
    return q.to(dtype=B.dtype), stats


def split_sentences(text: Any, cap: int, *, max_chars: int | None = None) -> list[str]:
    cap = _positive_int(cap, "cap")
    raw = _text_field(text)
    if max_chars is not None:
        max_chars = _positive_int(max_chars, "max_chars")
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", raw) if x.strip()]
    if not sents:
        sents = [raw.strip()]
    out: list[str] = []
    for sent in sents[:cap]:
        sent = sent[:max_chars] if max_chars is not None else sent
        if sent.strip():
            out.append(sent.strip())
    if not out:
        fallback = raw[: max_chars or 400].strip()
        out = [fallback] if fallback else [""]
    return out


def _text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_text_field(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_text_field(v) for v in value)
    return str(value)


def verbalize(spans: Sequence[str], claim: str, *, max_span_chars: int | None = None) -> str:
    clean: list[str] = []
    for span in spans:
        s = str(span).strip()
        if not s:
            continue
        if max_span_chars is not None:
            s = s[: _positive_int(max_span_chars, "max_span_chars")].strip()
        clean.append(s)
    if not clean:
        raise ValueError("cannot verbalize evidence with zero non-empty spans")
    claim = str(claim).strip()
    if not claim:
        raise ValueError("cannot verbalize empty claim")
    return "Evidence:\n" + "\n".join(f"- {s}" for s in clean) + f"\nClaim: {claim}"


def load_evidence_claims_jsonl(path: str | Path, *, cap_spans: int = 5, max_chars: int = 240) -> list[EvidenceClaim]:
    """Load custom ISR rows from JSONL.

    Accepted rows:
      ``{"claim": str, "spans": [str, ...], "supported": bool}``
      ``{"claim": str, "evidence": str, "supported": bool}``

    ``label`` may be used instead of ``supported`` with values such as
    ``supported``, ``refuted``, ``hallucinated``, or ``unsupported``.
    """

    p = Path(path)
    rows: list[EvidenceClaim] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_no} invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, Mapping):
                raise ValueError(f"{p}:{line_no} row must be an object")
            if not isinstance(obj.get("claim"), str):
                raise ValueError(f"{p}:{line_no} claim must be a string")
            if "spans" in obj:
                raw_spans = obj["spans"]
                if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
                    raise ValueError(f"{p}:{line_no} spans must be a list of strings")
                spans = tuple(str(x).strip() for x in raw_spans if str(x).strip())[:cap_spans]
            elif "evidence" in obj:
                spans = tuple(split_sentences(obj["evidence"], cap_spans, max_chars=max_chars))
            else:
                raise ValueError(f"{p}:{line_no} must contain spans or evidence")
            supported = _parse_supported_label(obj, path=p, line=line_no)
            metadata = {k: v for k, v in obj.items() if k not in {"claim", "spans", "evidence", "supported", "label"}}
            rows.append(EvidenceClaim(obj["claim"], spans, supported, id=obj.get("id", line_no), metadata=metadata))
    if not rows:
        raise ValueError(f"no ISR rows found in {p}")
    return rows


def _parse_supported_label(row: Mapping[str, Any], *, path: Path, line: int) -> bool:
    if "supported" in row:
        val = row["supported"]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)) and val in {0, 1}:
            return bool(val)
        if isinstance(val, str):
            return _parse_label_string(val, path=path, line=line)
        raise ValueError(f"{path}:{line} supported must be bool, 0/1, or a supported/refuted label string")
    if "label" in row:
        return _parse_label_string(str(row["label"]), path=path, line=line)
    raise ValueError(f"{path}:{line} must contain supported or label")


def _parse_label_string(label: str, *, path: Path, line: int) -> bool:
    lab = label.strip().lower().replace("-", "_").replace(" ", "_")
    if lab in {"support", "supported", "supports", "true", "yes", "entails", "entailed"}:
        return True
    if lab in {"unsupported", "not_supported", "refute", "refuted", "false", "no", "hallucinated", "hallu"}:
        return False
    raise ValueError(f"{path}:{line} unsupported ISR label {label!r}")


def load_isr_dataset(name: str, n: int, *, cap_spans: int = 5, max_chars: int = 240) -> list[EvidenceClaim]:
    """Load one of the built-in ISR verifier datasets used by the prototype."""

    n = _positive_int(n, "n")
    cap_spans = _positive_int(cap_spans, "cap_spans")
    max_chars = _positive_int(max_chars, "max_chars")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ISROptionalDependencyError("load_isr_dataset requires the optional datasets package") from exc

    name = str(name).lower().strip()
    rows: list[EvidenceClaim] = []
    if name == "vitaminc":
        ds = load_dataset("tals/vitaminc", split="test")
        for row_idx, ex in enumerate(ds):
            lab = str(ex.get("label", "")).upper()
            if lab.startswith("NOT"):
                continue
            rows.append(
                EvidenceClaim(
                    str(ex.get("claim", "")),
                    tuple(split_sentences(ex.get("evidence", ""), cap_spans, max_chars=max_chars)),
                    lab.startswith("SUPPORT"),
                    id=row_idx,
                    metadata={"dataset": name},
                )
            )
            if len(rows) >= n:
                break
    elif name == "hotpot":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for row_idx, ex in enumerate(ds):
            titles = [str(t) for t in ex["context"]["title"]]
            sents = ex["context"]["sentences"]
            gold_titles = {str(t) for t in ex["supporting_facts"]["title"]}
            para = {title: _text_field(sentences)[:max_chars] for title, sentences in zip(titles, sents, strict=True)}
            gold = [para[t] for t in titles if t in gold_titles and para.get(t)]
            distractors = [para[t] for t in titles if t not in gold_titles and para.get(t)]
            if not gold:
                continue
            q = str(ex.get("question", "")).strip().rstrip("?")
            claim = f"The answer to '{q}?' is {str(ex.get('answer', '')).strip()}."
            rows.append(
                EvidenceClaim(
                    claim,
                    tuple((gold + distractors[:2])[:cap_spans]),
                    True,
                    id=f"{row_idx}:supported",
                    metadata={"dataset": name},
                )
            )
            if len(rows) >= n:
                break
            if distractors:
                rows.append(
                    EvidenceClaim(
                        claim,
                        tuple(distractors[:cap_spans]),
                        False,
                        id=f"{row_idx}:distractor",
                        metadata={"dataset": name},
                    )
                )
            if len(rows) >= n:
                break
    elif name == "ragtruth":
        ds = load_dataset("wandb/RAGTruth-processed", split="test")
        pos_target = (n + 1) // 2
        neg_target = n // 2
        pos_n = 0
        neg_n = 0
        for row_idx, ex in enumerate(ds):
            labels = ex.get("hallucination_labels_processed") or {}
            hallu = isinstance(labels, Mapping) and any(int(v) > 0 for v in labels.values())
            if hallu and neg_n >= neg_target:
                continue
            if (not hallu) and pos_n >= pos_target:
                continue
            ctx = ex.get("context") or ex.get("input_str") or ""
            out = str(ex.get("output", ""))[:300]
            if not out.strip():
                continue
            rows.append(
                EvidenceClaim(
                    out,
                    tuple(split_sentences(ctx, cap_spans, max_chars=max_chars)),
                    not hallu,
                    id=row_idx,
                    metadata={"dataset": name},
                )
            )
            neg_n += int(hallu)
            pos_n += int(not hallu)
            if len(rows) >= n:
                break
    else:
        raise ValueError("dataset must be one of: vitaminc, hotpot, ragtruth")
    if not rows:
        raise ValueError(f"dataset {name!r} produced no ISR rows")
    return rows[:n]


def _call_p_yes(
    backend: Any,
    text: str,
    *,
    query: str = SUPPORT_QUERY,
    choices: Sequence[str] = YESNO,
) -> float:
    choices = _validate_choices(choices)
    if hasattr(backend, "p_yes"):
        fn = backend.p_yes
        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_varkw or "query" in params or "choices" in params:
            return _validate_probability(fn(text, query=query, choices=choices), "p_yes")
        # Compatibility with tiny/prototype backends that intentionally accept
        # only the verbalized text. Avoid a try/except TypeError fallback here:
        # TypeError inside a backend implementation should surface as a real bug.
        return _validate_probability(fn(text), "p_yes")
    if hasattr(backend, "prefill_cache") and hasattr(backend, "event_logits_from_cache"):
        _, pkv = backend.prefill_cache(text)
        logits = backend.event_logits_from_cache(pkv, query, choices)
        return _softmax_first_probability(logits)
    raise TypeError("backend must expose p_yes(...) or prefill_cache/event_logits_from_cache(...)")


def make_orderings(n_spans: int, num_orderings: int, rng: np.random.Generator) -> list[list[int]]:
    n_spans = _positive_int(n_spans, "n_spans")
    num_orderings = _positive_int(num_orderings, "num_orderings")
    orderings = [list(range(n_spans))]
    for _ in range(1, num_orderings):
        orderings.append([int(x) for x in rng.permutation(n_spans).tolist()])
    return orderings


def score_claim(
    backend: Any,
    example: EvidenceClaim,
    *,
    row_index: int,
    seed: int = 0,
    num_orderings: int = 6,
    dispersion_penalty: float = 1.0,
    query: str = SUPPORT_QUERY,
    choices: Sequence[str] = YESNO,
    fit_controller: bool = False,
    use_layers: int = 4,
    kv_ridge: float = 1e-3,
    kv_max_norm: float = 10.0,
    include_ordering_scores: bool = False,
    include_orderings: bool = False,
    include_raw: bool = False,
    strict_controller_errors: bool = False,
) -> dict[str, Any]:
    choices = _validate_choices(choices)
    dispersion_penalty = _finite_float(dispersion_penalty, "dispersion_penalty")
    rng = np.random.default_rng(int(seed) * 100003 + int(row_index))
    orderings = make_orderings(len(example.spans), num_orderings, rng)
    texts = [verbalize([example.spans[j] for j in order], example.claim) for order in orderings]
    if hasattr(backend, "p_yes_batch"):
        qs = [float(q) for q in backend.p_yes_batch(texts, query=query, choices=choices)]
    else:
        qs = [_call_p_yes(backend, text, query=query, choices=choices) for text in texts]
    q_arr = np.asarray(qs, dtype=float)
    if not np.isfinite(q_arr).all():
        raise ValueError("one or more ISR ordering scores is non-finite")
    rec: dict[str, Any] = {
        "status": "ok",
        "row_index": int(row_index),
        "id": example.id,
        "supported": int(example.supported),
        "n_spans": len(example.spans),
        "num_orderings": int(num_orderings),
        "q_canon": float(q_arr[0]),
        "q_marg": float(q_arr.mean()),
        "q_std": float(q_arr.std(ddof=0)),
        "isr": float(q_arr.mean() - dispersion_penalty * q_arr.std(ddof=0)),
    }
    if include_ordering_scores:
        rec["ordering_scores"] = [float(x) for x in q_arr.tolist()]
    if include_orderings:
        rec["orderings"] = orderings
    if include_raw:
        rec["claim"] = example.claim
        rec["spans"] = list(example.spans)
    if fit_controller and len(example.spans) > 1:
        try:
            if hasattr(backend, "kv_debias_probability"):
                kv = backend.kv_debias_probability(
                    verbalize(example.spans, example.claim),
                    rec["q_marg"],
                    query=query,
                    choices=choices,
                    use_layers=use_layers,
                    ridge=kv_ridge,
                    max_delta_norm=kv_max_norm,
                )
            else:
                kv = score_kv_debias_probability(
                    backend,
                    verbalize(example.spans, example.claim),
                    rec["q_marg"],
                    query=query,
                    choices=choices,
                    use_layers=use_layers,
                    ridge=kv_ridge,
                    max_delta_norm=kv_max_norm,
                )
            rec.update(kv.to_dict())
            rec["kv_status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - row-level benchmark robustness
            if strict_controller_errors:
                raise
            rec["kv_status"] = "error"
            rec["kv_error"] = repr(exc)[:500]
    elif fit_controller:
        rec["kv_status"] = "skipped_single_span"
    return rec


def _legacy_cache_dims(pkv: Any) -> tuple[int, int, int]:
    if not isinstance(pkv, (list, tuple)) or not pkv:
        raise ValueError("legacy KV backend returned an unsupported cache object")
    first = pkv[0]
    if not isinstance(first, (list, tuple)) or not first:
        raise ValueError("legacy KV cache layer must be a non-empty tuple/list")
    key = first[0]
    if not hasattr(key, "shape") or len(key.shape) < 3:
        raise ValueError("legacy KV cache key tensor must have at least 3 dimensions")
    return int(len(pkv)), int(key.shape[1]), int(key.shape[2])


def score_kv_debias_probability(
    backend: Any,
    text: str,
    q_bar: float,
    *,
    query: str = SUPPORT_QUERY,
    choices: Sequence[str] = YESNO,
    use_layers: int = 4,
    ridge: float = 1e-3,
    max_delta_norm: float = 10.0,
    kinds: Sequence[str] = ("v",),
    position_policy: str = "all",
    make_dense_spec_fn: Callable[..., Any] | None = None,
) -> KVDebiasResult:
    """Apply the prototype's closed-form NTK KV order-debias projection.

    ``backend`` must expose the legacy KV-controller methods:
    ``prefill_cache``, ``make_layout_from_cache``, and
    ``event_logits_from_cache``. The function computes a local Jacobian of
    centered YES/NO event logits with respect to a dense KV delta, solves a
    ridge projection toward the order-marginal target probability, clips the
    delta norm, and reports the debiased YES probability plus diagnostics.
    """

    choices = _validate_choices(choices)
    q_bar = _validate_probability(q_bar, "q_bar")
    use_layers = _positive_int(use_layers, "use_layers")
    ridge = _finite_float(ridge, "ridge")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    max_delta_norm = _finite_float(max_delta_norm, "max_delta_norm")
    if max_delta_norm <= 0.0:
        raise ValueError("max_delta_norm must be positive")
    if make_dense_spec_fn is None:
        make_dense_spec_fn = _load_make_dense_spec()
    for method in ("prefill_cache", "make_layout_from_cache", "event_logits_from_cache"):
        if not hasattr(backend, method):
            raise TypeError(f"KV debias backend must expose {method}(...)")

    _, pkv = backend.prefill_cache(text)
    n_layers, n_heads, seq_len = _legacy_cache_dims(pkv)
    spec = make_dense_spec_fn(
        n_layers=min(use_layers, n_layers),
        n_heads=n_heads,
        seq_len=seq_len,
        kinds=tuple(kinds),
        position_policy=str(position_policy),
        name="ntkmirror_isr_dbias",
    )
    layout = backend.make_layout_from_cache(pkv, spec)
    q0 = layout.zeros_flat(requires_grad=True)
    logits0 = centered(backend.event_logits_from_cache(pkv, query, choices, layout=layout, q=q0))
    logits0 = torch.as_tensor(logits0)
    if logits0.ndim != 1 or logits0.numel() != 2:
        raise ValueError("KV debias expects two centered event logits")
    if not bool(torch.isfinite(logits0).all()):
        raise ValueError("baseline KV event logits contain NaN or Inf")
    grads = []
    for i in range(int(logits0.numel())):
        grad = torch.autograd.grad(logits0[i], q0, retain_graph=i < int(logits0.numel()) - 1)[0]
        grads.append(grad.detach())
    B = torch.stack(grads, dim=0)
    dtype = getattr(backend, "dtype", logits0.dtype)
    device = getattr(backend, "device", logits0.device)
    target = torch.as_tensor([q_bar, 1.0 - q_bar], dtype=dtype, device=device)
    d = (centered_log_probs_from_probs(target).to(device=logits0.device, dtype=logits0.dtype) - logits0.detach()).detach()
    q, stats = solve_ridge_projection(B, d, ridge=ridge)
    q = q.to(device=device, dtype=dtype)
    delta_norm = float(torch.linalg.vector_norm(q.detach().float()).item())
    clipped = False
    if delta_norm > max_delta_norm:
        q = q * (max_delta_norm / (delta_norm + 1e-12))
        delta_norm = float(torch.linalg.vector_norm(q.detach().float()).item())
        clipped = True
    with torch.no_grad():
        logits_final = backend.event_logits_from_cache(pkv, query, choices, layout=layout, q=q.detach())
    return KVDebiasResult(
        q_kv=_softmax_first_probability(logits_final),
        delta_norm=delta_norm,
        delta_clipped=bool(clipped),
        event_dim=int(stats["event_dim"]),
        controller_dim=int(stats["controller_dim"]),
        ridge=float(ridge),
        max_delta_norm=float(max_delta_norm),
    )


def run_isr_auc(
    backend: Any,
    examples: Sequence[EvidenceClaim],
    *,
    seed: int = 0,
    num_orderings: int = 6,
    dispersion_penalty: float = 1.0,
    leak: float = 0.1,
    shard: int = 0,
    nshards: int = 1,
    query: str = SUPPORT_QUERY,
    choices: Sequence[str] = YESNO,
    fit_controller: bool = False,
    use_layers: int = 4,
    kv_ridge: float = 1e-3,
    kv_max_norm: float = 10.0,
    include_ordering_scores: bool = False,
    include_orderings: bool = False,
    include_raw: bool = False,
    fail_fast: bool = False,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples must be non-empty")
    _positive_int(num_orderings, "num_orderings")
    _finite_float(dispersion_penalty, "dispersion_penalty")
    validate_leak(leak)
    shard = int(shard)
    nshards = _positive_int(nshards, "nshards")
    if shard < 0 or shard >= nshards:
        raise ValueError("shard must be in [0, nshards)")
    mine = [(i, ex) for i, ex in enumerate(examples) if i % nshards == shard]
    if not mine:
        raise ValueError("selected shard contains no examples")
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for local_idx, (row_index, ex) in enumerate(mine, start=1):
        try:
            rows.append(
                score_claim(
                    backend,
                    ex,
                    row_index=row_index,
                    seed=seed,
                    num_orderings=num_orderings,
                    dispersion_penalty=dispersion_penalty,
                    query=query,
                    choices=choices,
                    fit_controller=fit_controller,
                    use_layers=use_layers,
                    kv_ridge=kv_ridge,
                    kv_max_norm=kv_max_norm,
                    include_ordering_scores=include_ordering_scores,
                    include_orderings=include_orderings,
                    include_raw=include_raw,
                    strict_controller_errors=fail_fast,
                )
            )
        except Exception as exc:  # noqa: BLE001 - benchmark row isolation
            if fail_fast:
                raise
            rows.append({"status": "error", "row_index": int(row_index), "error": repr(exc)[:500]})
        if progress_callback is not None:
            progress_callback(local_idx, len(mine), time.time() - t0)
    summary = summarize_rows(rows, leak=leak)
    return {
        "schema_version": ISR_SCHEMA_VERSION,
        "feature": "isr_kv_order_debias_auc",
        "created_at": time.time(),
        "n_loaded": len(examples),
        "n_scored": len(mine),
        "shard": shard,
        "nshards": nshards,
        "query": query,
        "choices": list(_validate_choices(choices)),
        "num_orderings": int(num_orderings),
        "dispersion_penalty": float(dispersion_penalty),
        "leak": float(leak),
        "fit_controller": bool(fit_controller),
        "rows": rows,
        "summary": summary,
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be rank-1")
    if not np.isfinite(arr).all():
        raise ValueError("values contain NaN or Inf")
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0  # one-indexed average rank for ties
        ranks[order[i:j]] = avg
        i = j
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[int | bool]) -> float | None:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.ndim != 1 or y.ndim != 1 or s.shape[0] != y.shape[0]:
        raise ValueError("scores and labels must be equally sized rank-1 arrays")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or Inf")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    pos_n = int((y == 1).sum())
    neg_n = int((y == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return None
    ranks = average_ranks(s)
    return float((ranks[y == 1].sum() - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n))


def validate_leak(leak: float) -> float:
    leak = _finite_float(leak, "leak")
    if leak < 0.0 or leak > 1.0:
        raise ValueError("leak must be in [0, 1]")
    return leak


def threshold_at_leak(scores: Sequence[float], labels: Sequence[int | bool], *, leak: float = 0.1) -> float | None:
    leak = validate_leak(leak)
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.ndim != 1 or y.ndim != 1 or s.shape[0] != y.shape[0]:
        raise ValueError("scores and labels must be equally sized rank-1 arrays")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or Inf")
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return None
    if leak >= 1.0:
        return float("-inf")
    if leak <= 0.0:
        return float(neg[-1])
    idx = int(np.ceil((1.0 - leak) * len(neg))) - 1
    idx = max(0, min(len(neg) - 1, idx))
    return float(neg[idx])


def tpr_at_leak(scores: Sequence[float], labels: Sequence[int | bool], leak: float = 0.1) -> float | None:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    thr = threshold_at_leak(s, y, leak=leak)
    if thr is None:
        return None
    pos = s[y == 1]
    if len(pos) == 0:
        return None
    return float((pos > thr).mean())


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, leak: float = 0.1) -> dict[str, Any]:
    validate_leak(leak)
    ok = [r for r in rows if r.get("status") == "ok"]
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_error": len(rows) - len(ok),
        "scores": {},
    }
    if ok:
        labels_all = [int(r["supported"]) for r in ok]
        out["pos"] = int(sum(labels_all))
        out["neg"] = int(len(labels_all) - sum(labels_all))
    for col in ("q_canon", "q_marg", "isr", "q_kv"):
        sc: list[float] = []
        yy: list[int] = []
        for r in ok:
            if col not in r:
                continue
            try:
                val = float(r[col])
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                sc.append(val)
                yy.append(int(r["supported"]))
        if not sc:
            continue
        out["scores"][col] = {
            "n": len(sc),
            "pos": int(sum(yy)),
            "neg": int(len(yy) - sum(yy)),
            "auroc": auroc(sc, yy),
            "tpr_at_leak": tpr_at_leak(sc, yy, leak),
        }
    return out


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, prefix=f".{p.name}.", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    tmp.replace(p)
