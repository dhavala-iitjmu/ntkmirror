from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import torch
from torch import nn

from .data import Example, batches, make_batch
from .layers import find_decoder_layers, infer_hidden_size, parse_layers
from .losses import causal_loss_from_logits, token_accuracy_from_logits


@dataclass
class SignedLogMaskState:
    """Serializable state for a signed log-mask controller.

    A selected gate j corresponds to one decoder layer and one residual-stream
    channel. When attached, the layer output is multiplied by exp(s_j), with
    s_j bounded to [-max_log_gate, max_log_gate].
    """

    layer_path: str
    n_layers: int
    hidden_size: int
    layer_indices: torch.Tensor
    channel_indices: torch.Tensor
    raw: torch.Tensor
    max_log_gate: float
    model_name: str | None = None
    model_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    hook_site: str = "layer_output"
    theory_version: str = "activation_control"
    created_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    VALID_HOOK_SITES: ClassVar[set[str]] = {"layer_output", "layer_input"}
    VALID_THEORY_VERSIONS: ClassVar[set[str]] = {"activation_control"}

    def __post_init__(self) -> None:
        self.layer_path = str(self.layer_path)
        self.n_layers = int(self.n_layers)
        self.hidden_size = int(self.hidden_size)
        self.max_log_gate = float(self.max_log_gate)
        self.model_name = None if self.model_name is None else str(self.model_name)
        self.model_revision = None if self.model_revision is None else str(self.model_revision)
        self.tokenizer_name = None if self.tokenizer_name is None else str(self.tokenizer_name)
        self.tokenizer_revision = None if self.tokenizer_revision is None else str(self.tokenizer_revision)
        self.hook_site = str(self.hook_site)
        self.theory_version = str(self.theory_version)
        if self.created_at is not None:
            self.created_at = float(self.created_at)
        self.metadata = dict(self.metadata or {})

        self.layer_indices = torch.as_tensor(self.layer_indices, dtype=torch.long).detach().cpu()
        self.channel_indices = torch.as_tensor(self.channel_indices, dtype=torch.long).detach().cpu()
        self.raw = torch.as_tensor(self.raw, dtype=torch.float32).detach().cpu()
        self.validate()

    @property
    def n_gates(self) -> int:
        return int(self.layer_indices.numel())

    def validate(self, *, max_gates: int | None = None) -> None:
        if not self.layer_path:
            raise ValueError("layer_path must be a non-empty string")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not math.isfinite(self.max_log_gate) or self.max_log_gate <= 0.0:
            raise ValueError("max_log_gate must be finite and positive")
        if self.hook_site not in self.VALID_HOOK_SITES:
            raise ValueError(f"hook_site must be one of {sorted(self.VALID_HOOK_SITES)}")
        if self.theory_version not in self.VALID_THEORY_VERSIONS:
            raise ValueError(f"theory_version must be one of {sorted(self.VALID_THEORY_VERSIONS)}")
        if self.layer_indices.ndim != 1 or self.channel_indices.ndim != 1 or self.raw.ndim != 1:
            raise ValueError("layer_indices, channel_indices, and raw must be rank-1 tensors")
        if self.layer_indices.numel() == 0:
            raise ValueError("controller must contain at least one gate")
        if self.layer_indices.numel() != self.channel_indices.numel():
            raise ValueError("layer_indices and channel_indices must have the same length")
        if self.raw.numel() != self.layer_indices.numel():
            raise ValueError("raw must have the same length as layer_indices")
        if max_gates is not None and self.layer_indices.numel() > int(max_gates):
            raise ValueError(f"controller has {self.layer_indices.numel()} gates, exceeding max_gates={max_gates}")
        if self.layer_indices.numel() > self.n_layers * self.hidden_size:
            raise ValueError("controller has more gates than available layer/channel positions")
        if bool((self.layer_indices < 0).any()) or bool((self.layer_indices >= self.n_layers).any()):
            raise ValueError("layer_indices contain values outside [0, n_layers)")
        if bool((self.channel_indices < 0).any()) or bool((self.channel_indices >= self.hidden_size).any()):
            raise ValueError("channel_indices contain values outside [0, hidden_size)")
        if not bool(torch.isfinite(self.raw).all()):
            raise ValueError("raw contains NaN or Inf")
        keys = set()
        for layer, channel in zip(self.layer_indices.tolist(), self.channel_indices.tolist(), strict=True):
            key = (int(layer), int(channel))
            if key in keys:
                raise ValueError(f"duplicate gate key {key}; duplicate gates must be coalesced before saving")
            keys.add(key)

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "layer_path": self.layer_path,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "layer_indices": self.layer_indices.detach().cpu(),
            "channel_indices": self.channel_indices.detach().cpu(),
            "raw": self.raw.detach().cpu(),
            "max_log_gate": float(self.max_log_gate),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "hook_site": self.hook_site,
            "theory_version": self.theory_version,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "SignedLogMaskState":
        if not isinstance(obj, Mapping):
            raise ValueError("controller payload must be a mapping")
        try:
            schema_version = int(obj.get("schema_version", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("controller schema_version must be an integer") from exc
        if schema_version not in {1, 2}:
            raise ValueError(f"unsupported controller schema_version={schema_version}")
        required = {"layer_path", "n_layers", "hidden_size", "layer_indices", "channel_indices", "raw", "max_log_gate"}
        missing = sorted(required.difference(obj))
        if missing:
            raise ValueError(f"controller payload missing required keys: {missing}")
        return cls(
            layer_path=str(obj["layer_path"]),
            n_layers=int(obj["n_layers"]),
            hidden_size=int(obj["hidden_size"]),
            layer_indices=torch.as_tensor(obj["layer_indices"], dtype=torch.long),
            channel_indices=torch.as_tensor(obj["channel_indices"], dtype=torch.long),
            raw=torch.as_tensor(obj["raw"], dtype=torch.float32),
            max_log_gate=float(obj["max_log_gate"]),
            model_name=obj.get("model_name"),
            model_revision=obj.get("model_revision"),
            tokenizer_name=obj.get("tokenizer_name"),
            tokenizer_revision=obj.get("tokenizer_revision"),
            created_at=obj.get("created_at"),
            hook_site=str(obj.get("hook_site", "layer_output")),
            theory_version=str(obj.get("theory_version", "activation_control")),
            metadata=dict(obj.get("metadata") or {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location="cpu",
        *,
        unsafe_legacy_load: bool = False,
        max_gates: int | None = None,
    ) -> "SignedLogMaskState":
        # Controller files are expected to be small tensor dictionaries.
        # `weights_only=True` avoids pickle object loading. This project requires
        # torch>=2.3, so the unsafe pickle fallback is opt-in only for trusted
        # legacy files.
        try:
            obj = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # older torch
            if not unsafe_legacy_load:
                raise RuntimeError(
                    "this torch version does not support weights_only=True; upgrade torch or pass "
                    "unsafe_legacy_load=True only for trusted controller files"
                )
            obj = torch.load(path, map_location=map_location)
        if not isinstance(obj, Mapping):
            raise ValueError("controller payload must be a mapping")
        state = cls.from_dict(obj)
        state.validate(max_gates=max_gates)
        return state


class _SignedLogMaskModule(nn.Module):
    """Small trainable module attached by forward hooks."""

    VALID_HOOK_SITES = SignedLogMaskState.VALID_HOOK_SITES

    def __init__(
        self,
        layers: Sequence[nn.Module],
        layer_indices: torch.Tensor,
        channel_indices: torch.Tensor,
        *,
        hidden_size: int,
        max_log_gate: float,
        hook_site: str = "layer_output",
        raw_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        layers_ref = tuple(layers)
        if not layers_ref:
            raise ValueError("layers must contain at least one decoder layer")
        self.hidden_size = int(hidden_size)
        self.max_log_gate = float(max_log_gate)
        self.hook_site = str(hook_site)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not math.isfinite(self.max_log_gate) or self.max_log_gate <= 0.0:
            raise ValueError("max_log_gate must be finite and positive")
        if self.hook_site not in self.VALID_HOOK_SITES:
            raise ValueError(f"hook_site must be one of {sorted(self.VALID_HOOK_SITES)}")
        # Keep decoder layers as non-owned references. Assigning an nn.ModuleList
        # to a child nn.Module would register the frozen base model as part of the
        # controller and pollute parameters()/state_dict().
        object.__setattr__(self, "_layers_ref", layers_ref)
        layer_indices = torch.as_tensor(layer_indices, dtype=torch.long).detach().clone().reshape(-1)
        channel_indices = torch.as_tensor(channel_indices, dtype=torch.long).detach().clone().reshape(-1)
        if layer_indices.numel() == 0:
            raise ValueError("controller must contain at least one gate")
        if layer_indices.numel() != channel_indices.numel():
            raise ValueError("layer_indices and channel_indices must have the same length")
        if bool((layer_indices < 0).any()) or bool((layer_indices >= len(layers_ref)).any()):
            raise ValueError("layer_indices contain values outside the decoder layer range")
        if bool((channel_indices < 0).any()) or bool((channel_indices >= self.hidden_size).any()):
            raise ValueError("channel_indices contain values outside [0, hidden_size)")
        keys = set()
        for layer, channel in zip(layer_indices.tolist(), channel_indices.tolist(), strict=True):
            key = (int(layer), int(channel))
            if key in keys:
                raise ValueError(f"duplicate gate key {key}; duplicate gates must be coalesced")
            keys.add(key)
        self.register_buffer("layer_indices", layer_indices.long())
        self.register_buffer("channel_indices", channel_indices.long())
        if raw_init is None:
            raw_init = torch.zeros(int(layer_indices.numel()), dtype=torch.float32)
        raw_init = torch.as_tensor(raw_init, dtype=torch.float32).detach().clone().reshape(-1)
        if raw_init.numel() != layer_indices.numel():
            raise ValueError("raw_init must have the same length as layer_indices")
        if not bool(torch.isfinite(raw_init).all()):
            raise ValueError("raw_init contains NaN or Inf")
        self.raw = nn.Parameter(raw_init.float())
        self._signed_log_override: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._rebuild_index()

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self._layers_ref

    @property
    def s(self) -> torch.Tensor:
        override = self._signed_log_override
        if override is not None:
            return override
        return self.max_log_gate * torch.tanh(self.raw)

    @property
    def is_attached(self) -> bool:
        return bool(self._handles)

    @staticmethod
    def raw_from_signed_log_values(values: torch.Tensor, max_log_gate: float) -> torch.Tensor:
        mg = float(max_log_gate)
        if mg <= 0:
            raise ValueError("max_log_gate must be positive")
        x = torch.clamp(values.float() / mg, -0.999999, 0.999999)
        return 0.5 * torch.log((1.0 + x) / (1.0 - x))

    @torch.no_grad()
    def set_signed_log_values_(self, values: torch.Tensor) -> None:
        vals = values.detach().to(device=self.raw.device, dtype=torch.float32).reshape_as(self.raw)
        if not bool(torch.isfinite(vals).all()):
            raise ValueError("signed log-gate values contain NaN or Inf")
        vals = torch.clamp(vals, -self.max_log_gate, self.max_log_gate)
        self.raw.copy_(self.raw_from_signed_log_values(vals, self.max_log_gate).to(self.raw.device))

    @torch.no_grad()
    def add_signed_log_values_(self, delta: torch.Tensor, *, scale: float = 1.0) -> None:
        self.set_signed_log_values_(self.s.detach().float() + float(scale) * delta.detach().float().to(self.raw.device))

    @contextlib.contextmanager
    def temporary_signed_log_values(self, values: torch.Tensor):
        if values.numel() != self.raw.numel():
            raise ValueError("temporary signed log values must have one entry per gate")
        if not bool(torch.isfinite(values.detach()).all()):
            raise ValueError("temporary signed log values contain NaN or Inf")
        old = self._signed_log_override
        vals = values.to(device=self.raw.device, dtype=torch.float32).reshape_as(self.raw)
        self._signed_log_override = vals
        try:
            yield
        finally:
            self._signed_log_override = old

    def _rebuild_index(self) -> None:
        self._by_layer.clear()
        for layer_id in sorted(set(int(x) for x in self.layer_indices.cpu().tolist())):
            pos = torch.nonzero(self.layer_indices == layer_id, as_tuple=False).flatten()
            ch = self.channel_indices[pos]
            self._by_layer[layer_id] = (pos, ch)

    def attach(self) -> None:
        if self._handles:
            return
        for layer_id in self._by_layer:
            if self.hook_site == "layer_input":
                self._handles.append(self.layers[layer_id].register_forward_pre_hook(self._make_pre_hook(layer_id)))
            else:
                self._handles.append(self.layers[layer_id].register_forward_hook(self._make_hook(layer_id)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextlib.contextmanager
    def attached(self):
        """Attach hooks for the duration of a scope without leaking ownership.

        If the controller was already attached by the caller, this context leaves
        it attached on exit. This fixes a subtle failure mode where helper
        methods such as validation/evaluation could detach a controller owned by
        an outer training or serving scope.
        """

        already_attached = self.is_attached
        if not already_attached:
            self.attach()
        try:
            yield self
        finally:
            if not already_attached:
                self.remove()

    @contextlib.contextmanager
    def detached(self):
        """Temporarily remove hooks, restoring the prior attachment state."""

        was_attached = self.is_attached
        if was_attached:
            self.remove()
        try:
            yield self
        finally:
            if was_attached:
                self.attach()

    def _scale_hidden(self, h: torch.Tensor, layer_id: int) -> torch.Tensor:
        if h.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hooked hidden state last dimension {h.shape[-1]} does not match hidden_size={self.hidden_size}"
            )
        pos, ch = self._by_layer[layer_id]
        pos = pos.to(device=h.device)
        ch = ch.to(device=h.device)
        scale = torch.ones(self.hidden_size, dtype=h.dtype, device=h.device)
        scale[ch] = torch.exp(self.s.to(device=h.device, dtype=h.dtype)[pos])
        shape = [1] * (h.ndim - 1) + [self.hidden_size]
        return h * scale.view(*shape)

    def _make_pre_hook(self, layer_id: int):
        def hook(_module, inputs):
            if not inputs:
                return inputs
            h = inputs[0]
            if not torch.is_tensor(h):
                raise TypeError("layer_input hook expected first positional input to be a tensor")
            return (self._scale_hidden(h, layer_id),) + tuple(inputs[1:])
        return hook

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            h = self._scale_hidden(h, layer_id)
            if rest is None:
                return h
            return (h,) + rest
        return hook

    def state(
        self,
        *,
        layer_path: str,
        n_layers: int,
        model_name: str | None,
        model_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        metadata: dict | None = None,
    ) -> SignedLogMaskState:
        return SignedLogMaskState(
            layer_path=layer_path,
            n_layers=n_layers,
            hidden_size=self.hidden_size,
            layer_indices=self.layer_indices.detach().cpu(),
            channel_indices=self.channel_indices.detach().cpu(),
            raw=self.raw.detach().cpu(),
            max_log_gate=self.max_log_gate,
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            created_at=time.time(),
            hook_site=self.hook_site,
            theory_version="activation_control",
            metadata=dict(metadata or {}),
        )


class _ActivationScoreCapture:
    """Retain decoder-layer input/output activations and their gradients for gate scoring."""

    def __init__(self, layers: Sequence[nn.Module], layer_ids: Sequence[int], *, hook_site: str) -> None:
        self.captured: dict[int, torch.Tensor] = {}
        self.hook_site = str(hook_site)
        if self.hook_site == "layer_input":
            self._handles = [layers[i].register_forward_pre_hook(self._make_pre_hook(i)) for i in layer_ids]
        elif self.hook_site == "layer_output":
            self._handles = [layers[i].register_forward_hook(self._make_hook(i)) for i in layer_ids]
        else:
            raise ValueError("hook_site must be 'layer_input' or 'layer_output'")

    def _retain(self, idx: int, h: torch.Tensor) -> None:
        h.retain_grad()
        self.captured[idx] = h

    def _make_pre_hook(self, idx: int):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                self._retain(idx, inputs[0])
            return inputs
        return hook

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self._retain(idx, h)
            return output
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _freeze(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


@contextlib.contextmanager
def _controller_temporarily_removed(controller: _SignedLogMaskModule | None):
    """Temporarily remove hooks without disturbing an outer unattached state."""
    was_attached = bool(controller is not None and controller.is_attached)
    if was_attached:
        controller.remove()
    try:
        yield
    finally:
        if was_attached and controller is not None:
            controller.attach()


def _kl_to_base_logits(controller_logits: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean KL(base || controller) on supervised next-token positions."""
    ctrl = controller_logits[:, :-1, :].contiguous().float()
    base = base_logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    if not bool(mask.any()):
        raise ValueError("retain batch contains no supervised next-token labels")
    base_logp = torch.log_softmax(base[mask], dim=-1)
    ctrl_logp = torch.log_softmax(ctrl[mask], dim=-1)
    base_p = base_logp.exp()
    return (base_p * (base_logp - ctrl_logp)).sum(dim=-1).mean()


def _attr_first(obj, names: Sequence[str]) -> str | None:
    for name in names:
        val = getattr(obj, name, None)
        if val is not None and str(val):
            return str(val)
    return None



class ForwardFineTuner:
    """LoRA-free forward-pass fine-tuner for Hugging Face causal LMs.

    The model weights stay frozen. We learn a sparse set of shared signed
    log-gates on decoder-layer output channels:

        h[layer, :, channel] <- exp(s[layer, channel]) * h[layer, :, channel]

    The same gates are used for support examples and query/inference prompts.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        gates: int = 5000,
        layers: str = "all",
        max_log_gate: float = 0.05,
        hook_site: str = "layer_output",
        model_name: str | None = None,
        model_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gates = int(gates)
        self.max_log_gate = float(max_log_gate)
        self.hook_site = str(hook_site)
        if self.gates <= 0:
            raise ValueError("gates must be positive")
        if not math.isfinite(self.max_log_gate) or self.max_log_gate <= 0.0:
            raise ValueError("max_log_gate must be finite and positive")
        if self.hook_site not in _SignedLogMaskModule.VALID_HOOK_SITES:
            raise ValueError(f"hook_site must be one of {sorted(_SignedLogMaskModule.VALID_HOOK_SITES)}")
        cfg = getattr(model, "config", None)
        self.model_name = model_name or _attr_first(cfg, ["_name_or_path", "name_or_path"])
        self.model_revision = model_revision or _attr_first(cfg, ["_commit_hash", "_ntkmirror_requested_revision"])
        self.tokenizer_name = tokenizer_name or _attr_first(tokenizer, ["name_or_path", "_name_or_path"])
        self.tokenizer_revision = tokenizer_revision or _attr_first(tokenizer, ["_commit_hash", "_ntkmirror_requested_revision"])
        self.layer_path, self.decoder_layers = find_decoder_layers(model)
        self.hidden_size = infer_hidden_size(model)
        self.layer_ids = parse_layers(layers, len(self.decoder_layers))
        self.controller: _SignedLogMaskModule | None = None
        _freeze(model)

    @property
    def has_controller(self) -> bool:
        return self.controller is not None

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            use_cache=False,
        )
        return causal_loss_from_logits(out.logits, batch["labels"])

    @torch.no_grad()
    def evaluate_nll(
        self,
        examples: Sequence[Example],
        *,
        batch_size: int = 8,
        max_length: int = 1024,
        use_controller: bool = True,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        if self.controller is None:
            context = contextlib.nullcontext()
        elif use_controller:
            context = self.controller.attached()
        else:
            context = self.controller.detached()
        with context:
            for chunk in batches(examples, batch_size):
                batch = make_batch(self.tokenizer, chunk, device=_device_of(self.model), max_length=max_length)
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    use_cache=False,
                )
                loss = causal_loss_from_logits(out.logits, batch["labels"])
                c, n = token_accuracy_from_logits(out.logits, batch["labels"])
                total_loss += float(loss.item()) * n
                total_tokens += n
                correct += c
        return {
            "nll": total_loss / max(1, total_tokens),
            "token_acc": correct / max(1, total_tokens),
            "tokens": float(total_tokens),
        }

    def _score_gates(
        self,
        examples: Sequence[Example],
        *,
        score_batches: int = 16,
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose gates by |dL/ds| at s=0, where dL/ds = <dL/dh, h>."""
        device = _device_of(self.model)
        scores = torch.zeros((len(self.decoder_layers), self.hidden_size), dtype=torch.float32, device="cpu")
        self.model.train(False)
        capture = _ActivationScoreCapture(self.decoder_layers, self.layer_ids, hook_site=self.hook_site)

        # Hidden states need a gradient source. Enabling the input embedding is
        # the least invasive option and is restored immediately afterwards.
        emb = self.model.get_input_embeddings()
        old_emb_req = emb.weight.requires_grad
        emb.weight.requires_grad_(True)
        try:
            seen = 0
            for chunk in batches(examples, batch_size):
                if seen >= score_batches:
                    break
                self.model.zero_grad(set_to_none=True)
                batch = make_batch(self.tokenizer, chunk, device=device, max_length=max_length)
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    use_cache=False,
                )
                loss = causal_loss_from_logits(out.logits, batch["labels"])
                loss.backward()
                for layer_id, h in capture.captured.items():
                    if h.grad is None:
                        continue
                    # Derivative wrt log gate s at s=0 is grad_h * h.
                    layer_score = (h.grad.detach().float() * h.detach().float()).abs().sum(dim=(0, 1))
                    scores[layer_id] += layer_score.cpu()
                seen += 1
            if seen == 0:
                raise ValueError("no score batches were processed")
        finally:
            capture.remove()
            emb.weight.requires_grad_(old_emb_req)
            self.model.zero_grad(set_to_none=True)
            _freeze(self.model)

        candidate_scores = scores[self.layer_ids]
        flat = candidate_scores.flatten()
        k = min(max(1, self.gates), int(flat.numel()))
        top = torch.topk(flat, k=k, largest=True).indices
        local_layer = top // self.hidden_size
        channel = top % self.hidden_size
        layer_indices = torch.tensor([self.layer_ids[int(i)] for i in local_layer], dtype=torch.long)
        channel_indices = channel.to(torch.long).cpu()
        return layer_indices, channel_indices

    def initialize_controller(
        self,
        examples: Sequence[Example],
        *,
        score_batches: int = 16,
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> dict[str, float]:
        """Score and initialise the sparse activation-control gate basis."""
        _freeze(self.model)
        device = _device_of(self.model)
        t0 = time.perf_counter()
        layer_indices, channel_indices = self._score_gates(
            examples, score_batches=score_batches, batch_size=batch_size, max_length=max_length
        )
        if self.controller is not None:
            self.controller.remove()
        self.controller = _SignedLogMaskModule(
            self.decoder_layers,
            layer_indices.to(device),
            channel_indices.to(device),
            hidden_size=self.hidden_size,
            max_log_gate=self.max_log_gate,
            hook_site=self.hook_site,
        ).to(device)
        return {
            "selected_gates": float(self.controller.raw.numel()),
            "select_seconds": float(time.perf_counter() - t0),
            "hook_site": self.hook_site,
        }

    def fit(
        self,
        examples: Sequence[Example],
        *,
        steps: int = 240,
        lr: float = 5e-3,
        batch_size: int = 8,
        score_batches: int = 16,
        max_length: int = 1024,
        weight_decay: float = 0.0,
        l2: float = 1e-5,
        validation_examples: Sequence[Example] | None = None,
        validation_interval: int | None = None,
        early_stop_patience: int = 0,
        select_best_on_validation: bool = True,
        retain_examples: Sequence[Example] | None = None,
        retain_weight: float = 0.0,
        kl_to_base: float = 0.0,
        verbose: bool = True,
    ) -> dict[str, float | list[dict[str, float]]]:
        if not examples:
            raise ValueError("fit requires at least one example")
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if validation_interval is not None and int(validation_interval) <= 0:
            raise ValueError("validation_interval must be positive when supplied")
        if early_stop_patience < 0:
            raise ValueError("early_stop_patience must be non-negative")
        if retain_weight < 0.0:
            raise ValueError("retain_weight must be non-negative")
        if kl_to_base < 0.0:
            raise ValueError("kl_to_base must be non-negative")
        if (retain_weight > 0.0 or kl_to_base > 0.0) and not retain_examples:
            raise ValueError("retain_examples are required when retain_weight or kl_to_base is non-zero")
        _freeze(self.model)
        device = _device_of(self.model)

        init_stats = self.initialize_controller(
            examples, score_batches=score_batches, batch_size=batch_size, max_length=max_length
        )
        select_seconds = float(init_stats["select_seconds"])
        if self.controller is None:
            raise RuntimeError("controller initialisation failed")

        opt = torch.optim.AdamW([self.controller.raw], lr=lr, weight_decay=weight_decay)
        train_batches = list(batches(examples, batch_size))
        if not train_batches:
            raise ValueError("no training batches")
        retain_batches = list(batches(retain_examples or [], batch_size))
        self.model.train(False)
        self.controller.attach()
        losses: list[float] = []
        task_losses: list[float] = []
        retain_losses: list[float] = []
        kl_losses: list[float] = []
        validation_rows: list[dict[str, float]] = []
        best_raw: torch.Tensor | None = None
        best_validation_nll = math.inf
        best_step = 0
        stale_validations = 0
        stopped_early = False
        completed_steps = 0
        if validation_examples:
            interval = int(validation_interval or max(1, min(50, steps // 10 or 1)))
        else:
            interval = 0
        train_t0 = time.perf_counter()
        try:
            for step in range(steps):
                chunk = train_batches[step % len(train_batches)]
                batch = make_batch(self.tokenizer, chunk, device=device, max_length=max_length)
                opt.zero_grad(set_to_none=True)
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    use_cache=False,
                )
                task_loss = causal_loss_from_logits(out.logits, batch["labels"])
                loss = task_loss
                retain_loss = None
                kl_loss = None
                if retain_batches:
                    retain_chunk = retain_batches[step % len(retain_batches)]
                    retain_batch = make_batch(self.tokenizer, retain_chunk, device=device, max_length=max_length)
                    retain_out = self.model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch.get("attention_mask"),
                        use_cache=False,
                    )
                    if retain_weight > 0.0:
                        retain_loss = causal_loss_from_logits(retain_out.logits, retain_batch["labels"])
                        loss = loss + float(retain_weight) * retain_loss
                    if kl_to_base > 0.0:
                        with _controller_temporarily_removed(self.controller):
                            with torch.no_grad():
                                base_logits = self.model(
                                    input_ids=retain_batch["input_ids"],
                                    attention_mask=retain_batch.get("attention_mask"),
                                    use_cache=False,
                                ).logits.detach()
                        kl_loss = _kl_to_base_logits(retain_out.logits, base_logits, retain_batch["labels"])
                        loss = loss + float(kl_to_base) * kl_loss
                if l2 > 0:
                    loss = loss + float(l2) * self.controller.s.float().pow(2).mean()
                loss.backward()
                opt.step()
                completed_steps = step + 1
                losses.append(float(loss.detach().item()))
                task_losses.append(float(task_loss.detach().item()))
                if retain_loss is not None:
                    retain_losses.append(float(retain_loss.detach().item()))
                if kl_loss is not None:
                    kl_losses.append(float(kl_loss.detach().item()))
                if verbose and (step == 0 or (step + 1) % max(1, steps // 5) == 0):
                    msg = f"step {step+1:4d}/{steps}: loss={losses[-1]:.6f} task={task_losses[-1]:.6f}"
                    if retain_loss is not None:
                        msg += f" retain={retain_losses[-1]:.6f}"
                    if kl_loss is not None:
                        msg += f" kl={kl_losses[-1]:.6f}"
                    print(msg, flush=True)
                if validation_examples and ((step + 1) % interval == 0 or (step + 1) == steps):
                    val = self.evaluate_nll(
                        validation_examples,
                        batch_size=batch_size,
                        max_length=max_length,
                        use_controller=True,
                    )
                    val_nll = float(val["nll"])
                    validation_rows.append({
                        "step": float(step + 1),
                        "nll": val_nll,
                        "token_acc": float(val["token_acc"]),
                        "tokens": float(val["tokens"]),
                    })
                    improved = val_nll < best_validation_nll - 1e-12
                    if improved:
                        best_validation_nll = val_nll
                        best_step = step + 1
                        best_raw = self.controller.raw.detach().clone()
                        stale_validations = 0
                    else:
                        stale_validations += 1
                    if verbose:
                        print(f"validation step {step+1:4d}: nll={val_nll:.6f} token_acc={float(val['token_acc']):.4f}", flush=True)
                    if early_stop_patience and stale_validations >= int(early_stop_patience):
                        stopped_early = True
                        break
        finally:
            self.controller.remove()
        if validation_examples and select_best_on_validation and best_raw is not None:
            with torch.no_grad():
                self.controller.raw.copy_(best_raw.to(device=self.controller.raw.device, dtype=self.controller.raw.dtype))
        train_seconds = time.perf_counter() - train_t0
        out: dict[str, float | list[dict[str, float]]] = {
            "selected_gates": float(self.controller.raw.numel()),
            "select_seconds": float(select_seconds),
            "train_seconds": float(train_seconds),
            "steps_completed": float(completed_steps),
            "loss_first": float(losses[0]) if losses else math.nan,
            "loss_last": float(losses[-1]) if losses else math.nan,
            "task_loss_first": float(task_losses[0]) if task_losses else math.nan,
            "task_loss_last": float(task_losses[-1]) if task_losses else math.nan,
            "hook_site": self.hook_site,
            "stopped_early": float(1.0 if stopped_early else 0.0),
            "retain_weight": float(retain_weight),
            "kl_to_base": float(kl_to_base),
        }
        if retain_losses:
            out["retain_loss_last"] = float(retain_losses[-1])
        if kl_losses:
            out["kl_loss_last"] = float(kl_losses[-1])
        if validation_examples:
            out["validation_best_nll"] = float(best_validation_nll)
            out["validation_best_step"] = float(best_step)
            out["validation"] = validation_rows
        return out

    def secant_diagnostics(
        self,
        examples: Sequence[Example],
        *,
        alphas: Sequence[float] = (0.125, 0.25, 0.5, 0.75, 1.0),
        drift_alphas: Sequence[float] = (0.0625, 0.1875, 0.375, 0.625, 0.875),
        eps: float = 1e-3,
        batch_size: int = 2,
        max_length: int = 1024,
        projection: str = "target",
        top_k: int = 32,
    ) -> dict:
        if self.controller is None:
            raise ValueError("load, fit, or initialize a controller before diagnostics")
        from .dual import controller_secant_diagnostics
        return controller_secant_diagnostics(
            self.model, self.tokenizer, self.controller, examples,
            alphas=alphas, drift_alphas=drift_alphas, eps=eps,
            batch_size=batch_size, max_length=max_length, projection=projection, top_k=top_k,
        ).to_dict()

    def dual_projection_diagnostics(
        self,
        support_examples: Sequence[Example],
        calibration_examples: Sequence[Example] | None = None,
        *,
        batch_size: int = 1,
        max_length: int = 1024,
        projection: str = "target",
        top_k: int = 32,
        target_step_size: float = 1e-5,
        ridge: float = 1e-4,
        cg_iters: int = 16,
        cg_tol: float = 1e-5,
        fd_eps: float = 1e-3,
        metric: str = "identity",
        metric_eps: float = 1e-6,
        jvp_mode: str = "exact",
        param_name_substrings: Sequence[str] | None = None,
        max_target_parameters: int = 0,
    ) -> dict:
        if self.controller is None:
            raise ValueError("load, fit, or initialize a controller before diagnostics")
        from .dual import dual_projection_diagnostics
        solution, _target = dual_projection_diagnostics(
            self.model, self.tokenizer, self.controller, support_examples, calibration_examples,
            batch_size=batch_size, max_length=max_length, projection=projection, top_k=top_k,
            target_step_size=target_step_size, ridge=ridge, cg_iters=cg_iters, cg_tol=cg_tol,
            fd_eps=fd_eps, metric=metric, metric_eps=metric_eps, jvp_mode=jvp_mode,
            param_name_substrings=param_name_substrings, max_target_parameters=max_target_parameters,
        )
        return solution.diagnostics.to_dict()

    def fit_dual(
        self,
        examples: Sequence[Example],
        *,
        steps: int = 8,
        target_step_size: float = 1e-5,
        apply_scale: float = 1.0,
        batch_size: int = 1,
        score_batches: int = 16,
        max_length: int = 1024,
        projection: str = "target",
        top_k: int = 32,
        ridge: float = 1e-4,
        cg_iters: int = 16,
        cg_tol: float = 1e-5,
        fd_eps: float = 1e-3,
        metric: str = "identity",
        metric_eps: float = 1e-6,
        jvp_mode: str = "exact",
        param_name_substrings: Sequence[str] | None = None,
        max_target_parameters: int = 0,
        verbose: bool = True,
    ) -> dict:
        if not examples:
            raise ValueError("fit_dual requires at least one example")
        if self.controller is None:
            init_stats = self.initialize_controller(
                examples, score_batches=score_batches, batch_size=batch_size, max_length=max_length
            )
        else:
            init_stats = {"selected_gates": float(self.controller.raw.numel()), "select_seconds": 0.0, "hook_site": self.hook_site}
        if self.controller is None:
            raise RuntimeError("controller initialisation failed")
        from .dual import dual_projection_diagnostics
        train_batches = list(batches(examples, batch_size))
        rows: list[dict] = []
        start = time.perf_counter()
        for step in range(int(steps)):
            chunk = train_batches[step % len(train_batches)]
            solution, _target = dual_projection_diagnostics(
                self.model, self.tokenizer, self.controller, chunk, chunk,
                batch_size=batch_size, max_length=max_length, projection=projection, top_k=top_k,
                target_step_size=target_step_size, ridge=ridge, cg_iters=cg_iters, cg_tol=cg_tol,
                fd_eps=fd_eps, metric=metric, metric_eps=metric_eps, jvp_mode=jvp_mode,
                param_name_substrings=param_name_substrings, max_target_parameters=max_target_parameters,
            )
            update = solution.update.to(device=_device_of(self.model), dtype=torch.float32)
            before_s = self.controller.s.detach().float().clone()
            intended = float(apply_scale) * update.detach().float().to(before_s.device)
            self.controller.add_signed_log_values_(update, scale=apply_scale)
            after_s = self.controller.s.detach().float().clone()
            applied = after_s - before_s
            clip_delta = intended - applied
            row = solution.diagnostics.to_dict()
            row["step"] = float(step + 1)
            row["apply_scale"] = float(apply_scale)
            row["applied_update_norm"] = float(torch.linalg.vector_norm(applied).item())
            row["clip_update_residual"] = float(torch.linalg.vector_norm(clip_delta).item() / max(float(torch.linalg.vector_norm(intended).item()), 1e-30))
            row["clip_fraction"] = float((clip_delta.abs() > 1e-7).float().mean().item()) if clip_delta.numel() else 0.0
            rows.append(row)
            if verbose:
                print(
                    f"dual step {step+1:4d}/{steps}: "
                    f"realized={row['realized_residual']:.4f} "
                    f"clipped={row.get('clipped_realized_residual', row['realized_residual']):.4f} "
                    f"range={row['range_residual']:.4f} "
                    f"cos={row['realized_cosine']:.4f} adj={row['adjoint_error']:.1e} "
                    f"sym={row['symmetry_error']:.1e} |u|={row['update_norm']:.4g} "
                    f"clip={row['clip_fraction']:.2%}",
                    flush=True,
                )
        return {
            "mode": "pathwise_activation_control_ntk",
            "init": init_stats,
            "steps": rows,
            "train_seconds": float(time.perf_counter() - start),
            "selected_gates": float(self.controller.raw.numel()),
        }

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        if self.controller is None:
            raise ValueError("no controller has been fitted")
        cfg = getattr(self.model, "config", None)
        model_name = self.model_name or _attr_first(cfg, ["_name_or_path", "name_or_path"])
        model_revision = self.model_revision or _attr_first(cfg, ["_commit_hash", "_ntkmirror_requested_revision"])
        tokenizer_name = self.tokenizer_name or _attr_first(self.tokenizer, ["name_or_path", "_name_or_path"])
        tokenizer_revision = self.tokenizer_revision or _attr_first(self.tokenizer, ["_commit_hash", "_ntkmirror_requested_revision"])
        self.controller.state(
            layer_path=self.layer_path,
            n_layers=len(self.decoder_layers),
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            metadata={
                "saved_at_unix": time.time(),
                "controller_format": "torch_weights_only_dict_v1",
                **dict(metadata or {}),
            },
        ).save(path)

    def _check_identity(
        self,
        *,
        label: str,
        saved: str | None,
        current: str | None,
        allow_mismatch: bool,
    ) -> None:
        if allow_mismatch or saved is None or current is None:
            return
        if str(saved) != str(current):
            raise ValueError(
                f"controller {label} {saved!r} does not match current {label} {current!r}; "
                "pass allow_model_mismatch=True only after manually verifying compatibility"
            )

    def load(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        allow_model_mismatch: bool = False,
    ) -> "ForwardFineTuner":
        if map_location is None:
            map_location = _device_of(self.model)
        state = SignedLogMaskState.load(path, map_location="cpu")
        if state.layer_path != self.layer_path:
            raise ValueError(
                f"controller layer_path {state.layer_path!r} does not match model layer_path {self.layer_path!r}"
            )
        if state.hidden_size != self.hidden_size:
            raise ValueError(
                f"controller hidden size {state.hidden_size} does not match model hidden size {self.hidden_size}"
            )
        if state.n_layers != len(self.decoder_layers):
            raise ValueError(
                f"controller layer count {state.n_layers} does not match model layer count {len(self.decoder_layers)}"
            )
        self._check_identity(
            label="model_name",
            saved=state.model_name,
            current=self.model_name,
            allow_mismatch=allow_model_mismatch,
        )
        self._check_identity(
            label="model_revision",
            saved=state.model_revision,
            current=self.model_revision,
            allow_mismatch=allow_model_mismatch,
        )
        self._check_identity(
            label="tokenizer_name",
            saved=state.tokenizer_name,
            current=self.tokenizer_name,
            allow_mismatch=allow_model_mismatch,
        )
        self._check_identity(
            label="tokenizer_revision",
            saved=state.tokenizer_revision,
            current=self.tokenizer_revision,
            allow_mismatch=allow_model_mismatch,
        )
        device = _device_of(self.model)
        if self.controller is not None:
            self.controller.remove()
        self.controller = _SignedLogMaskModule(
            self.decoder_layers,
            state.layer_indices.to(device),
            state.channel_indices.to(device),
            hidden_size=self.hidden_size,
            max_log_gate=state.max_log_gate,
            hook_site=state.hook_site,
            raw_init=state.raw.to(device),
        ).to(device)
        self.hook_site = state.hook_site
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        return_full_text: bool = True,
        **generate_kwargs,
    ) -> str:
        if self.controller is None:
            raise ValueError("load or fit a controller before generate")
        device = _device_of(self.model)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        self.model.eval()
        with self.controller.attached():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                **generate_kwargs,
            )
        ids = output_ids[0] if return_full_text else output_ids[0, prompt_len:]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def selected_gates_json(self) -> list[dict[str, int]]:
        if self.controller is None:
            return []
        return [
            {"layer": int(layer_idx), "channel": int(channel_idx)}
            for layer_idx, channel_idx in zip(
                self.controller.layer_indices.detach().cpu().tolist(),
                self.controller.channel_indices.detach().cpu().tolist(),
                strict=True,
            )
        ]

    def write_manifest(self, path: str | Path) -> None:
        obj = {
            "layer_path": self.layer_path,
            "n_layers": len(self.decoder_layers),
            "hidden_size": self.hidden_size,
            "gates": self.selected_gates_json(),
            "max_log_gate": self.max_log_gate,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "hook_site": self.hook_site,
            "theory_version": "activation_control",
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
