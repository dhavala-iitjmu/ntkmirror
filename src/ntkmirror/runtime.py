from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .compose import compose_states
from .controller import SignedLogMaskState, _SignedLogMaskModule
from .data import Example, batches, make_batch
from .layers import find_decoder_layers, infer_hidden_size
from .losses import causal_loss_from_logits, token_accuracy_from_logits


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _attr_first(obj, names: Sequence[str]) -> str | None:
    for name in names:
        val = getattr(obj, name, None)
        if val is not None and str(val):
            return str(val)
    return None


@dataclass(frozen=True)
class RuntimePolicy:
    """Admission policy for request-time controller application."""

    require_model_identity: bool = True
    allow_unknown_identity: bool = True
    max_controllers: int = 4
    max_gates: int | None = None
    serialize_requests: bool = True


class ControllerRuntime:
    """Serving-oriented wrapper for per-request controller application.

    PyTorch forward hooks are model-global. This runtime therefore serialises
    controller application by default and removes hooks in a ``finally`` block.
    Heterogeneous per-row controllers are intentionally not silently supported.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        model_name: str | None = None,
        model_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        policy: RuntimePolicy | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy or RuntimePolicy()
        self.layer_path, self.decoder_layers = find_decoder_layers(model)
        self.hidden_size = infer_hidden_size(model)
        cfg = getattr(model, "config", None)
        self.model_name = model_name or _attr_first(cfg, ["_name_or_path", "name_or_path"])
        self.model_revision = model_revision or _attr_first(cfg, ["_commit_hash", "_ntkmirror_requested_revision"])
        self.tokenizer_name = tokenizer_name or (_attr_first(tokenizer, ["name_or_path", "_name_or_path"]) if tokenizer is not None else None)
        self.tokenizer_revision = tokenizer_revision or (_attr_first(tokenizer, ["_commit_hash", "_ntkmirror_requested_revision"]) if tokenizer is not None else None)
        self._lock = threading.RLock()

    def _check_identity(self, label: str, saved: str | None, current: str | None) -> None:
        if not self.policy.require_model_identity:
            return
        if saved is None or current is None:
            if self.policy.allow_unknown_identity:
                return
            raise ValueError(f"controller {label} or runtime {label} is unknown; policy requires exact identity")
        if str(saved) != str(current):
            raise ValueError(f"controller {label} {saved!r} does not match runtime {label} {current!r}")

    def validate_state(self, state: SignedLogMaskState) -> None:
        state.validate(max_gates=self.policy.max_gates)
        if state.layer_path != self.layer_path:
            raise ValueError(f"controller layer_path {state.layer_path!r} does not match runtime {self.layer_path!r}")
        if state.n_layers != len(self.decoder_layers):
            raise ValueError(f"controller has {state.n_layers} layers, runtime has {len(self.decoder_layers)}")
        if state.hidden_size != self.hidden_size:
            raise ValueError(f"controller hidden_size {state.hidden_size} does not match runtime {self.hidden_size}")
        self._check_identity("model_name", state.model_name, self.model_name)
        self._check_identity("model_revision", state.model_revision, self.model_revision)
        self._check_identity("tokenizer_name", state.tokenizer_name, self.tokenizer_name)
        self._check_identity("tokenizer_revision", state.tokenizer_revision, self.tokenizer_revision)

    def load_state(self, controller: str | Path | SignedLogMaskState) -> SignedLogMaskState:
        state = controller if isinstance(controller, SignedLogMaskState) else SignedLogMaskState.load(controller, map_location="cpu")
        self.validate_state(state)
        return state

    def compose(self, controllers: Sequence[str | Path | SignedLogMaskState], *, weights: Sequence[float] | None = None, max_log_gate: float | None = None) -> SignedLogMaskState:
        if not controllers:
            raise ValueError("at least one controller is required")
        if len(controllers) > int(self.policy.max_controllers):
            raise ValueError(f"policy permits at most {self.policy.max_controllers} controllers per request")
        states = [self.load_state(c) for c in controllers]
        return compose_states(states, weights=weights, max_log_gate=max_log_gate)

    def _module_from_state(self, state: SignedLogMaskState) -> _SignedLogMaskModule:
        device = _device_of(self.model)
        return _SignedLogMaskModule(
            self.decoder_layers,
            state.layer_indices.to(device),
            state.channel_indices.to(device),
            hidden_size=self.hidden_size,
            max_log_gate=state.max_log_gate,
            hook_site=state.hook_site,
            raw_init=state.raw.to(device),
        ).to(device)

    @contextlib.contextmanager
    def apply(self, controller: str | Path | SignedLogMaskState | Sequence[str | Path | SignedLogMaskState], *, weights: Sequence[float] | None = None, max_log_gate: float | None = None) -> Iterator[_SignedLogMaskModule]:
        """Attach one request-scoped controller and always remove its hooks."""

        if isinstance(controller, (str, Path, SignedLogMaskState)):
            state = self.load_state(controller)
        else:
            state = self.compose(list(controller), weights=weights, max_log_gate=max_log_gate)
        lock = self._lock if self.policy.serialize_requests else contextlib.nullcontext()
        with lock:
            module = self._module_from_state(state)
            module.attach()
            try:
                yield module
            finally:
                module.remove()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        controller: str | Path | SignedLogMaskState | Sequence[str | Path | SignedLogMaskState],
        *,
        max_new_tokens: int = 128,
        return_full_text: bool = True,
        weights: Sequence[float] | None = None,
        max_log_gate: float | None = None,
        **generate_kwargs,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("ControllerRuntime.generate requires a tokenizer")
        device = _device_of(self.model)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        self.model.eval()
        with self.apply(controller, weights=weights, max_log_gate=max_log_gate):
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                **generate_kwargs,
            )
        ids = output_ids[0] if return_full_text else output_ids[0, prompt_len:]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def evaluate_nll(
        self,
        examples: Sequence[Example],
        controller: str | Path | SignedLogMaskState | Sequence[str | Path | SignedLogMaskState],
        *,
        batch_size: int = 8,
        max_length: int = 1024,
        weights: Sequence[float] | None = None,
        max_log_gate: float | None = None,
    ) -> dict[str, float]:
        if self.tokenizer is None:
            raise ValueError("ControllerRuntime.evaluate_nll requires a tokenizer")
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        device = _device_of(self.model)
        self.model.eval()
        with self.apply(controller, weights=weights, max_log_gate=max_log_gate):
            for chunk in batches(examples, batch_size):
                batch = make_batch(self.tokenizer, chunk, device=device, max_length=max_length)
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), use_cache=False)
                loss = causal_loss_from_logits(out.logits, batch["labels"])
                c, n = token_accuracy_from_logits(out.logits, batch["labels"])
                total_loss += float(loss.item()) * n
                total_tokens += n
                correct += c
        return {"nll": total_loss / max(1, total_tokens), "token_acc": correct / max(1, total_tokens), "tokens": float(total_tokens)}

    def generate_batch(
        self,
        prompts: Sequence[str],
        controllers: Sequence[str | Path | SignedLogMaskState] | str | Path | SignedLogMaskState,
        **kwargs,
    ) -> list[str]:
        """Generate for a batch only when all rows share one controller."""

        if isinstance(controllers, (str, Path, SignedLogMaskState)):
            controller = controllers
        else:
            seq = list(controllers)
            if len(seq) != 1:
                raise NotImplementedError("heterogeneous per-row controllers are not supported; split the batch by controller")
            controller = seq[0]
        return [self.generate(prompt, controller, **kwargs) for prompt in prompts]
