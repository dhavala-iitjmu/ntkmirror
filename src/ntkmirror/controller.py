from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

    @property
    def n_gates(self) -> int:
        return int(self.layer_indices.numel())

    def to_dict(self) -> dict:
        return {
            "layer_path": self.layer_path,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "layer_indices": self.layer_indices.detach().cpu(),
            "channel_indices": self.channel_indices.detach().cpu(),
            "raw": self.raw.detach().cpu(),
            "max_log_gate": float(self.max_log_gate),
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "SignedLogMaskState":
        return cls(
            layer_path=str(obj["layer_path"]),
            n_layers=int(obj["n_layers"]),
            hidden_size=int(obj["hidden_size"]),
            layer_indices=torch.as_tensor(obj["layer_indices"], dtype=torch.long),
            channel_indices=torch.as_tensor(obj["channel_indices"], dtype=torch.long),
            raw=torch.as_tensor(obj["raw"], dtype=torch.float32),
            max_log_gate=float(obj["max_log_gate"]),
            model_name=obj.get("model_name"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "SignedLogMaskState":
        # Controller files are expected to be small tensor dictionaries.
        # `weights_only=True` avoids pickle object loading on newer PyTorch;
        # keep a fallback for older versions. Only load controllers you trust.
        try:
            obj = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # older torch
            obj = torch.load(path, map_location=map_location)
        return cls.from_dict(obj)


class _SignedLogMaskModule(nn.Module):
    """Small trainable module attached by forward hooks."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        layer_indices: torch.Tensor,
        channel_indices: torch.Tensor,
        *,
        hidden_size: int,
        max_log_gate: float,
        raw_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if layer_indices.numel() != channel_indices.numel():
            raise ValueError("layer_indices and channel_indices must have the same length")
        self.layers = layers
        self.hidden_size = int(hidden_size)
        self.max_log_gate = float(max_log_gate)
        self.register_buffer("layer_indices", layer_indices.detach().clone().long())
        self.register_buffer("channel_indices", channel_indices.detach().clone().long())
        if raw_init is None:
            raw_init = torch.zeros(int(layer_indices.numel()), dtype=torch.float32)
        self.raw = nn.Parameter(raw_init.detach().clone().float())
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._rebuild_index()

    @property
    def s(self) -> torch.Tensor:
        return self.max_log_gate * torch.tanh(self.raw)

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
            self._handles.append(self.layers[layer_id].register_forward_hook(self._make_hook(layer_id)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            pos, ch = self._by_layer[layer_id]
            pos = pos.to(device=h.device)
            ch = ch.to(device=h.device)
            scale = torch.ones(self.hidden_size, dtype=h.dtype, device=h.device)
            scale[ch] = torch.exp(self.s.to(device=h.device, dtype=h.dtype)[pos])
            h = h * scale.view(1, 1, -1)
            if rest is None:
                return h
            return (h,) + rest
        return hook

    def state(self, *, layer_path: str, n_layers: int, model_name: str | None) -> SignedLogMaskState:
        return SignedLogMaskState(
            layer_path=layer_path,
            n_layers=n_layers,
            hidden_size=self.hidden_size,
            layer_indices=self.layer_indices.detach().cpu(),
            channel_indices=self.channel_indices.detach().cpu(),
            raw=self.raw.detach().cpu(),
            max_log_gate=self.max_log_gate,
            model_name=model_name,
        )


class _ActivationScoreCapture:
    """Retain decoder layer activations and their gradients for gate scoring."""

    def __init__(self, layers: Sequence[nn.Module], layer_ids: Sequence[int]) -> None:
        self.captured: dict[int, torch.Tensor] = {}
        self._handles = [layers[i].register_forward_hook(self._make_hook(i)) for i in layer_ids]

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h.retain_grad()
            self.captured[idx] = h
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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gates = int(gates)
        self.max_log_gate = float(max_log_gate)
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
        was_attached = False
        if use_controller and self.controller is not None:
            self.controller.attach()
            was_attached = True
        try:
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
        finally:
            if was_attached:
                self.controller.remove()
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
        capture = _ActivationScoreCapture(self.decoder_layers, self.layer_ids)

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
        verbose: bool = True,
    ) -> dict[str, float]:
        if not examples:
            raise ValueError("fit requires at least one example")
        _freeze(self.model)
        device = _device_of(self.model)

        t0 = time.perf_counter()
        layer_indices, channel_indices = self._score_gates(
            examples, score_batches=score_batches, batch_size=batch_size, max_length=max_length
        )
        self.controller = _SignedLogMaskModule(
            self.decoder_layers,
            layer_indices.to(device),
            channel_indices.to(device),
            hidden_size=self.hidden_size,
            max_log_gate=self.max_log_gate,
        ).to(device)
        select_seconds = time.perf_counter() - t0

        opt = torch.optim.AdamW([self.controller.raw], lr=lr, weight_decay=weight_decay)
        train_batches = list(batches(examples, batch_size))
        if not train_batches:
            raise ValueError("no training batches")
        self.model.train(False)
        self.controller.attach()
        losses: list[float] = []
        train_t0 = time.perf_counter()
        try:
            for step in range(int(steps)):
                chunk = train_batches[step % len(train_batches)]
                batch = make_batch(self.tokenizer, chunk, device=device, max_length=max_length)
                opt.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                if l2 > 0:
                    loss = loss + float(l2) * self.controller.s.float().pow(2).mean()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().item()))
                if verbose and (step == 0 or (step + 1) % max(1, steps // 5) == 0):
                    print(f"step {step+1:4d}/{steps}: loss={losses[-1]:.6f}", flush=True)
        finally:
            self.controller.remove()
        train_seconds = time.perf_counter() - train_t0
        return {
            "selected_gates": float(self.controller.raw.numel()),
            "select_seconds": float(select_seconds),
            "train_seconds": float(train_seconds),
            "loss_first": float(losses[0]) if losses else math.nan,
            "loss_last": float(losses[-1]) if losses else math.nan,
        }

    def save(self, path: str | Path) -> None:
        if self.controller is None:
            raise ValueError("no controller has been fitted")
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        self.controller.state(
            layer_path=self.layer_path,
            n_layers=len(self.decoder_layers),
            model_name=model_name,
        ).save(path)

    def load(self, path: str | Path, *, map_location: str | torch.device | None = None) -> "ForwardFineTuner":
        if map_location is None:
            map_location = _device_of(self.model)
        state = SignedLogMaskState.load(path, map_location="cpu")
        if state.hidden_size != self.hidden_size:
            raise ValueError(
                f"controller hidden size {state.hidden_size} does not match model hidden size {self.hidden_size}"
            )
        if state.n_layers != len(self.decoder_layers):
            raise ValueError(
                f"controller layer count {state.n_layers} does not match model layer count {len(self.decoder_layers)}"
            )
        device = _device_of(self.model)
        self.controller = _SignedLogMaskModule(
            self.decoder_layers,
            state.layer_indices.to(device),
            state.channel_indices.to(device),
            hidden_size=self.hidden_size,
            max_log_gate=state.max_log_gate,
            raw_init=state.raw.to(device),
        ).to(device)
        return self

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 128, **generate_kwargs) -> str:
        if self.controller is None:
            raise ValueError("load or fit a controller before generate")
        device = _device_of(self.model)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
        self.model.eval()
        self.controller.attach()
        try:
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                **generate_kwargs,
            )
        finally:
            self.controller.remove()
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
        }
        Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
