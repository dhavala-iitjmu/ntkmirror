from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch.nn as nn


_LAYER_PATHS = (
    "model.layers",              # Llama, Mistral, Qwen2/Qwen2.5
    "model.decoder.layers",      # OPT/BART-style decoder stacks
    "transformer.h",             # GPT-2
    "gpt_neox.layers",           # GPT-NeoX/Pythia
    "decoder.layers",            # some encoder-decoder decoders
)


def _get_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model) -> tuple[str, Sequence[nn.Module]]:
    """Return (path, decoder_layers) for common Hugging Face causal LMs."""
    for path in _LAYER_PATHS:
        layers = _get_attr_path(model, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
            return path, layers

    # Fallback: choose the largest ModuleList. This keeps the package usable on
    # architectures we did not name, while failing loudly if no stack exists.
    best_name = None
    best = None
    best_len = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > best_len:
            best_name = name
            best = module
            best_len = len(module)
    if best is not None and best_len > 1:
        return str(best_name), best
    raise ValueError(
        "could not locate decoder layers. Pass a standard Hugging Face causal LM "
        "or extend ntkmirror.layers._LAYER_PATHS."
    )


def infer_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model", "word_embed_proj_dim"):
        if cfg is not None and hasattr(cfg, name):
            value = int(getattr(cfg, name))
            if value > 0:
                return value
    emb = model.get_input_embeddings()
    if hasattr(emb, "embedding_dim"):
        return int(emb.embedding_dim)
    raise ValueError("could not infer hidden size from model.config or input embeddings")


def parse_layers(spec: str, n_layers: int) -> list[int]:
    """Parse a tiny layer spec: all, last, last:N, or comma-separated indices."""
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec == "last":
        return [n_layers - 1]
    if spec.startswith("last:"):
        k = int(spec.split(":", 1)[1])
        if k <= 0:
            raise ValueError("last:N requires N>0")
        return list(range(max(0, n_layers - k), n_layers))
    out = []
    for part in spec.split(","):
        idx = int(part.strip())
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"layer index {idx} outside [0, {n_layers})")
        out.append(idx)
    if not out:
        raise ValueError("empty layer specification")
    return sorted(set(out))
