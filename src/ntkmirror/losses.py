from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over labels != -100."""
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    if not bool(mask.any()):
        raise ValueError("batch contains no supervised next-token labels")
    return F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="mean")


def token_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    if not bool(mask.any()):
        return 0, 0
    pred = shift_logits.argmax(dim=-1)
    correct = int((pred[mask] == shift_labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total
