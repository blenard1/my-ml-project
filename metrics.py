from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def soft_iou(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def saliency_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 1.0,
    iou_weight: float = 0.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce_weight * bce + iou_weight * (1.0 - soft_iou(logits, targets))


@torch.no_grad()
def binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    labels = targets >= 0.5

    tp = (preds & labels).sum().float()
    fp = (preds & ~labels).sum().float()
    fn = (~preds & labels).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    mae = torch.mean(torch.abs(probs - targets))

    return {
        "iou": float(iou.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "f1": float(f1.detach().cpu()),
        "mae": float(mae.detach().cpu()),
    }


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class MetricAverager:
    meters: dict[str, AverageMeter] = field(default_factory=dict)

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for key, value in values.items():
            self.meters.setdefault(key, AverageMeter()).update(value, n=n)

    def averages(self) -> dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}
