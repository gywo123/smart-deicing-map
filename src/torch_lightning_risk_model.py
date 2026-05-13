"""도로 겨울철 위험 예측용 PyTorch + PyTorch Lightning 모델."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class IcingRiskNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive values")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class IcingRiskLightningModule(pl.LightningModule):
    """도로 세그먼트별 결빙 확률을 예측하는 이진 위험도 모델."""

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        pos_weight: float = 1.0,
        hidden_dims: tuple[int, ...] = (64, 64),
        dropout: float = 0.1,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in the range (0, 1)")

        self.save_hyperparameters()
        self.model = IcingRiskNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """0~1 범위의 결빙 확률을 반환한다."""
        return torch.sigmoid(self.predict_logits(features))

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        """손실 계산이나 보정 후처리에 사용할 raw logit을 반환한다."""
        if features.ndim != 2:
            raise ValueError("features must be a 2D tensor shaped (batch, input_dim)")
        if features.shape[-1] != self.hparams.input_dim:
            raise ValueError(f"expected {self.hparams.input_dim} features, got {features.shape[-1]}")
        return self.model(features)

    def predict_labels(self, features: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        """설정된 확률 threshold를 기준으로 이진 결빙 라벨을 반환한다."""
        cutoff = self.hparams.threshold if threshold is None else threshold
        if not 0 < cutoff < 1:
            raise ValueError("threshold must be in the range (0, 1)")
        return (self(features) >= cutoff).long()

    def _classification_metrics(self, probs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        preds = (probs >= self.hparams.threshold).long()
        labels = targets.long()
        true_pos = ((preds == 1) & (labels == 1)).float().sum()
        false_pos = ((preds == 1) & (labels == 0)).float().sum()
        false_neg = ((preds == 0) & (labels == 1)).float().sum()
        true_neg = ((preds == 0) & (labels == 0)).float().sum()
        eps = torch.tensor(1e-8, device=probs.device)

        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        specificity = true_neg / (true_neg + false_pos + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        balanced_acc = (recall + specificity) / 2
        acc = (preds == labels).float().mean()

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_acc": balanced_acc,
        }

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self.predict_logits(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self._pos_weight)
        probs = torch.sigmoid(logits)
        metrics = self._classification_metrics(probs, y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, prog_bar=name in {"acc", "recall", "f1"}, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class AccidentRiskLightningModule(IcingRiskLightningModule):
    """겨울철 사고 확률을 예측하는 MLP 이진 분류기.

    결빙 모델과 같은 tabular MLP 구조를 의도적으로 재사용한다.
    두 모델 파이프라인을 쓰면 프로젝트의 기존 MLP 계열을 유지하면서도
    "이 도로가 얼 것인가?"와 "이 장소가 제설 우선순위를 검증할 만큼
    위험한가?"를 분리해서 설명할 수 있다.
    """
