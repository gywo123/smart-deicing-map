"""PyTorch + PyTorch Lightning model for road icing risk prediction."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class IcingRiskNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class IcingRiskLightningModule(pl.LightningModule):
    """Binary risk model predicting probability of icing for each road segment."""

    def __init__(self, input_dim: int, lr: float = 1e-3, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = IcingRiskNet(input_dim=input_dim)
        self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.model(features)
        return torch.sigmoid(logits)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=self._pos_weight)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        acc = (preds == y.long()).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
