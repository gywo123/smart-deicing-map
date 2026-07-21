"""실제 ASOS 지면온도 예측용 PyTorch Lightning MLP."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class GroundTemperatureNet(nn.Module):
    """표준화된 기상 피처에서 미래 지면온도 변화량을 예측한다."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(hidden <= 0 for hidden in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class GroundTemperatureLightningModule(pl.LightningModule):
    """실제 ASOS 지면온도 변화량을 학습하는 residual 회귀 모델."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.08,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_plateau_scheduler: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = GroundTemperatureNet(input_dim, hidden_dims, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.hparams.input_dim:
            raise ValueError(
                f"expected features shaped (batch, {self.hparams.input_dim}), "
                f"got {tuple(features.shape)}"
            )
        return self.model(features)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        features, target = batch
        prediction = self(features)
        loss = F.smooth_l1_loss(prediction, target, beta=0.5)
        rmse = torch.sqrt(torch.mean((prediction - target) ** 2))
        if self._trainer is not None:
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=stage == "val")
            self.log(f"{stage}_rmse", rmse, on_step=False, on_epoch=True, prog_bar=stage == "val")
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> object:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.use_plateau_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
