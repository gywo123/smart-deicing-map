import pytest


torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

from src.torch_lightning_risk_model import IcingRiskLightningModule


def test_lightning_module_forward_shape():
    module = IcingRiskLightningModule(input_dim=8)
    x = torch.randn(4, 8)
    y = module(x)
    assert y.shape == (4,)
    assert torch.all((y >= 0) & (y <= 1))
