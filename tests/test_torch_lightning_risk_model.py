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


def test_lightning_module_predict_labels_shape():
    module = IcingRiskLightningModule(input_dim=8, hidden_dims=(16, 8), dropout=0.0, threshold=0.4)
    x = torch.randn(4, 8)
    labels = module.predict_labels(x)
    assert labels.shape == (4,)
    assert torch.all((labels == 0) | (labels == 1))


def test_lightning_module_rejects_wrong_feature_count():
    module = IcingRiskLightningModule(input_dim=8)
    x = torch.randn(4, 7)
    with pytest.raises(ValueError, match="expected 8 features"):
        module(x)
