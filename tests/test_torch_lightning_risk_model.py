import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.torch_lightning_risk_model import (  # noqa: E402
    GroundTemperatureLightningModule,
    GroundTemperatureNet,
)


def test_ground_temperature_net_forward_shape():
    model = GroundTemperatureNet(input_dim=17, hidden_dims=(16, 8), dropout=0.0)
    output = model(torch.randn(5, 17))

    assert output.shape == (5,)


def test_ground_temperature_lightning_module_returns_one_value_per_row():
    module = GroundTemperatureLightningModule(input_dim=17, hidden_dims=(16, 8), dropout=0.0)
    output = module(torch.randn(5, 17))

    assert output.shape == (5,)


def test_ground_temperature_module_rejects_wrong_feature_count():
    module = GroundTemperatureLightningModule(input_dim=17)

    with pytest.raises(ValueError, match="expected features"):
        module(torch.randn(2, 16))


def test_ground_temperature_training_step_returns_finite_loss():
    module = GroundTemperatureLightningModule(input_dim=17, hidden_dims=(16,), dropout=0.0)
    features = torch.randn(8, 17)
    target = torch.randn(8)

    loss = module.training_step((features, target), 0)

    assert torch.isfinite(loss)
