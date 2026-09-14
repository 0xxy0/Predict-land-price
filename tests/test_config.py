from pathlib import Path

from house_prediction.config import Settings


def test_config_settings() -> None:
    settings = Settings()
    assert settings.model_path == Path("models/house_price_ridge.joblib")
    assert settings.metadata_path == Path("models/house_price_ridge.metadata.json")
    assert settings.model_version == "0.1.0"
