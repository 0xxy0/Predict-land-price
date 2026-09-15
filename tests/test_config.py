from pathlib import Path

from house_prediction.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_settings() -> None:
    settings = Settings()
    assert settings.model_path == Path(PROJECT_ROOT / "models/house_price_ridge.joblib")
    assert settings.metadata_path == Path(PROJECT_ROOT / "models/house_price_ridge.metadata.json")
    assert settings.model_version == "0.1.0"


def test_default_api_settings() -> None:
    settings = Settings()

    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000
    assert settings.model_version == "0.1.0"


def test_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HOUSE_API_PORT", "9000")
    monkeypatch.setenv("HOUSE_MODEL_VERSION", "1.2.0")

    settings = Settings()

    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
    assert settings.model_version == "1.2.0"
