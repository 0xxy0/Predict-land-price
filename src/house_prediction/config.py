from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="HOUSE_",
        extra="ignore",
    )

    model_path: Path = PROJECT_ROOT / "models" / "house_price_ridge.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "house_price_ridge.metadata.json"
    model_version: str = "0.1.0"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
