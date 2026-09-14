from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    model_path: Path = PROJECT_ROOT / "models" / "house_price_ridge.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "house_price_ridge.metadata.json"
    model_version: str = "0.1.0"