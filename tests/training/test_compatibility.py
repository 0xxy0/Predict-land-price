import math
from pathlib import Path

from sklearn.pipeline import Pipeline

from house_prediction.training.metadata import load_metadata
from house_prediction.training.persistence import load_model


MODEL_PATH = Path("models/house_price_ridge.joblib")
METADATA_PATH = Path("models/house_price_ridge.metadata.json")


def test_saved_model_is_compatible() -> None:
    model = load_model(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)

    assert isinstance(model, Pipeline)
    assert model.named_steps["model"].__class__.__name__ == "Ridge"
    assert metadata["model_version"] == "0.1.0"
    assert metadata["model_name"] == "ridge"
    
def test_saved_metadata_contains_required_fields() -> None:
    metadata = load_metadata(METADATA_PATH)

    required_fields = {
        "model_name",
        "model_version",
        "alpha",
        "split_strategy",
        "test_size",
        "evaluation_metrics",
        "feature_columns",
        "training_rows",
        "dataset_sha256",
    }

    assert required_fields <= metadata.keys()