from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, path: Path) -> None:
    """Save a fitted model to disk

    Args:
        model (Pipeline): model
        path (Path): output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """Load a fitted model from disk

    Args:
        path (Path): input path
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise ValueError(f"Loaded object is not a Pipeline: {type(model)}")
    return model
