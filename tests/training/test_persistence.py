from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from tests.training.test_split import make_frame_with_price

from house_prediction.training.model import build_ridge_model
from house_prediction.training.persistence import load_model, save_model


def test_fitted_model_can_be_saved(tmp_path: Path) -> None:
    frame = make_frame_with_price()
    train = frame.iloc[:8]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    model = build_ridge_model()
    model.fit(X_train, y_train)

    path = tmp_path / "models" / "ridge_model.joblib"

    save_model(model, path)

    assert path.exists()


def test_saved_model_can_be_loaded(tmp_path: Path) -> None:
    frame = make_frame_with_price()
    train = frame.iloc[:8]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    model = build_ridge_model()
    model.fit(X_train, y_train)

    path = tmp_path / "ridge_model.joblib"
    save_model(model, path)

    loaded_model = load_model(path)

    assert isinstance(loaded_model, Pipeline)


def test_predictions_before_and_after_loading_are_identical(
    tmp_path: Path,
) -> None:
    frame = make_frame_with_price()
    train = frame.iloc[:8]
    test = frame.iloc[8:]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]
    X_test = test.drop(columns=["price"])

    model = build_ridge_model()
    model.fit(X_train, y_train)

    predictions_before = model.predict(X_test)

    path = tmp_path / "ridge_model.joblib"
    save_model(model, path)

    loaded_model = load_model(path)
    predictions_after = loaded_model.predict(X_test)

    np.testing.assert_array_equal(predictions_before, predictions_after)


def test_missing_model_path_raises_file_not_found_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing" / "ridge_model.joblib"

    with pytest.raises(FileNotFoundError):
        load_model(path)


def test_parent_directory_is_created_automatically(
    tmp_path: Path,
) -> None:
    frame = make_frame_with_price()
    train = frame.iloc[:8]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    model = build_ridge_model()
    model.fit(X_train, y_train)

    path = tmp_path / "models" / "ridge" / "ridge_model.joblib"

    assert not path.parent.exists()

    save_model(model, path)

    assert path.parent.exists()
    assert path.exists()
