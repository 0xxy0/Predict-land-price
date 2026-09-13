import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from house_prediction.training.evaluate import evaluate_regression
from house_prediction.training.model import build_ridge_model
from house_prediction.training.split import chronological_split, separate_features_target


def train_ridge_model(
    frame: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float]]:
    """Train a Ridge regression model on the provided DataFrame."""
    train, test = chronological_split(frame)
    X_train, y_train = separate_features_target(train)
    X_test, y_test = separate_features_target(test)

    ridge_model = build_ridge_model(alpha=1.0)
    ridge_model.fit(X_train, y_train)

    ridge_predictions = ridge_model.predict(X_test)
    assert isinstance(ridge_predictions, np.ndarray)
    ridge_metrics = evaluate_regression(y_test, ridge_predictions)

    return ridge_model, ridge_metrics


def train_final_ridge_model(
    frame: pd.DataFrame,
) -> Pipeline:
    """Train a Ridge regression model on the entire DataFrame."""
    X, y = separate_features_target(frame)

    ridge_model = build_ridge_model(alpha=1.0)
    ridge_model.fit(X, y)

    return ridge_model
