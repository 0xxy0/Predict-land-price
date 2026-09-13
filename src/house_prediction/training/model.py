from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from house_prediction.features.preprocess import build_feature_pipeline


def build_mean_price_baseline() -> Pipeline:
    """Build a baseline that predicts the training mean price."""
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def build_ridge_model(alpha: float = 1.0) -> Pipeline:
    """Build a Ridge regression model with the shared feature pipeline."""
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero.")

    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
