import numpy as np
import pandas as pd
import pytest

from house_prediction.training.evaluate import evaluate_regression


def test_mae_is_calculated_correctly() -> None:
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 310.0])

    metrics = evaluate_regression(actual, predicted)

    assert metrics["mae"] == pytest.approx(13.333333333333334)


def test_rmse_is_calculated_correctly() -> None:
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 310.0])

    metrics = evaluate_regression(actual, predicted)

    assert metrics["rmse"] == pytest.approx(14.142135623730951)


def test_result_contains_mae_and_rmse() -> None:
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 310.0])

    metrics = evaluate_regression(actual, predicted)

    assert set(metrics) == {"mae", "rmse"}


def test_perfect_predictions_return_zero_metrics() -> None:
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = np.array([100.0, 200.0, 300.0])

    metrics = evaluate_regression(actual, predicted)

    assert metrics == {"mae": 0.0, "rmse": 0.0}
