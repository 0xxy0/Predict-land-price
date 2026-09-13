import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from house_prediction.training.train import train_final_ridge_model, train_ridge_model


def make_frame_with_price() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2016-11-25",
                "2014-06-01",
                "2014-08-01",
                "2014-10-15",
                "2015-01-10",
                "2015-03-20",
                "2015-07-01",
                "2015-09-12",
                "2016-02-05",
                "2016-06-18",
            ],
            "bedrooms": [3, 2, 4, 3, 2, 4, 3, 5, 2, 4],
            "bathrooms": [2.0, 1.0, 3.0, 2.0, 1.5, 3.0, 2.0, 4.0, 1.0, 2.5],
            "floors": [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "waterfront": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "view": [0, 0, 2, 0, 1, 2, 0, 3, 0, 1],
            "condition": [3, 3, 4, 3, 3, 4, 3, 5, 3, 4],
            "sqft_living": [
                1500,
                1000,
                2500,
                1800,
                1200,
                2700,
                1600,
                3200,
                1100,
                2400,
            ],
            "sqft_lot": [
                5000,
                4000,
                7000,
                5500,
                4500,
                8000,
                5200,
                9000,
                4200,
                6500,
            ],
            "sqft_above": [
                1200,
                800,
                2000,
                1400,
                1000,
                2100,
                1300,
                2500,
                900,
                1900,
            ],
            "sqft_basement": [
                300,
                200,
                500,
                400,
                200,
                600,
                300,
                700,
                200,
                500,
            ],
            "yr_built": [
                1994,
                1984,
                2005,
                1998,
                1990,
                2008,
                1995,
                2010,
                1985,
                2003,
            ],
            "yr_renovated": [
                0,
                0,
                2010,
                0,
                0,
                2015,
                0,
                2020,
                0,
                2012,
            ],
            "city": [
                "Seattle",
                "Seattle",
                "Bellevue",
                "Seattle",
                "Seattle",
                "Bellevue",
                "Seattle",
                "Bellevue",
                "Seattle",
                "Bellevue",
            ],
            "state": ["WA"] * 10,
            "zip_code": [
                "98101",
                "98101",
                "98004",
                "98103",
                "98115",
                "98004",
                "98105",
                "98004",
                "98101",
                "98004",
            ],
            "price": [
                300000.0,
                200000.0,
                500000.0,
                350000.0,
                275000.0,
                550000.0,
                375000.0,
                700000.0,
                225000.0,
                525000.0,
            ],
        }
    )


def test_returned_object_is_fitted_pipeline() -> None:
    frame = make_frame_with_price()

    model, _ = train_ridge_model(frame)

    assert isinstance(model, Pipeline)

    predictions = model.predict(frame.drop(columns=["price"]))

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(frame)


def test_metrics_contain_mae_and_rmse() -> None:
    frame = make_frame_with_price()

    _, metrics = train_ridge_model(frame)

    assert set(metrics) == {"mae", "rmse"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_model_can_predict() -> None:
    frame = make_frame_with_price()
    model, _ = train_ridge_model(frame)
    X = frame.drop(columns=["price"]).iloc[:2]  # Take first two rows for prediction
    predictions = model.predict(X)
    assert predictions is not None
    assert len(predictions) == 2
    assert isinstance(predictions, np.ndarray)


def test_frame_not_mutated() -> None:
    frame = make_frame_with_price()
    original_frame = frame.copy()
    train_ridge_model(frame)
    pd.testing.assert_frame_equal(frame, original_frame)  # Ensure the original frame is unchanged


def test_train_final_ridge_model_returns_predictable_pipeline() -> None:
    frame = make_frame_with_price()
    original = frame.copy(deep=True)

    model = train_final_ridge_model(frame)

    predictions = model.predict(frame.drop(columns=["price"]))

    assert isinstance(model, Pipeline)
    assert len(predictions) == len(frame)
    pd.testing.assert_frame_equal(frame, original)
