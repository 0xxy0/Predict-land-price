import numpy as np
import pandas as pd

from house_prediction.training.evaluate import evaluate_regression
from house_prediction.training.model import build_mean_price_baseline, build_ridge_model


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


def test_ridge_beats_mean_baseline_chronologically() -> None:
    df = make_frame_with_price()
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    X_test = test.drop(columns=["price"])
    y_test = test["price"]

    baseline = build_mean_price_baseline()
    ridge = build_ridge_model(alpha=1.0)

    baseline.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    assert isinstance(baseline_predictions, np.ndarray)

    baseline_metrics = evaluate_regression(
        y_test,
        baseline_predictions,
    )

    ridge_predictions = ridge.predict(X_test)
    assert isinstance(ridge_predictions, np.ndarray)

    ridge_metrics = evaluate_regression(
        y_test,
        ridge_predictions,
    )

    assert ridge_metrics["mae"] < baseline_metrics["mae"]
    assert ridge_metrics["rmse"] < baseline_metrics["rmse"]
