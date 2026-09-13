import pandas as pd
from sklearn.pipeline import Pipeline

from house_prediction.training.model import build_mean_price_baseline


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


def test_build_mean_price_baseline_returns_pipeline() -> None:
    model = build_mean_price_baseline()

    assert isinstance(model, Pipeline)


def test_mean_price_baseline_fits_successfully() -> None:
    frame = make_frame_with_price()

    X = frame.drop(columns=["price"])
    y = frame["price"]

    model = build_mean_price_baseline()

    model.fit(X, y)


def test_mean_price_baseline_predictions_match_test_rows() -> None:
    frame = make_frame_with_price()

    train = frame.iloc[:8]
    test = frame.iloc[8:]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    X_test = test.drop(columns=["price"])

    model = build_mean_price_baseline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(test)


def test_mean_price_baseline_predictions_are_constant() -> None:
    frame = make_frame_with_price()

    train = frame.iloc[:8]
    test = frame.iloc[8:]

    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    X_test = test.drop(columns=["price"])

    model = build_mean_price_baseline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(set(predictions)) == 1


def test_mean_price_baseline_does_not_mutate_input() -> None:
    frame = make_frame_with_price()
    original_frame = frame.copy()

    X = frame.drop(columns=["price"])
    y = frame["price"]

    model = build_mean_price_baseline()
    model.fit(X, y)

    pd.testing.assert_frame_equal(frame, original_frame)
