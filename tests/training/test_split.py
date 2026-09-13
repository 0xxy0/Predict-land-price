import pandas as pd
import pytest

from house_prediction.training.split import (
    chronological_split,
    separate_features_target,
)


def make_frame_with_price():
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


def test_missing_price_column_raises_error() -> None:
    frame = make_frame_with_price().drop(columns=["price"])

    with pytest.raises(ValueError, match="Input frame must contain a 'price' column."):
        separate_features_target(frame)


def test_target_absent_from_features() -> None:
    features, target = separate_features_target(make_frame_with_price())

    assert "price" not in features.columns
    assert target.name == "price"
    assert len(features) == len(target)


def test_input_not_mutating() -> None:
    frame = make_frame_with_price()
    original_frame = frame.copy()

    separate_features_target(frame)

    pd.testing.assert_frame_equal(frame, original_frame)


def test_chronological_split_returns_expected_shapes() -> None:
    frame = make_frame_with_price()
    train, test = chronological_split(frame, test_size=0.2)

    assert len(train) + len(test) == len(frame)
    assert len(test) == pytest.approx(len(frame) * 0.2, rel=1e-2)


def test_chronological_split_row_count() -> None:
    frame = make_frame_with_price()
    train, test = chronological_split(frame, test_size=0.2)

    assert len(train) == pytest.approx(len(frame) * 0.8)
    assert len(test) == pytest.approx(len(frame) * 0.2)


def test_index_reset_after_split() -> None:
    frame = make_frame_with_price()
    train, test = chronological_split(frame, test_size=0.5)

    assert train.index.equals(pd.RangeIndex(len(train)))
    assert test.index.equals(pd.RangeIndex(len(test)))


def test_invalid_test_size_raises_error() -> None:
    frame = make_frame_with_price()

    with pytest.raises(ValueError, match="test_size must be between 0 and 1."):
        chronological_split(frame, test_size=1.5)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1."):
        chronological_split(frame, test_size=-0.1)


def test_missing_date_column_raises_error() -> None:
    frame = make_frame_with_price().drop(columns=["date"])

    with pytest.raises(ValueError, match="Input frame must contain a 'date' column."):
        chronological_split(frame, test_size=0.2)


def test_chronological_assertion() -> None:
    frame = make_frame_with_price()
    train, test = chronological_split(frame, test_size=0.3)

    train_dates = pd.to_datetime(train["date"])
    test_dates = pd.to_datetime(test["date"])

    assert train_dates.max() <= test_dates.min()
    assert train_dates.is_monotonic_increasing
    assert test_dates.is_monotonic_increasing


def test_random_split_for_comparison() -> None:
    """Random split should produce the requested train/test sizes."""
    frame = make_frame_with_price()

    test = frame.sample(frac=0.2, random_state=42)
    train = frame.drop(test.index)

    assert len(train) == 8
    assert len(test) == 2
    assert len(train) + len(test) == len(frame)


def test_random_split_is_reproducible() -> None:
    """A fixed random_state should produce the same random split."""
    frame = make_frame_with_price()

    test_one = frame.sample(frac=0.2, random_state=42)
    train_one = frame.drop(test_one.index)

    test_two = frame.sample(frac=0.2, random_state=42)
    train_two = frame.drop(test_two.index)

    pd.testing.assert_frame_equal(train_one, train_two)
    pd.testing.assert_frame_equal(test_one, test_two)
