import pandas as pd

from house_prediction.features.build import FEATURE_COLUMNS, build_features


def make_cleaned_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2014-05-02 00:00:00",
                "price": 450000.0,
                "bedrooms": 3,
                "bathrooms": 1.5,
                "floors": 1.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "sqft_living": 1340,
                "sqft_lot": 7912,
                "sqft_above": 1340,
                "sqft_basement": 0,
                "yr_built": 1955,
                "yr_renovated": 0,
                "street": "Example Street",
                "city": "Seattle",
                "statezip": "WA 98101",
                "state": "WA",
                "zip_code": "98101",
                "country": "USA",
            },
            {
                "date": "2015-11-20 00:00:00",
                "price": 600000.0,
                "bedrooms": 4,
                "bathrooms": 2.0,
                "floors": 2.0,
                "waterfront": 0,
                "view": 1,
                "condition": 4,
                "sqft_living": 2000,
                "sqft_lot": 8000,
                "sqft_above": 1600,
                "sqft_basement": 400,
                "yr_built": 1980,
                "yr_renovated": 2005,
                "street": "Second Street",
                "city": "Seattle",
                "statezip": "WA 98102",
                "state": "WA",
                "zip_code": "98102",
                "country": "USA",
            },
        ]
    )


def test_build_features_returns_expected_columns() -> None:
    features = build_features(make_cleaned_frame())

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert "price" not in features.columns
    assert "street" not in features.columns
    assert "date" not in features.columns
    assert "statezip" not in features.columns
    assert "yr_built" not in features.columns
    assert "yr_renovated" not in features.columns


def test_build_features_derives_date_and_age_features() -> None:
    features = build_features(make_cleaned_frame())

    assert features["sale_year"].tolist() == [2014, 2015]
    assert features["sale_month"].tolist() == [5, 11]
    assert features["age_at_sale"].tolist() == [59, 35]


def test_build_features_derives_renovation_indicator() -> None:
    features = build_features(make_cleaned_frame())

    assert features["renovation_indicator"].tolist() == [0, 1]


def test_build_features_does_not_mutate_input() -> None:
    frame = make_cleaned_frame()
    original = frame.copy(deep=True)

    build_features(frame)

    pd.testing.assert_frame_equal(frame, original)
