import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from house_prediction.features.preprocess import (
    NUMERIC_FEATURES,
    build_preprocessor,
)


@pytest.fixture
def make_cleaned_frame():
    """Create a small cleaned dataframe matching the preprocessing inputs."""
    return pd.DataFrame(
        {
            "bedrooms": [3, 4, 2],
            "bathrooms": [2.0, 3.0, 1.0],
            "floors": [1.0, 2.0, 1.0],
            "waterfront": [0, 1, 0],
            "view": [0, 2, 0],
            "condition": [3, 4, 3],
            "sqft_living": [1500, 2500, 1000],
            "sqft_lot": [5000, 7000, 4000],
            "sqft_above": [1200, 2000, 800],
            "sqft_basement": [300, 500, 200],
            "age_at_sale": [20, 10, 30],
            "renovation_indicator": [0, 1, 0],
            "sale_year": [2014, 2015, 2014],
            "sale_month": [6, 7, 8],
            "city": ["Seattle", "Bellevue", "Seattle"],
            "state": ["WA", "WA", "WA"],
            "zip_code": ["98101", "98004", "98101"],
        }
    )


def test_build_preprocessor_returns_column_transformer():
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_numeric_columns_are_scaled(make_cleaned_frame):
    preprocessor = build_preprocessor()

    transformed = np.asarray(preprocessor.fit_transform(make_cleaned_frame))

    numeric_output = transformed[:, : len(NUMERIC_FEATURES)]

    # StandardScaler should produce approximately zero mean
    # and unit standard deviation for numeric features.
    assert np.allclose(numeric_output.mean(axis=0), 0.0)
    assert np.allclose(numeric_output.std(axis=0), 1.0)


def test_categorical_columns_are_one_hot_encoded(make_cleaned_frame):
    preprocessor = build_preprocessor()

    preprocessor.fit(make_cleaned_frame)

    categorical_transformer = preprocessor.named_transformers_["categorical"]

    assert isinstance(categorical_transformer, OneHotEncoder)

    transformed = np.asarray(preprocessor.transform(make_cleaned_frame))

    # One-hot encoding creates more columns than the original
    # categorical columns because each category gets its own column.
    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_unknown_categories_do_not_raise_errors(make_cleaned_frame):
    preprocessor = build_preprocessor()

    preprocessor.fit(make_cleaned_frame)

    new_frame = make_cleaned_frame.copy()

    new_frame.loc[0, "city"] = "UnknownCity"
    new_frame.loc[0, "state"] = "XX"
    new_frame.loc[0, "zip_code"] = "99999"

    transformed = preprocessor.transform(new_frame)

    assert transformed.shape[0] == new_frame.shape[0]


def test_transformed_output_is_numeric(make_cleaned_frame):
    preprocessor = build_preprocessor()

    transformed = np.asarray(preprocessor.fit_transform(make_cleaned_frame))

    assert np.issubdtype(transformed.dtype, np.number)


def test_transformed_row_count_equals_input(make_cleaned_frame):
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(make_cleaned_frame)

    assert transformed.shape[0] == make_cleaned_frame.shape[0]
