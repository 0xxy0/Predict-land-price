import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from house_prediction.features.preprocess import (
    build_feature_pipeline,
)


@pytest.fixture
def make_frame_for_pipeline():
    """Create a small cleaned dataframe for testing the feature pipeline, including date, yr_renovated and other removed fields."""
    return pd.DataFrame(
        {
            "date": [
                "2014-06-01",
                "2015-07-01",
                "2014-08-01",
            ],
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
            "yr_built": [1994, 2005, 1984],
            "yr_renovated": [0, 2010, 0],
            "city": ["Seattle", "Bellevue", "Seattle"],
            "state": ["WA", "WA", "WA"],
            "zip_code": ["98101", "98004", "98101"],
        }
    )


def test_build_feature_pipeline_returns_pipeline():
    pipeline = build_feature_pipeline()

    assert isinstance(pipeline, Pipeline)


def test_pipeline_can_fit_and_transform(make_frame_for_pipeline):
    pipeline = build_feature_pipeline()

    transformed = np.asarray(pipeline.fit_transform(make_frame_for_pipeline))

    assert transformed.ndim == 2


def test_pipeline_output_is_numeric(make_frame_for_pipeline):
    pipeline = build_feature_pipeline()

    transformed = np.asarray(pipeline.fit_transform(make_frame_for_pipeline))

    assert np.issubdtype(transformed.dtype, np.number)


def test_pipeline_output_row_count_matches_input(make_frame_for_pipeline):
    pipeline = build_feature_pipeline()

    transformed = np.asarray(pipeline.fit_transform(make_frame_for_pipeline))

    assert transformed.shape[0] == make_frame_for_pipeline.shape[0]


def test_pipeline_handles_unseen_categories(make_frame_for_pipeline):
    pipeline = build_feature_pipeline()

    pipeline.fit(make_frame_for_pipeline)

    new_frame = make_frame_for_pipeline.copy()

    new_frame.loc[0, "city"] = "UnknownCity"
    new_frame.loc[0, "zip_code"] = "99999"

    transformed = np.asarray(pipeline.transform(new_frame))

    assert transformed.shape[0] == new_frame.shape[0]


def test_pipeline_does_not_modify_input(make_frame_for_pipeline):
    original = make_frame_for_pipeline.copy(deep=True)

    pipeline = build_feature_pipeline()

    pipeline.fit_transform(make_frame_for_pipeline)

    pd.testing.assert_frame_equal(
        make_frame_for_pipeline,
        original,
    )
