from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from house_prediction.features.build import build_features

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "age_at_sale",
    "renovation_indicator",
    "sale_year",
    "sale_month",
]

CATEGORICAL_FEATURES = [
    "city",
    "state",
    "zip_code",
]


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer for model features."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_feature_pipeline() -> Pipeline:
    """Build a pipeline that includes feature engineering and preprocessing."""

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(build_features, validate=False)),
            ("preprocessing", build_preprocessor()),
        ]
    )
