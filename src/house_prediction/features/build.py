import pandas as pd

FEATURE_COLUMNS = [
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
    "city",
    "state",
    "zip_code",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the house prediction model.
    """
    frame = df.copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"])

    frame["renovation_indicator"] = (frame["yr_renovated"] != 0).astype(int)
    frame["sale_year"] = frame["date"].dt.year
    frame["sale_month"] = frame["date"].dt.month
    frame["age_at_sale"] = frame["sale_year"] - frame["yr_built"]

    # Ensure all required columns are present
    missing_columns = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return frame[FEATURE_COLUMNS].copy()
