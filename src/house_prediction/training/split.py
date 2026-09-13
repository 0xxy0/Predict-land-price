import pandas as pd


def separate_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw feature columns from the price target."""
    if "price" not in frame.columns:
        raise ValueError("Input frame must contain a 'price' column.")

    copied_frame = frame.copy(deep=True)
    features = copied_frame.drop(columns=["price"])
    target = copied_frame["price"].copy()

    return features, target


def chronological_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically into training and testing frames."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if "date" not in frame.columns:
        raise ValueError("Input frame must contain a 'date' column.")

    sorted_frame = frame.copy(deep=True)
    sorted_frame["date"] = pd.to_datetime(sorted_frame["date"])
    sorted_frame = sorted_frame.sort_values("date").reset_index(drop=True)

    test_rows = max(1, int(len(sorted_frame) * test_size))
    split_index = len(sorted_frame) - test_rows

    train_frame = sorted_frame.iloc[:split_index].reset_index(drop=True)
    test_frame = sorted_frame.iloc[split_index:].reset_index(drop=True)

    return train_frame, test_frame
