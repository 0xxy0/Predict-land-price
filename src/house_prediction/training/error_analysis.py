import numpy as np
import pandas as pd


def build_error_report(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Return evaluation rows with prediction and error columns."""
    if len(frame) != len(predictions):
        raise ValueError("Prediction count must match frame row count.")

    report = frame.copy(deep=True)
    report["prediction"] = predictions
    report["error"] = report["price"] - report["prediction"]
    report["absolute_error"] = report["error"].abs()
    report["price_band"] = pd.cut(
        report["price"],
        bins=[0, 250_000, 500_000, 750_000, 1_000_000, np.inf],
        labels=["0-250k", "250k-500k", "500k-750k", "750k-1M", "over-1M"],
    )
    report["sale_year"] = pd.to_datetime(report["date"]).dt.year

    return report
