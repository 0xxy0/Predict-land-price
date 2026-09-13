import numpy as np
import pandas as pd


def evaluate_regression(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}
