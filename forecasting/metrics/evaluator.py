from __future__ import annotations

from typing import Dict

import numpy as np


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Return MSE, MAE, MAPE (%), and RMSE for the given arrays.

    MAPE is skipped where actual == 0 to avoid division-by-zero.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    mask = actual != 0
    mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100) if mask.any() else float("nan")
    return {"MSE": mse, "MAE": mae, "MAPE (%)": mape, "RMSE": rmse}
