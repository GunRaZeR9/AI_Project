from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _normalize(data: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize *data* and return ``(norm_array, params_dict)``."""
    if mode == "minmax":
        x_min = float(data.min())
        x_max = float(data.max())
        rng = (x_max - x_min) if x_max != x_min else 1.0
        return (data - x_min) / rng, {"mode": mode, "x_min": x_min, "x_max": x_max}
    if mode == "zscore":
        x_mean = float(data.mean())
        x_std = float(data.std())
        x_std = x_std if x_std != 0 else 1.0
        return (data - x_mean) / x_std, {"mode": mode, "x_mean": x_mean, "x_std": x_std}
    return data.copy(), {"mode": "none"}


def _apply_norm_params(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply previously computed normalization parameters to new data."""
    mode = params["mode"]
    if mode == "minmax":
        rng = (params["x_max"] - params["x_min"]) if params["x_max"] != params["x_min"] else 1.0
        return (data - params["x_min"]) / rng
    if mode == "zscore":
        return (data - params["x_mean"]) / (params["x_std"] if params["x_std"] != 0 else 1.0)
    return data.copy()


def _denormalize(norm_data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Reverse a normalization transform."""
    mode = params["mode"]
    if mode == "minmax":
        rng = (params["x_max"] - params["x_min"]) if params["x_max"] != params["x_min"] else 1.0
        return norm_data * rng + params["x_min"]
    if mode == "zscore":
        return norm_data * (params["x_std"] if params["x_std"] != 0 else 1.0) + params["x_mean"]
    return norm_data.copy()
