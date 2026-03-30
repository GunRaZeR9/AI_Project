from __future__ import annotations

import warnings
import os
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd


def _validate_gpu_runtime(torch_module) -> None:
    """Run a tiny forward pass to catch runtime GPU backend issues early."""
    import torch.nn as nn

    x = torch_module.randn(4, 1, device="cuda")
    drop = nn.Dropout(p=0.1).to("cuda")
    _ = drop(x)
    torch_module.cuda.synchronize()


@lru_cache(maxsize=1)
def _get_device() -> str:
    """Return a safe training device.

    Default behavior is conservative for interactive apps (e.g., Streamlit):
    use CPU unless GPU usage is explicitly opted in.

    Environment controls:
    - FORECAST_DEVICE=cpu|cuda : hard override
    - FORECAST_USE_GPU=1       : opt in to GPU auto-detection
    - FORECAST_VALIDATE_GPU=1  : run tiny CUDA runtime validation
    """
    forced = os.environ.get("FORECAST_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced

    use_gpu = os.environ.get("FORECAST_USE_GPU", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not use_gpu:
        return "cpu"

    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        # On some ROCm setups (e.g., gfx1036), first GPU allocation can segfault
        # unless HSA override is set. Avoid crash-prone auto-selection by requiring
        # explicit runtime configuration.
        if getattr(torch.version, "hip", None) and not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            warnings.warn(
                "ROCm detected, but HSA_OVERRIDE_GFX_VERSION is not set; falling back to CPU to avoid runtime crash. "
                "Set HSA_OVERRIDE_GFX_VERSION=10.3.0 and FORECAST_USE_GPU=1 to force GPU.",
                RuntimeWarning,
            )
            return "cpu"

        try:
            should_validate = os.environ.get("FORECAST_VALIDATE_GPU", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if should_validate:
                _validate_gpu_runtime(torch)
            return "cuda"
        except Exception as exc:
            backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
            warnings.warn(
                f"GPU backend ({backend}) is available but failed runtime validation; "
                f"falling back to CPU. Reason: {exc}",
                RuntimeWarning,
            )
            return "cpu"
    except ImportError:
        return "cpu"


def train_test_split_series(
    series: pd.Series, test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """Return a chronological train / test split of *series*."""
    n = len(series)
    split = int(np.floor(n * (1 - test_size)))
    return series.iloc[:split].copy(), series.iloc[split:].copy()
