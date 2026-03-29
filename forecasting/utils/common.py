from __future__ import annotations

import warnings
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
    """Return a validated GPU device, else safely fall back to CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        try:
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
