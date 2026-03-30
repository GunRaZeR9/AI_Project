from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from forecasting.normalization.scalers import _apply_norm_params, _denormalize


def gru_forecast(
    result: Dict[str, Any],
    seed_vals: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Recursive multi-step forecast from a trained result dict."""
    import torch

    model = result["model"]
    norm_params = result["norm_params"]
    seq_len = result["seq_len"]
    device = result.get("device", model.device)

    seed = seed_vals.astype(float)
    norm_seed = _apply_norm_params(seed, norm_params)
    window: List[float] = list(norm_seed[-seq_len:] if len(norm_seed) >= seq_len else norm_seed)

    preds_norm: List[float] = []
    model.eval_mode()
    with torch.no_grad():
        for _ in range(steps):
            ctx = np.array(window[-seq_len:], dtype=np.float32)
            x_in = torch.from_numpy(ctx).unsqueeze(0).unsqueeze(-1).to(device)
            p = float(model(x_in).squeeze())
            preds_norm.append(p)
            window.append(p)

    return _denormalize(np.array(preds_norm, dtype=float), norm_params)


def gru_one_step_predict(
    result: Dict[str, Any],
    full_vals: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    """One-step-ahead teacher-forced predictions on full_vals[start_idx:]."""
    import torch

    model = result["model"]
    norm_params = result["norm_params"]
    seq_len = result["seq_len"]
    device = result.get("device", model.device)

    vals = full_vals.astype(float)
    norm_vals = _apply_norm_params(vals, norm_params)

    preds_norm: List[float] = []
    model.eval_mode()
    with torch.no_grad():
        for i in range(start_idx, len(vals)):
            ctx_start = max(0, i - seq_len)
            ctx = norm_vals[ctx_start:i]
            if len(ctx) < seq_len:
                fill = ctx[0] if len(ctx) else 0.0
                ctx = np.concatenate([np.full(seq_len - len(ctx), fill), ctx])
            ctx = np.array(ctx[-seq_len:], dtype=np.float32)
            x_in = torch.from_numpy(ctx).unsqueeze(0).unsqueeze(-1).to(device)
            p = float(model(x_in).squeeze())
            preds_norm.append(p)

    return _denormalize(np.array(preds_norm, dtype=float), norm_params)


def ensemble_mean_one_step_predict(
    results: List[Dict[str, Any]],
    full_vals: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    """Mean aggregation for one-step-ahead predictions across models."""
    if not results:
        raise ValueError("results cannot be empty")
    preds = [gru_one_step_predict(r, full_vals, start_idx=start_idx) for r in results]
    stacked = np.stack(preds, axis=0)
    return np.mean(stacked, axis=0)
