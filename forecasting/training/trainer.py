from __future__ import annotations

import random
import warnings
from typing import Any, Dict, List

import numpy as np

from forecasting.models.architectures import _GRUNet, _MODEL_CLASSES
from forecasting.normalization.scalers import _apply_norm_params, _normalize
from forecasting.training.lr_scheduler import get_scheduler
from forecasting.utils.common import _get_device


def _is_gpu_runtime_failure(exc: Exception) -> bool:
    """Return True for known CUDA/ROCm runtime failures that should fallback to CPU."""
    msg = str(exc).lower()
    gpu_markers = (
        "miopen",
        "hiprtc",
        "rocm",
        "miopenstatus",
        "cuda error",
        "cudnn",
    )
    return any(marker in msg for marker in gpu_markers)


def _set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding for numpy/python/torch."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_gru(
    train_vals: np.ndarray,
    seq_len: int = 60,
    hidden: int = 64,
    num_layers: int = 2,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_train_points: int = 12_000,
    normalization: str = "minmax",
    val_fraction: float = 0.1,
    activation: str = "relu",
    dropout: float = 0.0,
    optimizer_name: str = "adam",
    loss_fn_name: str = "mse",
    weight_decay: float = 0.0,
    l1_lambda: float = 0.0,
    model_type: str = "gru",
    initial_state: Dict[str, Any] | None = None,
    device_override: str | None = None,
    random_seed: int | None = None,
    epoch_callback=None,
    lr_scheduler_type: str = "constant",
    lr_scheduler_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train a GRU/LSTM/RNN model and return a results dictionary.
    
    Args:
        lr_scheduler_type: Type of learning rate scheduler ('constant', 'step', 
                          'exponential', 'cosine', 'warmup_decay'). Default: 'constant'.
        lr_scheduler_kwargs: Additional arguments for the scheduler (e.g., {'step_size': 10}).
    """
    import torch
    import torch.nn as nn

    device = device_override or _get_device()
    if random_seed is not None:
        _set_global_seed(int(random_seed))

    data = train_vals.astype(float)
    if len(data) > max_train_points:
        data = data[-max_train_points:]

    val_cut = max(seq_len + 1, int(len(data) * (1 - val_fraction)))
    train_data = data[:val_cut]
    val_data = data[val_cut:]

    norm_train, norm_params = _normalize(train_data, normalization)

    X_tr, y_tr = [], []
    for i in range(len(norm_train) - seq_len):
        X_tr.append(norm_train[i : i + seq_len])
        y_tr.append(norm_train[i + seq_len])
    if not X_tr:
        raise ValueError(f"Not enough training data ({len(norm_train)}) for seq_len={seq_len}.")

    X_tr = np.array(X_tr, dtype=np.float32)
    y_tr = np.array(y_tr, dtype=np.float32)

    try:
        has_val = False
        X_val_t = y_val_t = None
        if len(val_data) > seq_len:
            combined = np.concatenate([train_data[-seq_len:], val_data])
            norm_comb = _apply_norm_params(combined, norm_params)
            X_val, y_val = [], []
            for i in range(len(norm_comb) - seq_len):
                X_val.append(norm_comb[i : i + seq_len])
                y_val.append(norm_comb[i + seq_len])
            X_val_t = torch.from_numpy(np.array(X_val, dtype=np.float32)).unsqueeze(-1).to(device)
            y_val_t = torch.from_numpy(np.array(y_val, dtype=np.float32)).unsqueeze(-1).to(device)
            has_val = True

        X_t = torch.from_numpy(X_tr).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(y_tr).unsqueeze(-1).to(device)

        _NetClass = _MODEL_CLASSES.get(model_type.lower(), _GRUNet)
        model = _NetClass(
            hidden=hidden,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            device=device,
        )

        # Load initial state if provided (federated learning use case)
        if initial_state is not None:
            model.load_state_dict(initial_state)

        _opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        optimiser = _opt_map.get(optimizer_name.lower(), torch.optim.Adam)(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Initialize learning rate scheduler
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        scheduler = get_scheduler(
            lr_scheduler_type, initial_lr=lr, epochs=epochs, **lr_scheduler_kwargs
        )

        _loss_map = {
            "mse": nn.MSELoss(),
            "rmse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(),
        }
        criterion = _loss_map.get(loss_fn_name.lower(), nn.MSELoss())

        train_loss_hist: List[float] = []
        val_loss_hist: List[float] = []

        N = len(X_t)
        for _epoch in range(epochs):
            # Update learning rate at the start of each epoch
            current_lr = scheduler.get_lr()
            for param_group in optimiser.param_groups:
                param_group["lr"] = current_lr

            model.train_mode()
            idx = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                bi = idx[start : start + batch_size]
                xb, yb = X_t[bi], y_t[bi]

                optimiser.zero_grad()
                pred_loss = criterion(model(xb), yb)
                if l1_lambda > 0:
                    l1_reg = l1_lambda * sum(p.abs().sum() for p in model.net.parameters())
                    total_loss = pred_loss + l1_reg
                else:
                    total_loss = pred_loss

                total_loss.backward()
                optimiser.step()
                epoch_loss += pred_loss.item()
                n_batches += 1

            train_loss_hist.append(epoch_loss / max(n_batches, 1))

            model.eval_mode()
            if has_val and X_val_t is not None:
                with torch.no_grad():
                    vl = criterion(model(X_val_t), y_val_t).item()
                val_loss_hist.append(vl)

            # Advance scheduler to next epoch
            scheduler.step()

            if epoch_callback is not None:
                epoch_callback(_epoch, list(train_loss_hist), list(val_loss_hist))

        model.eval_mode()
        return {
            "model": model,
            "norm_params": norm_params,
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "seq_len": seq_len,
            "device": device,
            "model_type": model_type.lower(),
            "seed": int(random_seed) if random_seed is not None else None,
        }
    except Exception as exc:
        if device == "cuda" and device_override is None and _is_gpu_runtime_failure(exc):
            warnings.warn(
                f"GPU training failed at runtime; retrying on CPU. Reason: {exc}",
                RuntimeWarning,
            )
            return train_gru(
                train_vals=train_vals,
                seq_len=seq_len,
                hidden=hidden,
                num_layers=num_layers,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                max_train_points=max_train_points,
                normalization=normalization,
                val_fraction=val_fraction,
                activation=activation,
                dropout=dropout,
                optimizer_name=optimizer_name,
                loss_fn_name=loss_fn_name,
                weight_decay=weight_decay,
                l1_lambda=l1_lambda,
                model_type=model_type,
                initial_state=initial_state,
                device_override="cpu",
                random_seed=random_seed,
                epoch_callback=epoch_callback,
                lr_scheduler_type=lr_scheduler_type,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
            )
        raise
