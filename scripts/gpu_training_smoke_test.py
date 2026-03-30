from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.training.trainer import train_gru


def _torch_runtime_info() -> str:
    try:
        import torch

        return (
            f"torch={torch.__version__}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"hip={getattr(torch.version, 'hip', None)}, "
            f"cuda={getattr(torch.version, 'cuda', None)}"
        )
    except Exception as exc:
        return f"torch runtime unavailable: {exc}"


def main() -> int:
    print("[smoke] Runtime:", _torch_runtime_info())

    # Small synthetic signal to keep training fast and deterministic.
    base = np.linspace(0.0, 4.0 * np.pi, 256, dtype=np.float32)
    train_vals = np.sin(base).astype(np.float32)

    try:
        result = train_gru(
            train_vals=train_vals,
            seq_len=16,
            hidden=8,
            num_layers=1,
            epochs=1,
            batch_size=8,
            max_train_points=220,
            normalization="minmax",
            dropout=0.1,
            model_type="gru",
        )
    except Exception as exc:
        print(f"[smoke] FAIL: training crashed: {exc}")
        return 1

    train_loss = result.get("train_loss", [])
    device = result.get("device", "unknown")

    if device != "cuda":
        print(f"[smoke] FAIL: expected device=cuda but got device={device}")
        print("[smoke] Hint: set FORECAST_USE_GPU=1 before running this test.")
        print(
            "[smoke] Hint: on ROCm cards that need it, also set HSA_OVERRIDE_GFX_VERSION=10.3.0."
        )
        return 2

    print(f"[smoke] PASS: training completed on device={device}")
    print(f"[smoke] train_loss_points={len(train_loss)}")
    if train_loss:
        print(f"[smoke] final_train_loss={train_loss[-1]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
