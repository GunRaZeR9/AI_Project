from __future__ import annotations

import os
import platform
import sys


def _print_line(label: str, value: str) -> None:
    print(f"[preflight] {label}: {value}")


def main() -> int:
    _print_line("platform", platform.platform())
    _print_line("python", sys.version.split()[0])

    if platform.system().lower() != "linux":
        _print_line("warning", "This script is intended for Ubuntu/Linux ROCm environments")

    _print_line("ROCM_HOME", os.environ.get("ROCM_HOME", "<unset>"))
    _print_line("HIP_PATH", os.environ.get("HIP_PATH", "<unset>"))

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        _print_line("FAIL", f"PyTorch import failed: {exc}")
        return 1

    _print_line("torch", torch.__version__)
    _print_line("torch.version.hip", str(getattr(torch.version, "hip", None)))
    _print_line("torch.version.cuda", str(getattr(torch.version, "cuda", None)))

    cuda_available = torch.cuda.is_available()
    _print_line("torch.cuda.is_available", str(cuda_available))
    if not cuda_available:
        _print_line("FAIL", "No GPU backend visible to torch")
        return 1

    try:
        name = torch.cuda.get_device_name(0)
        _print_line("device[0]", name)
    except Exception as exc:
        _print_line("warning", f"Unable to query device name: {exc}")

    try:
        x = torch.randn(8, 1, device="cuda")
        y = nn.Dropout(0.1).to("cuda")(x)
        _ = float(y.mean().item())
        torch.cuda.synchronize()
    except Exception as exc:
        _print_line("FAIL", f"GPU runtime check failed: {exc}")
        return 1

    _print_line("PASS", "ROCm/HIP runtime check succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
