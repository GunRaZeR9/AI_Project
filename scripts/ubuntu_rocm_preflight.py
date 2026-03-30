from __future__ import annotations

import os
import platform
import subprocess
import sys


def _print_line(label: str, value: str) -> None:
    print(f"[preflight] {label}: {value}")


def _group_names() -> list[str]:
    try:
        import grp

        return sorted({grp.getgrgid(gid).gr_name for gid in os.getgroups()})
    except Exception:
        return []


def _detect_gfx_arches() -> list[str]:
    for cmd in (("/opt/rocm/bin/rocminfo",), ("rocminfo",)):
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                continue
            arches: set[str] = set()
            for line in proc.stdout.splitlines():
                if "amdgcn-amd-amdhsa--gfx" in line:
                    arch = line.split("gfx", 1)[-1].strip()
                    if arch:
                        arches.add(f"gfx{arch.split()[0]}")
            return sorted(arches)
        except Exception:
            continue
    return []


def main() -> int:
    _print_line("platform", platform.platform())
    _print_line("python", sys.version.split()[0])

    if platform.system().lower() != "linux":
        _print_line("warning", "This script is intended for Ubuntu/Linux ROCm environments")

    _print_line("ROCM_HOME", os.environ.get("ROCM_HOME", "<unset>"))
    _print_line("HIP_PATH", os.environ.get("HIP_PATH", "<unset>"))
    groups = _group_names()
    _print_line("groups", ",".join(groups) if groups else "<unknown>")

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
        if groups and "render" not in groups:
            _print_line("hint", "Current user is not in render group; run: sudo usermod -aG render,video $USER then relogin")
        _print_line("FAIL", "No GPU backend visible to torch")
        return 1

    arches = _detect_gfx_arches()
    if arches:
        _print_line("rocminfo.gfx", ",".join(arches))

    # Avoid a known hard crash on some gfx1036 systems when override is missing.
    if "gfx1036" in arches and not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        _print_line("hint", "Detected gfx1036. Set HSA_OVERRIDE_GFX_VERSION=10.3.0 before running PyTorch GPU kernels")
        _print_line("FAIL", "Missing HSA_OVERRIDE_GFX_VERSION for gfx1036 compatibility")
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
