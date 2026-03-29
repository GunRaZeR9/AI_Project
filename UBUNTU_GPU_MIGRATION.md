# Ubuntu Native GPU Training Migration (ROCm)

This project can run on Windows with CPU fallback, but reliable AMD GPU training is expected on native Ubuntu with ROCm.

## Target Platform

- Preferred: native Ubuntu install (dual-boot or dedicated disk)
- Recommended ROCm family: 7.2 (match your PyTorch wheel)
- Use an AMD-supported GPU/CPU + Ubuntu version pair from AMD's ROCm compatibility docs

## 1) Install Ubuntu and Base Tools

```bash
sudo apt update
sudo apt install -y git curl wget build-essential python3 python3-venv python3-pip
```

## 2) Install ROCm (native Ubuntu)

Follow AMD's official ROCm on Radeon/Ryzen instructions for your exact distro.
After install, verify:

```bash
/opt/rocm/bin/rocminfo | head -n 40
/opt/rocm/bin/rocm-smi
```

If these commands fail, do not continue to PyTorch setup yet.

## 3) Create Python Environment

```bash
cd /path/to/AI_Project
python3 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4) Install Project Dependencies

```bash
pip install -r project_requirements_ubuntu_rocm.txt
```

Then install a ROCm-compatible PyTorch build that matches your ROCm release.
Use the official PyTorch selector for ROCm and install exactly one torch/torchvision/torchaudio set.

## 5) Run ROCm Preflight

```bash
python scripts/ubuntu_rocm_preflight.py
```

Expected result: torch loads, ROCm/HIP metadata is present, and a tiny GPU tensor/dropout check passes.

## 6) Run Training Smoke Test

```bash
python scripts/gpu_training_smoke_test.py
```

Expected result: `PASS` and `device=cuda` (HIP backend uses the `cuda` device string in PyTorch).

## 7) Run the App

```bash
streamlit run app.py
```

## Notes

- If preflight fails with MIOpen/HIP compile errors, verify ROCm install, Linux headers, and wheel compatibility first.
- This repository already includes a CPU fallback path so work can continue while GPU stack issues are fixed.
- Keep Windows for editing/UI if preferred, and run heavy training jobs in Ubuntu.
