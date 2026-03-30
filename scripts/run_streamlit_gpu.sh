#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv-linux/bin/activate" ]]; then
  echo "Error: .venv-linux is missing. Create it first with: python3 -m venv .venv-linux"
  exit 1
fi

source .venv-linux/bin/activate

if ! command -v streamlit >/dev/null 2>&1; then
  echo "Error: streamlit not found in .venv-linux. Install dependencies first."
  exit 1
fi

# Explicit ROCm + GPU opt-in for this project.
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm/hip}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export FORECAST_DEVICE="${FORECAST_DEVICE:-cuda}"
export FORECAST_USE_GPU=1
export FORECAST_ROCM_DISABLE_DROPOUT="${FORECAST_ROCM_DISABLE_DROPOUT:-1}"
export FORECAST_ALLOW_CPU_FALLBACK="${FORECAST_ALLOW_CPU_FALLBACK:-0}"

exec streamlit run app.py "$@"