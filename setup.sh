#!/usr/bin/env bash
# setup.sh -- install framegen-rocm on AMD ROCm (Linux).
#
# Usage:
#   bash setup.sh              # detect ROCm, build venv, install everything
#   bash setup.sh --no-venv    # use the current environment (no venv)
#
# It auto-detects your ROCm version (via amd-smi) and installs the matching
# PyTorch ROCm wheel from the correct index.

set -euo pipefail

make_venv=1
[[ "$${1:-}" == "--no-venv" ]] && make_venv=0
PY="${PY:-python3}"
VENV=".venv"

# --- 1. detect ROCm ---
ROCM_VER=""
if command -v amd-smi >/dev/null 2>&1; then
  ROCM_VER=$(amd-smi version 2>/dev/null | grep -i 'ROCm' | sed -E 's/.*[Vv]ersion:[[:space:]]*//;s/^[^0-9.]*//' | cut -d. -f1,2 || true)
fi
ROCM_VER="${ROCM_VER:-""}"
echo "Detected ROCm: ${ROCM_VER:-<none found>}" | cat
if [ -z "$ROCM_VER" ]; then
  echo "  -> warning: no ROCm detected (amd-smi). Install ROCm or set ROCM_VER manually."
fi

# --- 2. venv ---
if [ "$make_venv" == 1 ]; then
  if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
  else
    "$PY" -m venv "$VENV"
    source "$VENV/bin/activate"
  fi
fi
python -m pip install -U pip

# --- 3. pick the ROCm index tag ---
case "$ROCM_VER" in
  "7.2")  IDX="rocm7.2";;
  "7.1")  IDX="rocm7.1";;
  "7.0")  IDX="rocm7.0";;
  "6.4")  IDX="rocm6.4";;
  "6.2")  IDX="rocm6.2";;
  "6.1")  IDX="rocm6.1";;
  "6.0")  IDX="rocm6.0";;
  "")     IDX="rocm6.4";;   # best-effort default
  *)      IDX="rocm${ROCM_VER}";;
esac
echo "PyTorch index: https://download.pytorch.org/whl/${IDX}"

# --- 4. install torch (ROCm) ---
echo "Installing PyTorch (ROCm) ..."
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${IDX}"

# --- 5. install the rest ---
echo "Installing base deps ..."
python -m pip install -e .

# --- 6. verify ---
echo
echo "=== verify ==="
python - <<'PY'
import torch
print("torch    :", torch.__version__)
print("rocm     :", getattr(torch, "version", {}).rocm)
print("cuda_api :", getattr(torch, "version", {}).cuda)
print("available:", torch.cuda.is_available())
try:
    print("gpu      :", torch.cuda.get_device_name(0))
except Exception as e:
    print("gpu      :", "N/A (", e, ")")
PY
echo
echo "All done. Try:"
echo "  python -m framegen.cli --list-backends"
echo "  bash make_demo.sh && python -m framegen.cli demo.mp4 --mult 2"
