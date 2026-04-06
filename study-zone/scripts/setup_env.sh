#!/usr/bin/env bash
# setup_env.sh — Setup môi trường shortcut-models bằng venv (thay thế conda)
# Chạy từ thư mục gốc repo: bash study-zone/scripts/setup_env.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$REPO_ROOT/study-zone/.venv"

echo "==> Tạo virtualenv tại $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrade pip"
pip install --upgrade pip

echo "==> Cài JAX (CPU) — đổi thành jax[cuda12] nếu dùng CUDA GPU"
# macOS CPU:
pip install "jax[cpu]"
# macOS Apple Silicon (Metal):
# pip install jax-metal

echo "==> Cài requirements.txt"
pip install \
    numpy \
    scipy==1.12.0 \
    matplotlib \
    tqdm \
    wandb \
    imageio \
    ml-collections \
    einops \
    jaxtyping \
    "flax==0.7.4" \
    "optax==0.1.7" \
    "orbax==0.1.9" \
    "chex==0.1.82" \
    tensorflow-cpu \
    "tensorflow-probability==0.22.0" \
    tensorflow-datasets \
    diffusers \
    tabulate \
    opt-einsum \
    absl-py \
    termcolor

echo ""
echo "==> DONE. Activate với:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "==> Kiểm tra JAX:"
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
