#!/usr/bin/env bash
# run_inference.sh — Chạy inference và lưu kết quả vào study-zone/
# Usage: bash study-zone/scripts/run_inference.sh <checkpoint_path> [num_steps]
#
# Ví dụ: bash study-zone/scripts/run_inference.sh study-zone/checkpoints/celeba_dit_b 1

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO_ROOT/study-zone/.venv/bin/activate"

LOAD_DIR="${1:?Cần truyền checkpoint path, VD: study-zone/checkpoints/celeba_dit_b}"
NUM_STEPS="${2:-1}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$REPO_ROOT/study-zone/samples/inference_${NUM_STEPS}step_${TIMESTAMP}"

mkdir -p "$OUT_DIR"

# Activate venv nếu có
[ -f "$VENV" ] && source "$VENV"

echo "==> Running inference: steps=$NUM_STEPS, ckpt=$LOAD_DIR"
echo "==> Output dir: $OUT_DIR"

cd "$REPO_ROOT"
# Note: không có sample.py — inference chạy qua train.py --mode inference
python train.py \
    --load_dir "$LOAD_DIR" \
    --save_dir "$OUT_DIR" \
    --mode inference \
    --model.train_type shortcut \
    --model.hidden_size 768 \
    --model.patch_size 2 \
    --model.depth 12 \
    --model.num_heads 12 \
    --model.mlp_ratio 4 \
    --dataset_name celebahq256 \
    --fid_stats data/celeba256_fidstats_ours.npz \
    --model.denoise_timesteps "$NUM_STEPS" \
    --model.cfg_scale 0 \
    --model.class_dropout_prob 1 \
    --model.num_classes 1 \
    2>&1 | tee "$REPO_ROOT/study-zone/logs/inference_${NUM_STEPS}step_${TIMESTAMP}.log"

echo "==> Done. Samples saved to $OUT_DIR"
