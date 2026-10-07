#!/usr/bin/env bash
# Download both checkpoints (~64 GB). Resumable: rerun after an interruption.
#   CKPT_DIR=/big/disk/checkpoints bash scripts/download.sh     # default: ./checkpoints
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${CKPT_DIR:-checkpoints}"
mkdir -p "$CKPT"
[ "$CKPT" = checkpoints ] || [ -e checkpoints ] || ln -s "$CKPT" checkpoints
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir "$CKPT/bonsai2-27b-mlx"   #  8.6 GB
hf download Qwen/Qwen3.8-27B                       --local-dir "$CKPT/qwen3.8-27b"       # 55.6 GB
python tools/verify_download.py "$CKPT/bonsai2-27b-mlx" "$CKPT/qwen3.8-27b" --sha
