#!/usr/bin/env bash
# Train the question-aware selector on the pseudo-label dataset.
#
# Usage: bash scripts/train.sh [NUM_GPUS] [TRAIN_ARGS...]
#
#   NUM_GPUS    Number of GPUs (default 1). Choose them with CUDA_VISIBLE_DEVICES.
#   TRAIN_ARGS  Extra arguments of scripts/train_selector.py, e.g. --epochs 2.
#
# Environment overrides:
#   DATA      Pseudo-label dataset (default: data/pseudo_labels)
#   FEATURES  BLIP features (default: data/features)
#   OUTPUT    Run directory (default: runs/selector)
#
# The run continues from OUTPUT/last.pt if it exists, so run the same command
# again to resume. Use a new OUTPUT for a fresh run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NUM_GPUS=${1:-1}
shift $(( $# < 1 ? $# : 1 ))
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi
DATA=${DATA:-$ROOT/data/pseudo_labels}
FEATURES=${FEATURES:-$ROOT/data/features}
OUTPUT=${OUTPUT:-$ROOT/runs/selector}

ARGS=(--data_dir "$DATA" --features_dir "$FEATURES" --output_dir "$OUTPUT" --resume "$@")
if (( NUM_GPUS > 1 )); then
  torchrun --nproc_per_node "$NUM_GPUS" "$ROOT/scripts/train_selector.py" "${ARGS[@]}"
else
  python "$ROOT/scripts/train_selector.py" "${ARGS[@]}"
fi
