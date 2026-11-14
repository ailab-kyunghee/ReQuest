#!/usr/bin/env bash
# Extract the per-question BLIP segment features used to train the selector.
#
# Usage: bash scripts/extract_features.sh [NUM_SHARDS]
#
#   NUM_SHARDS  Processes to run in parallel (default 1), spread over the GPUs in
#               CUDA_VISIBLE_DEVICES (default: all). Each needs about 4 CPU cores.
#
# Environment overrides:
#   DATA      Pseudo-label dataset (default: data/pseudo_labels)
#   VIDEOS    LLaVA-Video-178K root (default: data/LLaVA-Video-178K)
#   FEATURES  Output directory (default: data/features)
#
# Finished questions are skipped, so run the same command again to resume.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NUM_SHARDS=${1:-1}
if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be a positive integer, got: $NUM_SHARDS" >&2
  exit 1
fi
DATA=${DATA:-$ROOT/data/pseudo_labels}
VIDEOS=${VIDEOS:-$ROOT/data/LLaVA-Video-178K}
FEATURES=${FEATURES:-$ROOT/data/features}

GPU_LIST=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
IFS=, read -ra GPUS <<< "$GPU_LIST"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

pids=()
for (( shard = 0; shard < NUM_SHARDS; shard++ )); do
  CUDA_VISIBLE_DEVICES=${GPUS[shard % ${#GPUS[@]}]} python "$ROOT/scripts/extract_blip_features.py" \
    --data_dir "$DATA" --video_root "$VIDEOS" --output_dir "$FEATURES" \
    --num_shards "$NUM_SHARDS" --shard "$shard" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
