#!/usr/bin/env bash
# Evaluate LLaVA-Video + ReQuest on one benchmark.
#
# Usage: bash scripts/eval.sh {videomme|longvideobench|mlvu} [NUM_GPUS] [LMMS_EVAL_ARGS...]
#
#   NUM_GPUS        Number of GPUs (default 1). Choose them with CUDA_VISIBLE_DEVICES.
#   LMMS_EVAL_ARGS  Extra lmms-eval arguments, e.g. --limit 20.
#
# Environment overrides:
#   SELECTOR  Selector checkpoint (default: checkpoints/request_selector_<benchmark>.pth)
#   OUTPUT    Output directory (default: outputs/<benchmark>)
#   PORT      Port used between GPU processes (default: 29500)
#   EXTRA_MODEL_ARGS  Extra model arguments, e.g. decode_workers=8
set -euo pipefail

BENCHMARK=${1:?"benchmark: videomme | longvideobench | mlvu"}
NUM_GPUS=${2:-1}
shift $(( $# < 2 ? $# : 2 ))

# Per-benchmark base routing threshold tau0; gamma_len = 0.3 for all.
case "$BENCHMARK" in
  videomme)       TASK=videomme_request;     TAU0=0.55 ;;
  longvideobench) TASK=longvideobench_val_v; TAU0=0.30 ;;
  mlvu)           TASK=mlvu_request;         TAU0=0.45 ;;
  *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELECTOR=${SELECTOR:-$ROOT/checkpoints/request_selector_$BENCHMARK.pth}
if [[ ! -f "$SELECTOR" ]]; then
  echo "Selector checkpoint not found: $SELECTOR" >&2
  echo "Download it from the GitHub Releases page (see README)." >&2
  exit 1
fi

OUTPUT=${OUTPUT:-outputs/$BENCHMARK}
MODEL_ARGS="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5"
MODEL_ARGS+=",max_frames_num=32,selector_path=$SELECTOR,tau0=$TAU0,gamma_len=0.3"
MODEL_ARGS+=",log_dir=$OUTPUT/request_log${EXTRA_MODEL_ARGS:+,$EXTRA_MODEL_ARGS}"

# Each GPU process holds a full copy of the models and answers a share of the
# questions.
LAUNCH=(accelerate launch --num_processes "$NUM_GPUS" --num_machines 1
        --mixed_precision no --dynamo_backend no)
if (( NUM_GPUS > 1 )); then
  LAUNCH+=(--multi_gpu --main_process_port "${PORT:-29500}")
fi

# The plugin registers the model. Its task configs are also passed with
# --include_path, because lmms-eval 0.4 builds the task list before it adds
# plugin task folders.
export LMMS_EVAL_PLUGINS=request_vqa
"${LAUNCH[@]}" -m lmms_eval \
  --model llava_vid_request \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --include_path "$ROOT/request_vqa/tasks" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT" \
  "$@"
