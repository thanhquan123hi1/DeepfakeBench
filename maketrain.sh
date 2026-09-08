#!/bin/bash
# ==============================================================================
# maketrain.sh: Easy Training Launcher for CLIP Bias PEFT Deepfake Detection
# ==============================================================================

set -e

# Setup environment
export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Default parameters
STRATEGY="all_bias"
GPUS=2
CONFIG="./training/config/detector/clip_bias.yaml"
TRAIN_DATA="FaceForensics++"
TEST_DATA="Celeb-DF-v2"
LAYER_RANGE=""
WEIGHTS=""

usage() {
  cat << EOF
================================================================================
  maketrain.sh - Simple CLI Training Launcher
================================================================================
Usage:
  ./maketrain.sh [STRATEGY] [GPUS] [OPTIONS]

Examples:
  ./maketrain.sh                          # Train 'all_bias' on 2 GPUs (DDP)
  ./maketrain.sh v_bias                   # Train 'v_bias' on 2 GPUs (DDP)
  ./maketrain.sh v_bias 1                 # Train 'v_bias' on 1 GPU
  ./maketrain.sh full 2                   # Full Fine-Tuning (100% backbone) on 2 GPUs
  ./maketrain.sh --strategy v_bias --layer-range late --gpus 2
  ./maketrain.sh --strategy ln_bias --train-data FaceForensics++ --test-data Celeb-DF-v2

Available Strategies:
  full, frozen, all_bias, linear_bias, ln_bias, attention_bias, qkv_bias,
  q_bias, k_bias, v_bias, attn_proj_bias, mlp_bias, mlp_fc1_bias, mlp_fc2_bias,
  bias_early, bias_middle, bias_late, or combined (e.g. v_bias+late)

Options:
  -s, --strategy STRATEGY   Bias tuning strategy (default: all_bias)
  -g, --gpus NUM            Number of GPUs to use (default: 2, set 1 for single-GPU)
  -l, --layer-range RANGE   Layer range (e.g., all, early, middle, late, or 16-23)
  -c, --config PATH         Detector config YAML path (default: clip_bias.yaml)
  -t, --train-data NAME     Training dataset (default: FaceForensics++)
  -e, --test-data NAME      Testing dataset (default: Celeb-DF-v2)
  -w, --weights PATH        Path to pretrained weights to initialize from
  -h, --help                Show this help message
================================================================================
EOF
  exit 0
}

# Handle positional arguments first if passed without flags
if [[ "$1" =~ ^[a-zA-Z0-9_+-]+$ ]] && [[ "$1" != -* ]]; then
  STRATEGY="$1"
  shift
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    GPUS="$1"
    shift
  fi
fi

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--strategy)
      STRATEGY="$2"
      shift 2
      ;;
    -g|--gpus)
      GPUS="$2"
      shift 2
      ;;
    -l|--layer-range)
      LAYER_RANGE="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG="$2"
      shift 2
      ;;
    -t|--train-data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    -e|--test-data)
      TEST_DATA="$2"
      shift 2
      ;;
    -w|--weights)
      WEIGHTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

# Build extra arguments
EXTRA_ARGS=()
if [ -n "$LAYER_RANGE" ]; then
  EXTRA_ARGS+=(--layer_range "$LAYER_RANGE")
fi
if [ -n "$WEIGHTS" ]; then
  EXTRA_ARGS+=(--weights_path "$WEIGHTS")
fi

echo "================================================================================"
echo "  LAUNCHING TRAINING JOB"
echo "  Detector Config : $CONFIG"
echo "  Tuning Strategy : $STRATEGY"
echo "  Layer Range     : ${LAYER_RANGE:-all}"
echo "  Train Dataset   : $TRAIN_DATA"
echo "  Test Dataset    : $TEST_DATA"
echo "  GPU Devices     : $GPUS"
echo "================================================================================"

if [ "$GPUS" -gt 1 ]; then
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
  torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" training/train.py \
    --detector_path "$CONFIG" \
    --train_dataset "$TRAIN_DATA" \
    --test_dataset "$TEST_DATA" \
    --tuning "$STRATEGY" \
    --ddp \
    "${EXTRA_ARGS[@]}"
else
  python3 training/train.py \
    --detector_path "$CONFIG" \
    --train_dataset "$TRAIN_DATA" \
    --test_dataset "$TEST_DATA" \
    --tuning "$STRATEGY" \
    "${EXTRA_ARGS[@]}"
fi
