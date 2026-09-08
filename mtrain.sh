#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

TUNING_STRATEGY="${1:-all_bias}"

echo "=========================================================="
echo "Starting 2-GPU DDP training CLIP Bias PEFT with strategy: $TUNING_STRATEGY"
echo "=========================================================="

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=2 --master_port="$MASTER_PORT" training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2" \
  --tuning "$TUNING_STRATEGY" \
  --ddp
