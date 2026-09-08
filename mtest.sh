torchrun --nproc_per_node=2 training/test.py \
  --detector_path ./training/config/detector/gend.yaml \
  --test_dataset "Celeb-DF-v2" \
  --weights_path /kaggle/working/DeepfakeBench/training/pretrained/Weight/genD.pth \
  --ddp
