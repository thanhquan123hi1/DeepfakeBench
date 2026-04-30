# Chạy trên 2 GPU
torchrun --nproc_per_node=2 training/test.py \
  --ddp \
  --detector_path training/config/detector/gend.yaml \
  --test_dataset Celeb-DF-v2\
  --weights_path /kaggle/input/datasets/xuanhuydinh/deepfakebench/Weight/model_gend.pth\
  --save_feat


