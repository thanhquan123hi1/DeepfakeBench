python training/test.py \
--detector_path ./training/config/detector/gend_effort.yaml  \
--test_dataset  "Celeb-DF-v2" "UADFV" "DFDCP" \
--weights_path /kaggle/input/datasets/xuanhuydinh/deepfakebench/Weight/model_gend.pth \
--save_feat \
--feat_out_dir /kaggle/tmp/tsne_pkls
