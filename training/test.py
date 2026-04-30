"""
DeepFakeBench DDP test.py (Multi-GPU Evaluation)
✅ Hỗ trợ DDP tương tự train.py - Sửa lỗi Duplicate GPU
✅ Tự động gom dữ liệu từ nhiều GPU để tính AUC/EER chính xác
✅ Xuất file .pkl cho t-SNE (Chỉ thực hiện tại Rank 0)
"""

import os
import sys
import yaml
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- PATH FIX (repo root) ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from metrics.utils import get_test_metrics
from detectors import DETECTOR

try:
    from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from training.dataset.lrl_dataset import LRLDataset
except Exception:
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    try:
        from dataset.lrl_dataset import LRLDataset
    except Exception:
        LRLDataset = None

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="DeepFakeBench DDP test + optional feature dump")
parser.add_argument("--detector_path", type=str, required=True, help="path to detector YAML file")
parser.add_argument("--test_dataset", nargs="+", default=None, help="list of test dataset names")
parser.add_argument("--weights_path", type=str, required=True, help="path to pretrained weights .pth")
parser.add_argument("--save_feat", action="store_true", default=False, help="save feature pkl for TSNE")
parser.add_argument("--feat_out_dir", type=str, default="tsne_pkls", help="output directory for tsne_dict_*.pkl")
parser.add_argument("--max_samples", type=int, default=None, help="optional cap on number of samples")
parser.add_argument("--ddp", action='store_true', default=False, help="Enable Distributed Data Parallel")
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

args = parser.parse_args()

# --- KHẮC PHỤC LỖI DUPLICATE GPU ---
if args.ddp:
    # Lấy LOCAL_RANK từ môi trường (do torchrun cấp)
    current_local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(current_local_rank)
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    device = torch.device("cuda", current_local_rank)
    is_main_process = (dist.get_rank() == 0)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_local_rank = 0
    is_main_process = True

def init_seed(config):
    seed = config.get("manualSeed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_testing_data(config):
    def get_test_data_loader(cfg, test_name):
        cfg = cfg.copy()
        cfg["test_dataset"] = test_name
        if cfg.get("dataset_type", None) == "lrl" and LRLDataset is not None:
            test_set = LRLDataset(config=cfg, mode="test")
        else:
            test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

        sampler = DistributedSampler(test_set, shuffle=False) if args.ddp else None
        
        dl = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=int(cfg["test_batchSize"]),
            shuffle=False,
            num_workers=int(cfg["workers"]),
            collate_fn=test_set.collate_fn,
            sampler=sampler,
            drop_last=False,
        )
        return dl

    test_data_loaders = {}
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders

@torch.no_grad()
def test_one_dataset(model, data_loader, max_samples=None):
    model.eval()
    pred_list, label_list, feat_list, label_spe_list = [], [], [], []
    n = 0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=not is_main_process)
    
    for _, data_dict in pbar:
        label_bin = torch.where(data_dict["label"] != 0, 1, 0).to(device)
        
        for k in ["image", "label"]:
            if k in data_dict and torch.is_tensor(data_dict[k]):
                data_dict[k] = data_dict[k].to(device)
        
        if data_dict["image"].ndim == 5:
            data_dict["image"] = data_dict["image"][:, 0]

        outputs = model(data_dict, inference=True)
        
        pred_list.append(outputs["prob"])
        label_list.append(label_bin)
        feat_list.append(outputs["feat"])
        
        if "label_spe" in data_dict and data_dict["label_spe"] is not None:
            label_spe_list.append(data_dict["label_spe"].to(device))
        else:
            label_spe_list.append(label_bin)

        n += outputs["feat"].shape[0]
        if max_samples is not None and n >= max_samples:
            break

    # Thu thập kết quả cục bộ tại mỗi GPU
    local_preds = torch.cat(pred_list, dim=0)
    local_labels = torch.cat(label_list, dim=0)
    local_feats = torch.cat(feat_list, dim=0)
    local_spe = torch.cat(label_spe_list, dim=0)

    if args.ddp:
        # Gom kết quả từ tất cả GPU về mọi rank (all_gather)
        world_size = dist.get_world_size()
        
        def gather_tensor(tensor):
            gathered_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_list, tensor)
            return torch.cat(gathered_list, dim=0)

        all_preds = gather_tensor(local_preds).cpu().numpy()
        all_labels = gather_tensor(local_labels).cpu().numpy()
        all_feats = gather_tensor(local_feats).cpu().numpy()
        all_spe = gather_tensor(local_spe).cpu().numpy()
    else:
        all_preds = local_preds.cpu().numpy()
        all_labels = local_labels.cpu().numpy()
        all_feats = local_feats.cpu().numpy()
        all_spe = local_spe.cpu().numpy()

    return all_preds, all_labels, all_feats, all_spe

def main():
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)

    test_cfg_path = os.path.join(ROOT, "training", "config", "test_config.yaml")
    if os.path.exists(test_cfg_path):
        with open(test_cfg_path, "r") as f:
            config.update(yaml.safe_load(f))

    if args.test_dataset:
        config["test_dataset"] = args.test_dataset

    init_seed(config)
    if config.get("cudnn", True):
        cudnn.benchmark = True

    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)

    if is_main_process:
        print(f"===> Loading weights from: {args.weights_path}")
    
    ckpt = torch.load(args.weights_path, map_location="cpu")
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    if args.ddp:
        model = DDP(model, device_ids=[current_local_rank], output_device=current_local_rank, find_unused_parameters=True)

    test_loaders = prepare_testing_data(config)
    
    for dataset_name, loader in test_loaders.items():
        if is_main_process:
            print(f"\n===== Testing on: {dataset_name} =====")
        
        pred_prob, label_bin, feat, label_spe = test_one_dataset(model, loader, args.max_samples)

        # Chỉ xử lý kết quả tại Rank 0 (Main Process)
        if is_main_process:
            metrics = get_test_metrics(y_pred=pred_prob, y_true=label_bin, img_names=None)
            for k, v in metrics.items():
                if k not in ["pred", "label", "dataset_dict"]:
                    print(f"{k}: {v:.4f}")

            if args.save_feat:
                os.makedirs(args.feat_out_dir, exist_ok=True)
                safe_name = dataset_name.replace("/", "_")
                out_path = os.path.join(args.feat_out_dir, f"tsne_{config['model_name']}_{safe_name}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump({
                        "feat": feat,
                        "label": label_bin,
                        "pred": pred_prob,
                        "label_spe": label_spe
                    }, f)
                print(f"✅ Features saved to {out_path}")

    if args.ddp:
        dist.barrier() # Đảm bảo tất cả rank xong việc trước khi đóng
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
