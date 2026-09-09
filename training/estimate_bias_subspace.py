#!/usr/bin/env python3
# ==============================================================================
# estimate_bias_subspace.py: Phase A - Estimate Shared Forensic Bias Subspace
# ==============================================================================
# Estimates a low-dimensional subspace U from gradients of multiple manipulation
# methods in FaceForensics++ (FF-DF, FF-F2F, FF-FS, FF-NT).
# ==============================================================================

import os
import sys
import argparse
import random
import logging
from typing import List, Dict

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure local modules are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors.clip_bias_detector import CLIPBiasDetector
from detectors.bias_subspace import (
    BiasParameterSpec,
    get_ordered_bias_specs,
    flatten_bias_gradients,
    compute_gradient_cosine_matrix,
    compute_method_projection_energies,
    build_subspace_svd,
    build_subspace_mean,
    SubspaceArtifact,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("estimate_bias_subspace")


def set_seed(seed: int = 1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedBatchSampler:
    """
    Samples exactly 50% Real (label 0) and 50% Fake (label 1) per batch.
    Guarantees a clean balanced binary task for gradient estimation.
    """
    def __init__(self, dataset: DeepfakeAbstractBaseDataset, batch_size: int, num_batches: int, seed: int = 1024):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = random.Random(seed)

        self.real_indices = [i for i, label in enumerate(dataset.label_list) if label == 0]
        self.fake_indices = [i for i, label in enumerate(dataset.label_list) if label == 1]

        if len(self.real_indices) == 0 or len(self.fake_indices) == 0:
            raise ValueError(
                f"Dataset must contain both real and fake samples. "
                f"Found {len(self.real_indices)} real and {len(self.fake_indices)} fake."
            )

        self.half_batch = batch_size // 2

    def __iter__(self):
        for _ in range(self.num_batches):
            real_batch = self.rng.sample(self.real_indices, min(self.half_batch, len(self.real_indices)))
            fake_batch = self.rng.sample(self.fake_indices, min(self.half_batch, len(self.fake_indices)))
            batch = real_batch + fake_batch
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def collect_method_gradient(
    model: CLIPBiasDetector,
    dataset: DeepfakeAbstractBaseDataset,
    method_name: str,
    bias_specs: List[BiasParameterSpec],
    batches_per_method: int,
    batch_size: int,
    device: torch.device,
    seed: int = 1024,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Collects and normalizes the mean gradient for a specific manipulation method over K balanced batches.
    """
    logger.info(f"===> Collecting gradients for method: '{method_name}' ({batches_per_method} balanced batches)")

    sampler = BalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_batches=batches_per_method,
        seed=seed
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        collate_fn=dataset.collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    loss_fn = nn.CrossEntropyLoss()
    accumulated_grad = None
    batch_count = 0

    model.eval()  # Keep batchnorm/dropout deterministic if any

    for step, batch_data in enumerate(dataloader):
        # Move batch data to device
        for key in batch_data:
            if batch_data[key] is not None and key != 'name':
                batch_data[key] = batch_data[key].to(device)

        model.zero_grad()

        # Forward pass
        pred_dict = model(batch_data)
        loss = loss_fn(pred_dict['cls'], batch_data['label'])

        # Backward pass
        loss.backward()

        # Extract only backbone bias gradients in canonical order
        g_k = flatten_bias_gradients(model.backbone, bias_specs, check_none=True)

        # Normalize per-batch gradient
        norm_k = g_k.norm(2) + eps
        g_hat_k = g_k / norm_k

        if accumulated_grad is None:
            accumulated_grad = g_hat_k.detach().clone()
        else:
            accumulated_grad += g_hat_k.detach()

        batch_count += 1
        logger.debug(f"[{method_name}] Batch {batch_count}/{batches_per_method} - Loss: {loss.item():.4f}, ||g||: {norm_k.item():.4f}")

    # Mean over K batches
    mean_grad = accumulated_grad / float(batch_count)

    # Final re-normalization
    mean_norm = mean_grad.norm(2) + eps
    tilde_g = mean_grad / mean_norm

    logger.info(f"[{method_name}] Finished: aggregated {batch_count} batches, unit norm: {tilde_g.norm(2).item():.4f}")
    return tilde_g.cpu()


def main():
    parser = argparse.ArgumentParser(description="Estimate Manipulation-Invariant Bias Subspace (MIBS)")
    parser.add_argument("--detector_path", type=str, default="./training/config/detector/clip_bias.yaml")
    parser.add_argument("--train_config_path", type=str, default="./training/config/train_config.yaml")
    parser.add_argument("--train_dataset", type=str, default="FaceForensics++")
    parser.add_argument("--tuning", type=str, default="all_bias")
    parser.add_argument("--subspace_rank", type=int, default=2)
    parser.add_argument("--subspace_method", type=str, choices=["shared_svd", "mean_direction"], default="shared_svd")
    parser.add_argument("--batches_per_method", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="./bias_subspace_rank2.pt")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    logger.info("=" * 80)
    logger.info("  MANIPULATION-INVARIANT BIAS SUBSPACE (MIBS) - ESTIMATION PIPELINE")
    logger.info(f"  Detector Config : {args.detector_path}")
    parser_info = f"Rank: {args.subspace_rank} | Method: {args.subspace_method} | Batches/Method: {args.batches_per_method} | BatchSize: {args.batch_size}"
    logger.info(f"  Configuration   : {parser_info}")
    logger.info(f"  Output Artifact : {args.output}")
    logger.info(f"  Device          : {device}")
    logger.info("=" * 80)

    # 1. Load configuration
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    if os.path.exists(args.train_config_path):
        with open(args.train_config_path, 'r') as f:
            train_cfg = yaml.safe_load(f)
            config.update(train_cfg)

    config['tuning'] = args.tuning
    config['train_batchSize'] = args.batch_size

    # 2. Build model with all_bias enabled
    logger.info("Initializing CLIP ViT-L/14 model with strategy='all_bias'...")
    model = CLIPBiasDetector(config)
    model.to(device)

    # Canonical specs
    bias_specs = model.bias_specs
    P = bias_specs[-1].end if bias_specs else 0
    logger.info(f"Extracted {len(bias_specs)} bias parameter tensors. Total bias dimension P = {P:,}")

    # 3. Manipulation methods in FaceForensics++
    methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
    logger.info(f"Target manipulation methods ({len(methods)}): {methods}")

    if args.subspace_method == "shared_svd" and args.subspace_rank > len(methods):
        raise ValueError(
            f"Requested rank {args.subspace_rank} exceeds number of manipulation methods {len(methods)}. "
            f"Maximum rank for {len(methods)} methods is {len(methods)}."
        )

    # 4. Collect gradients for each manipulation method
    method_gradients: Dict[str, torch.Tensor] = {}

    for method_name in methods:
        method_config = dict(config)
        method_config['train_dataset'] = [method_name]

        dataset = DeepfakeAbstractBaseDataset(config=method_config, mode='train')
        logger.info(f"Loaded dataset for {method_name}: {len(dataset)} samples (Real + {method_name})")

        g_tilde = collect_method_gradient(
            model=model,
            dataset=dataset,
            method_name=method_name,
            bias_specs=bias_specs,
            batches_per_method=args.batches_per_method,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed
        )
        method_gradients[method_name] = g_tilde

    # 5. Gradient agreement diagnostics (Cosine Similarity)
    cosine_matrix, method_order = compute_gradient_cosine_matrix(method_gradients)

    print("\n" + "=" * 60)
    print("  GRADIENT COSINE SIMILARITY MATRIX")
    print("=" * 60)
    header = f"{'Method':<10}" + "".join([f"{m:>10}" for m in method_order])
    print(header)
    print("-" * len(header))
    for i, m1 in enumerate(method_order):
        row = f"{m1:<10}" + "".join([f"{cosine_matrix[i, j].item():>10.4f}" for j in range(len(method_order))])
        print(row)
    print("=" * 60 + "\n")

    # 6. Subspace construction
    if args.subspace_method == "shared_svd":
        U_shared, singular_values, explained_energy = build_subspace_svd(
            method_gradients,
            rank=args.subspace_rank
        )
        rank_to_save = args.subspace_rank
    else:  # mean_direction
        U_shared, singular_values, explained_energy = build_subspace_mean(method_gradients)
        rank_to_save = 1

    print("=" * 60)
    print("  SUBSPACE SPECTRAL DIAGNOSTICS")
    print("=" * 60)
    for i, s_val in enumerate(singular_values):
        retained = " (Retained in U)" if i < rank_to_save else ""
        print(f"  σ{i+1} = {s_val.item():.6f}{retained}")
    print(f"\n  Total Retained Rank : {rank_to_save}")
    print(f"  Explained Energy    : {explained_energy * 100:.2f} %")
    print(f"  U Matrix Shape      : [{U_shared.shape[0]}, {U_shared.shape[1]}]")

    # Verify orthonormality
    gram = U_shared.t() @ U_shared
    eye = torch.eye(rank_to_save, dtype=U_shared.dtype)
    max_ortho_err = (gram - eye).abs().max().item()
    print(f"  Orthonormality Err  : {max_ortho_err:.2e} (max |U.T @ U - I|)")
    print("=" * 60 + "\n")

    # 7. Method projection energy diagnostics: R_m = ||U^T g_m||^2 / ||g_m||^2
    projection_energies = compute_method_projection_energies(U_shared, method_gradients)
    print("=" * 60)
    print(f"  PROJECTION ENERGY ONTO U (rank={rank_to_save})")
    print("=" * 60)
    for m in method_order:
        r_val = projection_energies.get(m, 0.0)
        print(f"  {m:<10}: {r_val:>5.1f} %")
    print("=" * 60 + "\n")

    # 8. Save artifact
    artifact = SubspaceArtifact(
        U=U_shared.cpu(),
        rank=rank_to_save,
        subspace_method=args.subspace_method,
        backbone_name=getattr(model.backbone.config, 'model_type', 'clip_vit'),
        tuning_strategy="all_bias",
        parameter_specs=[s.to_dict() for s in bias_specs],
        total_bias_params=P,
        methods=method_order,
        gradient_cosine_matrix=cosine_matrix.cpu(),
        singular_values=singular_values.cpu(),
        explained_energy=explained_energy,
        batches_per_method=args.batches_per_method,
        seed=args.seed,
        mean_gradients=method_gradients,
        projection_energies=projection_energies
    )

    artifact.save(args.output)
    logger.info(f"Subspace artifact successfully written to: {args.output}")

    # Verify reload
    reloaded = SubspaceArtifact.load(args.output, validate_specs=bias_specs)
    logger.info(f"Artifact reload test PASSED! Verified U shape: {list(reloaded.U.shape)}")


if __name__ == "__main__":
    main()
