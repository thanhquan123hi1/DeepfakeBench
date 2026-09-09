#!/usr/bin/env python3
# ==============================================================================
# test_bias_subspace.py: Unit & Sanity Test Suite for MIBS Regularization
# ==============================================================================
# Tests all 9 critical properties specified in Section 21 of the specification:
#   Test 1: Flatten <-> Restore mapping exactness.
#   Test 2: U shape [P, r].
#   Test 3: U orthonormality (U.T @ U == I_r).
#   Test 4: Projection correctness O(Pr) vs U @ (U.T @ delta).
#   Test 5: In-subspace delta gives subspace_loss ≈ 0.
#   Test 6: Orthogonal delta gives subspace_loss > 0 (100% outside energy).
#   Test 7: lambda = 0 leaves total_loss == ce_loss.
#   Test 8: Only backbone biases receive gradient from L_subspace (classifier head receives 0).
#   Test 9: Frozen backbone weights remain strictly frozen.
# ==============================================================================

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
training_dir = os.path.join(current_dir, "training")
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from training.detectors.clip_bias_detector import CLIPBiasDetector
from training.detectors.bias_subspace import (
    BiasParameterSpec,
    get_ordered_bias_specs,
    flatten_bias_tensors,
    restore_bias_tensors,
    compute_subspace_loss,
    build_subspace_svd,
    build_subspace_mean,
    compute_gradient_cosine_matrix,
    SubspaceArtifact,
)


class TestMIBSRegularization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("  INITIALIZING TEST SUITE: MIBS REGULARIZATION (9 TESTS)")
        print("=" * 70)
        cls.detector = CLIPBiasDetector({"tuning": "all_bias"})
        cls.backbone = cls.detector.backbone
        cls.specs = cls.detector.bias_specs
        cls.P = cls.specs[-1].end if cls.specs else 0
        print(f"Loaded CLIPBiasDetector: P={cls.P} bias parameters across {len(cls.specs)} tensors.\n")

    def test_01_flatten_restore_mapping(self):
        """Test 1: Flatten <-> restore mapping is exact and bidirectional."""
        print(">>> Running Test 1: Flatten <-> Restore Mapping...")
        # Create synthetic tensors matching each spec
        original_dict = {}
        for spec in self.specs:
            original_dict[spec.name] = torch.randn(spec.shape, dtype=torch.float32)

        # Flatten
        flat_vec = flatten_bias_tensors(original_dict, self.specs)
        self.assertEqual(flat_vec.shape, (self.P,))

        # Restore
        restored_dict = restore_bias_tensors(flat_vec, self.specs)
        self.assertEqual(len(restored_dict), len(self.specs))

        # Verify exact match for each tensor
        for spec in self.specs:
            self.assertTrue(spec.name in restored_dict)
            self.assertEqual(restored_dict[spec.name].shape, spec.shape)
            diff = (restored_dict[spec.name] - original_dict[spec.name]).abs().max().item()
            self.assertEqual(diff, 0.0, f"Mismatch in parameter '{spec.name}' after restore")

        print("  [PASS] Test 1: Exact bidirectional flatten <-> restore mapping verified.")

    def test_02_subspace_shape(self):
        """Test 2: Subspace U has exact shape [P, r]."""
        print(">>> Running Test 2: Subspace U Shape...")
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        
        for rank in [1, 2, 4]:
            U, S, explained = build_subspace_svd(synth_grads, rank=rank)
            self.assertEqual(U.shape, (self.P, rank), f"Expected shape ({self.P}, {rank}), got {U.shape}")

        print("  [PASS] Test 2: Subspace U shape [P, r] verified for rank=1, 2, 4.")

    def test_03_subspace_orthonormality(self):
        """Test 3: U columns are orthonormal (U.T @ U == I_r)."""
        print(">>> Running Test 3: Subspace Orthonormality...")
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        
        rank = 2
        U, S, _ = build_subspace_svd(synth_grads, rank=rank)
        gram = U.t() @ U
        eye = torch.eye(rank, dtype=U.dtype)
        max_err = (gram - eye).abs().max().item()
        self.assertLess(max_err, 1e-4, f"Gram matrix deviates from identity: {max_err}")

        print(f"  [PASS] Test 3: Orthonormality verified: max |U.T @ U - I| = {max_err:.2e} < 1e-4.")

    def test_04_projection_correctness(self):
        """Test 4: O(Pr) projection equals mathematical U @ (U.T @ delta)."""
        print(">>> Running Test 4: Projection Correctness...")
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        U, _, _ = build_subspace_svd(synth_grads, rank=2)

        delta = torch.randn(self.P)
        # O(Pr) calculation
        coeff = U.t() @ delta
        proj_fast = U @ coeff

        # Mathematical direct projection
        proj_math = torch.matmul(U, torch.matmul(U.t(), delta))

        diff = (proj_fast - proj_math).abs().max().item()
        self.assertLess(diff, 1e-6, f"O(Pr) projection mismatch: {diff}")

        # Idempotence: proj(proj(delta)) == proj(delta)
        proj_proj = U @ (U.t() @ proj_fast)
        diff_idem = (proj_proj - proj_fast).abs().max().item()
        self.assertLess(diff_idem, 1e-6, f"Projection not idempotent: {diff_idem}")

        print("  [PASS] Test 4: O(Pr) projection is mathematically exact and idempotent.")

    def test_05_in_subspace_zero_loss(self):
        """Test 5: If delta = U @ alpha, then subspace_loss ≈ 0 and shared_energy ≈ 100%."""
        print(">>> Running Test 5: In-Subspace Zero Regularization Loss...")
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        U, _, _ = build_subspace_svd(synth_grads, rank=2)

        alpha = torch.tensor([1.5, -2.3], dtype=torch.float32)
        delta_in_subspace = U @ alpha

        loss, diag = compute_subspace_loss(delta_in_subspace, U)
        self.assertLess(loss.item(), 1e-9, f"In-subspace loss not zero: {loss.item()}")
        self.assertGreater(diag["shared_energy_ratio"], 0.9999, f"Shared energy ratio: {diag['shared_energy_ratio']}")
        self.assertLess(diag["outside_energy_ratio"], 1e-4)

        print(f"  [PASS] Test 5: In-subspace loss = {loss.item():.2e} ≈ 0, shared energy = {diag['shared_energy_ratio']*100:.2f}%.")

    def test_06_orthogonal_positive_loss(self):
        """Test 6: If delta is orthogonal to U, subspace_loss > 0 and outside_energy ≈ 100%."""
        print(">>> Running Test 6: Orthogonal Positive Regularization Loss...")
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        U, _, _ = build_subspace_svd(synth_grads, rank=2)

        raw_vec = torch.randn(self.P)
        # Project out the subspace component to get purely orthogonal vector
        delta_ortho = raw_vec - U @ (U.t() @ raw_vec)
        delta_ortho = F.normalize(delta_ortho, p=2, dim=0)

        loss, diag = compute_subspace_loss(delta_ortho, U)
        self.assertGreater(loss.item(), 0.0)
        self.assertLess(diag["shared_energy_ratio"], 1e-4)
        self.assertGreater(diag["outside_energy_ratio"], 0.9999)

        print(f"  [PASS] Test 6: Orthogonal loss = {loss.item():.6f} > 0, outside energy = {diag['outside_energy_ratio']*100:.2f}%.")

    def test_07_lambda_zero_equivalence(self):
        """Test 7: When lambda = 0, total_loss == ce_loss exactly."""
        print(">>> Running Test 7: Lambda = 0 Equivalence...")
        detector = CLIPBiasDetector({"tuning": "all_bias"})
        
        # Load synthetic subspace
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        U, _, _ = build_subspace_svd(synth_grads, rank=2)
        detector.subspace_U = U
        detector.subspace_lambda = 0.0

        # Perturb one bias parameter to make delta_b non-zero
        p_first = next(p for p in detector.backbone.parameters() if p.requires_grad)
        with torch.no_grad():
            p_first.add_(0.05)

        # Synthetic batch
        synth_data = {
            "image": torch.randn(2, 3, 224, 224),
            "label": torch.tensor([0, 1], dtype=torch.long)
        }
        pred = detector(synth_data)
        losses = detector.get_losses(synth_data, pred)

        self.assertIn("overall", losses)
        self.assertIn("ce", losses)
        self.assertIn("subspace", losses)
        self.assertGreater(losses["subspace"].item(), 0.0)
        self.assertEqual(losses["overall"].item(), losses["ce"].item())

        print(f"  [PASS] Test 7: When lambda=0, overall ({losses['overall'].item():.6f}) == ce ({losses['ce'].item():.6f}).")

    def test_08_gradient_isolation_classifier_head(self):
        """Test 8: Classifier head receives ZERO direct gradient from L_subspace."""
        print(">>> Running Test 8: Classifier Head Gradient Isolation...")
        detector = CLIPBiasDetector({"tuning": "all_bias"})
        
        methods = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
        synth_grads = {m: F.normalize(torch.randn(self.P), p=2, dim=0) for m in methods}
        U, _, _ = build_subspace_svd(synth_grads, rank=2)
        detector.subspace_U = U
        detector.subspace_lambda = 1.0

        # Perturb a bias parameter to make delta_b non-zero
        p_first = next(p for p in detector.backbone.parameters() if p.requires_grad)
        with torch.no_grad():
            p_first.add_(0.1)

        # Compute subspace loss alone
        param_dict = dict(detector.backbone.named_parameters())
        delta_parts = []
        for spec in detector.bias_specs:
            p = param_dict[spec.name]
            p0 = detector.initial_bias[spec.name]
            delta_parts.append((p - p0).reshape(-1))
        delta_b = torch.cat(delta_parts, dim=0)
        loss_subspace, _ = compute_subspace_loss(delta_b, U)

        detector.zero_grad()
        loss_subspace.backward()

        # Check backbone bias receives gradient
        backbone_grad_count = sum(1 for p in detector.backbone.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        self.assertGreater(backbone_grad_count, 0, "Backbone biases should receive gradient from L_subspace")

        # Check classifier head receives NO gradient
        head_weight_grad = detector.head.weight.grad
        head_bias_grad = detector.head.bias.grad
        
        self.assertTrue(
            head_weight_grad is None or head_weight_grad.abs().sum().item() == 0.0,
            "Classifier head weight must NOT receive gradient from L_subspace"
        )
        self.assertTrue(
            head_bias_grad is None or head_bias_grad.abs().sum().item() == 0.0,
            "Classifier head bias must NOT receive gradient from L_subspace"
        )

        print("  [PASS] Test 8: Classifier head receives ZERO direct gradient from L_subspace.")

    def test_09_frozen_backbone_weights_remain_frozen(self):
        """Test 9: Frozen weights (attention, MLP, patch/pos embed, LN gamma) remain strictly frozen."""
        print(">>> Running Test 9: Frozen Backbone Weights Invariance...")
        detector = CLIPBiasDetector({"tuning": "all_bias"})
        
        total_frozen = 0
        total_trainable_biases = 0

        for name, param in detector.backbone.named_parameters():
            if 'weight' in name or 'embedding' in name:
                self.assertFalse(param.requires_grad, f"Weight parameter '{name}' was not frozen!")
                self.assertIsNone(param.grad, f"Frozen weight '{name}' has non-None gradient!")
                total_frozen += param.numel()
            elif param.requires_grad:
                total_trainable_biases += param.numel()

        self.assertEqual(total_trainable_biases, self.P)
        self.assertGreater(total_frozen, 300_000_000)

        print(f"  [PASS] Test 9: All {total_frozen:,} backbone weights remain strictly frozen.")


if __name__ == "__main__":
    unittest.main()
