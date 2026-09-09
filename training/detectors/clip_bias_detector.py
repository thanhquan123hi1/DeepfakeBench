import os
import logging
from typing import Union, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from .bias_tuning import (
    TuningConfig,
    apply_bias_tuning,
    print_trainable_parameters,
    verify_tuning_configuration,
)
from .bias_subspace import (
    BiasParameterSpec,
    get_ordered_bias_specs,
    snapshot_initial_bias,
    compute_subspace_loss,
    SubspaceArtifact,
)

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip_bias')
class CLIPBiasDetector(AbstractDetector):
    """
    CLIP ViT detector with parameter-efficient bias ablation support and
    Manipulation-Invariant Bias Subspace (MIBS) Regularization.
    
    Allows isolating and training specific bias parameter subsets (e.g. all_bias, v_bias, ln_bias)
    and softly regularizing bias updates (b - b0) toward a shared low-dimensional subspace U
    derived from multiple manipulation methods.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super(CLIPBiasDetector, self).__init__()
        self.config = config or {}
        
        # 1. Build Backbone & Head
        self.backbone = self.build_backbone(self.config)
        hidden_dim = getattr(self.backbone.config, 'hidden_size', 1024)
        self.head = nn.Linear(hidden_dim, 2)
        
        # 2. Build Loss Function
        self.loss_func = self.build_loss(self.config)
        
        # Feature normalization preference
        self.normalize_features = self.config.get('normalize_features', True)
        
        # 3. Setup Tuning Strategy
        self.tuning_config: Optional[TuningConfig] = None
        self.tuning_metadata: Dict[str, Any] = {}
        self._init_tuning_strategy(self.config)

        # 4. Canonical Bias Parameter Specs & Initial Bias Snapshot (b0)
        self.bias_specs: List[BiasParameterSpec] = get_ordered_bias_specs(self.backbone, self.tuning_config)
        self.initial_bias: Dict[str, torch.Tensor] = snapshot_initial_bias(self.backbone, self.bias_specs)

        # 5. Subspace Regularization State (MIBS)
        self.subspace_artifact: Optional[SubspaceArtifact] = None
        self.subspace_lambda: float = 0.0
        self.latest_subspace_metrics: Dict[str, float] = {}
        self.register_buffer('subspace_U', None, persistent=False)

        # Check config for subspace initialization
        self._init_subspace_from_config(self.config)

    def build_backbone(self, config: Dict[str, Any]) -> nn.Module:
        clip_name = config.get('clip_model_name', "openai/clip-vit-large-patch14")
        logger.info(f"Loading CLIP model for CLIPBiasDetector: {clip_name}")
        try:
            clip_model = CLIPModel.from_pretrained(clip_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(clip_name, local_files_only=True)
        return clip_model.vision_model

    def build_loss(self, config: Dict[str, Any]) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _init_tuning_strategy(self, config: Dict[str, Any]):
        tuning_entry = config.get('tuning', 'all_bias')
        if isinstance(tuning_entry, str):
            self.tuning_config = TuningConfig.from_string(tuning_entry)
        elif isinstance(tuning_entry, dict):
            self.tuning_config = TuningConfig(
                parameter_type=tuning_entry.get('parameter_type', 'all_bias'),
                layer_range=tuning_entry.get('layer_range', 'all'),
                train_classifier=tuning_entry.get('train_classifier', True),
                include_pre_post_ln=tuning_entry.get('include_pre_post_ln', True),
            )
        elif isinstance(tuning_entry, TuningConfig):
            self.tuning_config = tuning_entry
        else:
            self.tuning_config = TuningConfig(parameter_type="all_bias", layer_range="all")

        # Apply requires_grad configuration
        self.tuning_metadata = apply_bias_tuning(
            model=self,
            backbone=self.backbone,
            classifier_head=self.head,
            tuning_config=self.tuning_config
        )

    def _init_subspace_from_config(self, config: Dict[str, Any]):
        subspace_cfg = config.get('bias_subspace', {})
        if isinstance(subspace_cfg, dict) and subspace_cfg.get('enabled', False):
            path = subspace_cfg.get('path', None)
            lambda_val = subspace_cfg.get('lambda', 0.01)
            if path:
                self.load_subspace(path, lambda_val=lambda_val)

    def load_subspace(
        self,
        artifact_or_path: Union[str, SubspaceArtifact],
        lambda_val: float = 0.01,
        atol: float = 1e-4
    ):
        """
        Loads and validates Manipulation-Invariant Bias Subspace artifact U.
        Strictly verifies parameter names, order, shapes, and orthonormality.
        Registers U as a non-persistent buffer for automatic device / DDP handling.
        """
        if isinstance(artifact_or_path, str):
            artifact = SubspaceArtifact.load(artifact_or_path, validate_specs=self.bias_specs, atol=atol)
        elif isinstance(artifact_or_path, SubspaceArtifact):
            artifact = artifact_or_path
            artifact.validate(self.bias_specs, atol=atol)
        else:
            raise TypeError(f"Expected path (str) or SubspaceArtifact, got {type(artifact_or_path)}")

        self.subspace_artifact = artifact
        self.subspace_lambda = float(lambda_val)
        U_tensor = artifact.U.to(torch.float32)
        self.register_buffer('subspace_U', U_tensor, persistent=False)
        logger.info(
            f"Loaded MIBS subspace artifact (rank={artifact.rank}, method={artifact.subspace_method}, "
            f"lambda={self.subspace_lambda}, P={artifact.total_bias_params})"
        )

    def set_tuning_strategy(self, strategy: Union[str, Dict[str, Any], TuningConfig], verify: bool = True):
        """
        Dynamically change tuning strategy on the detector.
        """
        if isinstance(strategy, str):
            self.tuning_config = TuningConfig.from_string(strategy)
        elif isinstance(strategy, dict):
            self.tuning_config = TuningConfig(
                parameter_type=strategy.get('parameter_type', 'all_bias'),
                layer_range=strategy.get('layer_range', 'all'),
                train_classifier=strategy.get('train_classifier', True),
                include_pre_post_ln=strategy.get('include_pre_post_ln', True),
            )
        elif isinstance(strategy, TuningConfig):
            self.tuning_config = strategy

        self.tuning_metadata = apply_bias_tuning(
            model=self,
            backbone=self.backbone,
            classifier_head=self.head,
            tuning_config=self.tuning_config
        )
        
        # Re-resolve bias specs and snapshot
        self.bias_specs = get_ordered_bias_specs(self.backbone, self.tuning_config)
        self.initial_bias = snapshot_initial_bias(self.backbone, self.bias_specs)

        if verify:
            verify_tuning_configuration(
                model=self,
                backbone=self.backbone,
                classifier_head=self.head,
                tuning_config=self.tuning_config,
                run_grad_check=False
            )

    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(data_dict['image'])
        feat = outputs.pooler_output
        return feat

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        raw_feat = self.features(data_dict)
        if self.normalize_features:
            feat_cls = F.normalize(raw_feat, p=2, dim=1)
        else:
            feat_cls = raw_feat

        pred = self.classifier(feat_cls)
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            'cls': pred,
            'prob': prob,
            'feat': raw_feat,
            'feat_cls': feat_cls
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss_ce = self.loss_func(pred, label)

        # Subspace regularization (MIBS)
        if self.subspace_U is not None:
            param_dict = dict(self.backbone.named_parameters())
            delta_parts = []
            for spec in self.bias_specs:
                p = param_dict[spec.name]
                p0 = self.initial_bias[spec.name]
                if p0.device != p.device:
                    p0 = p0.to(p.device)
                    self.initial_bias[spec.name] = p0
                delta_parts.append((p - p0).reshape(-1))

            delta_b = torch.cat(delta_parts, dim=0)

            # Ensure U matches device and dtype of delta_b
            U = self.subspace_U
            if U.device != delta_b.device or U.dtype != delta_b.dtype:
                U = U.to(device=delta_b.device, dtype=delta_b.dtype)
                self.subspace_U = U

            loss_subspace, diagnostics = compute_subspace_loss(delta_b, U)
            self.latest_subspace_metrics = diagnostics

            if self.subspace_lambda > 0:
                loss_overall = loss_ce + self.subspace_lambda * loss_subspace
            else:
                loss_overall = loss_ce

            return {
                'overall': loss_overall,
                'ce': loss_ce,
                'subspace': loss_subspace
            }

        return {'overall': loss_ce}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metrics = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        if self.latest_subspace_metrics:
            metrics['bias/update_norm'] = self.latest_subspace_metrics['delta_norm']
            metrics['bias/shared_norm'] = self.latest_subspace_metrics['shared_norm']
            metrics['bias/outside_norm'] = self.latest_subspace_metrics['outside_norm']
            metrics['bias/shared_energy_ratio'] = self.latest_subspace_metrics['shared_energy_ratio']
            metrics['bias/outside_energy_ratio'] = self.latest_subspace_metrics['outside_energy_ratio']
        return metrics
