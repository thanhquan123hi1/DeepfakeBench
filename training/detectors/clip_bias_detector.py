import os
import logging
from typing import Union, Dict, Any, Optional

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

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip_bias')
class CLIPBiasDetector(AbstractDetector):
    """
    CLIP ViT detector with parameter-efficient bias ablation support.
    Allows isolating and training specific bias parameter subsets (e.g. v_bias, linear_bias, ln_bias)
    and depth ranges (early, middle, late) without altering model architecture, loss, or evaluation.
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
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
