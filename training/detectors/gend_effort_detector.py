import os
import logging
import torch
import torch.nn as nn
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='gend_effort')
class GenDEffortDetector(AbstractDetector):
    def __init__(self, config=None):
        super(GenDEffortDetector, self).__init__()
        self.config = config or {}
        
        logger.info("Building Detector: CLIP ViT-L/14 + Linear Head + BitFit/LN Tuning")
        
        # 1. Khởi tạo Backbone (CLIP Vision)
        self.backbone = self.build_backbone(self.config)
        
        # 2. Linear Head đơn giản (1024 -> 2)
        # ViT-L/14 output feature dimension là 1024
        self.head = nn.Linear(1024, 2)
        
        # 3. Thiết lập duy nhất CrossEntropyLoss
        self.build_loss(self.config)
        
        # 4. Đóng băng và mở khóa tham số (BitFit + LayerNorm)
        self._setup_trainable_params()

    def build_backbone(self, config):
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        except Exception:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        return clip_model.vision_model

    def build_loss(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lấy trọng số từ config (mặc định ưu tiên class Fake để tránh bỏ lọt)
        weight_real = float(config.get('weight_real', 1.0))
        weight_fake = float(config.get('weight_fake', 2.0))
        class_weights = torch.tensor([weight_real, weight_fake], device=device)
        
        label_smoothing = float(config.get('label_smoothing', 0.1))

        self.loss_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    def _setup_trainable_params(self):
        # Đóng băng toàn bộ trọng số Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        count = 0
        # Mở khóa Linear Head
        for p in self.head.parameters():
            p.requires_grad = True
            count += p.numel()
            
        # Mở khóa LayerNorm và các Bias (BitFit) trong Backbone
        for name, p in self.backbone.named_parameters():
            if 'layer_norm' in name or 'layernorm' in name or 'bias' in name:
                p.requires_grad = True
                count += p.numel()

        logger.info(f"Tuning Strategy: BitFit + LN + Linear Head. Total trainable params: {count:,}")

    def features(self, data_dict: dict) -> torch.tensor:
        # Lấy pooler_output (vector đại diện cho toàn bộ ảnh)
        outputs = self.backbone(data_dict['image'])
        return outputs.pooler_output 

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # Đưa thẳng qua Linear Head
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        feat = self.features(data_dict)
        pred = self.classifier(feat)
        
        # Xác suất cho class 1 (Fake)
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        return {'cls': pred, 'prob': prob, 'feat': feat}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        # Chỉ tính Cross Entropy Loss
        loss_ce = self.loss_ce(pred, label)

        return {
            'overall': loss_ce,
            'loss_ce': loss_ce
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}