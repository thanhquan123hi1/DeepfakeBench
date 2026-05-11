import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        logger.info("Loading CLIP ViT-L/14 for GenD-Effort CE Only + Linear Head...")

        self.backbone = self.build_backbone(self.config)

        # Linear Head: đầu vào là feature đã L2 normalize
        self.head = nn.Linear(1024, 2)

        self.build_loss(self.config)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self._setup_trainable_params()

    def build_backbone(self, config):
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        except Exception:
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                local_files_only=True
            )

        return clip_model.vision_model

    def build_loss(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        weight_real = float(config.get('weight_real', 1.0))
        weight_fake = float(config.get('weight_fake', 2.0))
        class_weights = torch.tensor([weight_real, weight_fake], device=device)

        label_smoothing = float(config.get('label_smoothing', 0.1))

        # Chỉ dùng CrossEntropyLoss
        self.loss_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    def _setup_trainable_params(self):
        # Đóng băng toàn bộ backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        count = 0

        # Mở khóa Linear Head
        for p in self.head.parameters():
            p.requires_grad = True
            count += p.numel()

        # Mở khóa LayerNorm + Bias trong backbone
        for name, p in self.backbone.named_parameters():
            if 'layer_norm' in name or 'layernorm' in name or 'bias' in name:
                p.requires_grad = True
                count += p.numel()

        logger.info(f"LN-Tuning + BitFit + Linear Head Initialized. Trainable params: {count}")

    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(data_dict['image'])
        feat = outputs.pooler_output
        return feat

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        # features đã được L2 normalize trước khi đưa vào Linear Head
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)

        # Chuẩn hóa L2 feature trước khi đưa vào Linear Head
        norm_features = F.normalize(raw_features, p=2, dim=1, eps=1e-6)

        pred = self.classifier(norm_features)

        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            'cls': pred,
            'prob': prob,
            'feat': raw_features,
            'feat_norm': norm_features
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        # Chỉ tính CE Loss
        loss_ce = self.loss_ce(pred, label)

        loss_dict = {
            'overall': loss_ce,
            'loss_ce': loss_ce
        }

        # Log riêng loss real/fake
        with torch.no_grad():
            mask_real = (label == 0)
            mask_fake = (label == 1)

            loss_dict['real_loss'] = (
                self.loss_ce(pred[mask_real], label[mask_real])
                if mask_real.sum() > 0
                else torch.tensor(0.0, device=pred.device)
            )

            loss_dict['fake_loss'] = (
                self.loss_ce(pred[mask_fake], label[mask_fake])
                if mask_fake.sum() > 0
                else torch.tensor(0.0, device=pred.device)
            )

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(),
            pred.detach()
        )

        return {
            'acc': acc,
            'auc': auc,
            'eer': eer,
            'ap': ap
        }
