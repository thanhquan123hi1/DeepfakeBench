import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

from training.loss.asymmetric_supcon_loss import AsymmetricSupConLoss


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='gend_effort')
class GenDEffortDetector(AbstractDetector):
    def __init__(self, config=None):
        super(GenDEffortDetector, self).__init__()
        self.config = config or {}

        logger.info("Loading CLIP ViT-L/14 for GenD-Effort + BCE + Asymmetric SupCon...")

        self.backbone = self.build_backbone(self.config)

        self.feature_dim = int(self.config.get('feature_dim', 1024))
        self.projection_dim = int(self.config.get('projection_dim', 256))

        # BCE binary classification head: output 1 logit
        self.head = nn.Linear(self.feature_dim, 1)

        # Projection head cho contrastive loss
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.projection_dim)
        )

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

        # BCEWithLogitsLoss:
        # label fake = 1 là positive class
        # pos_weight chỉ tác động lên positive class
        pos_weight_value = weight_fake / max(weight_real, 1e-8)
        pos_weight = torch.tensor([pos_weight_value], device=device)

        self.loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Hệ số cho loss phụ
        self.lambda_asym = float(config.get('lambda_asym', 0.05))

        self.loss_asym = AsymmetricSupConLoss(
            temperature=float(config.get('temperature', 0.07)),
            contrast_mode=config.get('contrast_mode', 'all'),
            base_temperature=float(config.get('base_temperature', 0.07)),
            real_label=int(config.get('real_label', 0)),
            lambda_fake=float(config.get('lambda_fake', 0.1)),
            fake_margin=float(config.get('fake_margin', 0.0)),
        )

    def _setup_trainable_params(self):
        """
        Fine-tune strategy:
        - Freeze toàn bộ backbone trước
        - Train:
            1. classifier head
            2. projection head
            3. LayerNorm + bias trong backbone
            4. last_k transformer blocks cuối
        """

        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable_count = 0

        # 1. Classifier head
        for p in self.head.parameters():
            p.requires_grad = True
            trainable_count += p.numel()

        # 2. Projection head
        for p in self.projection_head.parameters():
            p.requires_grad = True
            trainable_count += p.numel()

        # 3. LayerNorm + bias trong backbone
        for name, p in self.backbone.named_parameters():
            name_lower = name.lower()

            is_norm = (
                'layer_norm' in name_lower
                or 'layernorm' in name_lower
                or 'layrnorm' in name_lower  # CLIP HF có pre_layrnorm/post_layernorm
            )

            is_bias = name_lower.endswith('.bias')

            if is_norm or is_bias:
                if not p.requires_grad:
                    p.requires_grad = True
                    trainable_count += p.numel()

        # 4. Last K transformer blocks
        last_k_blocks = int(self.config.get('last_k_blocks', 2))

        if last_k_blocks > 0:
            try:
                layers = self.backbone.encoder.layers
                total_layers = len(layers)
                start_idx = max(0, total_layers - last_k_blocks)

                for idx in range(start_idx, total_layers):
                    for p in layers[idx].parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            trainable_count += p.numel()

                logger.info(
                    f"Unfroze last {last_k_blocks} CLIP ViT blocks: "
                    f"layers [{start_idx}, {total_layers - 1}]"
                )

            except Exception as e:
                logger.warning(f"Could not unfreeze last_k_blocks because: {e}")

        logger.info(
            "Fine-tuning setup: Head + Projection Head + LN/Bias + Last Blocks. "
            f"Trainable params: {trainable_count}"
        )

    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(data_dict['image'])
        feat = outputs.pooler_output
        return feat

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, 1024]

        Returns:
            logits: [B]
        """
        logits = self.head(features).squeeze(1)
        return logits

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)

        # Feature đã L2 normalize cho classifier
        cls_features = F.normalize(raw_features, p=2, dim=1, eps=1e-6)

        # BCE logits
        logits = self.classifier(cls_features)

        # Projection feature cho Asymmetric SupCon
        proj_features = self.projection_head(raw_features)

        # Không cần normalize ở đây vì loss_asym đã normalize bên trong.
        # Nhưng nếu muốn chắc chắn, có thể bật dòng dưới:
        # proj_features = F.normalize(proj_features, p=2, dim=1, eps=1e-6)

        prob = torch.sigmoid(logits)

        return {
            'cls': logits,              # [B]
            'prob': prob,               # [B], xác suất fake
            'feat': raw_features,       # [B, 1024]
            'feat_norm': cls_features,  # [B, 1024]
            'feat_proj': proj_features  # [B, projection_dim]
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']

        # BCE cần float label
        label_float = label.float()

        logits = pred_dict['cls']
        proj_features = pred_dict['feat_proj']

        # 1. BCE loss
        loss_bce = self.loss_bce(logits, label_float)

        # 2. Asymmetric SupCon loss
        loss_asym = self.loss_asym(proj_features, label)

        # 3. Total loss
        loss_overall = loss_bce + self.lambda_asym * loss_asym

        loss_dict = {
            'overall': loss_overall,
            'loss_bce': loss_bce,
            'loss_asym': loss_asym,
        }

        # Log riêng real/fake BCE
        with torch.no_grad():
            mask_real = (label == 0)
            mask_fake = (label == 1)

            loss_dict['real_loss'] = (
                self.loss_bce(logits[mask_real], label_float[mask_real])
                if mask_real.sum() > 0
                else torch.tensor(0.0, device=logits.device)
            )

            loss_dict['fake_loss'] = (
                self.loss_bce(logits[mask_fake], label_float[mask_fake])
                if mask_fake.sum() > 0
                else torch.tensor(0.0, device=logits.device)
            )

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']

        # cls hiện tại là 1-logit [B].
        # Nếu calculate_metrics_for_train đang nhận logits [B, 2],
        # ta convert tạm thành 2-logit.
        logits_fake = pred_dict['cls'].detach()

        logits_two_class = torch.stack(
            [-logits_fake, logits_fake],
            dim=1
        )

        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(),
            logits_two_class
        )

        return {
            'acc': acc,
            'auc': auc,
            'eer': eer,
            'ap': ap
        }
