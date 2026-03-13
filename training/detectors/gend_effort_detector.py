import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

logger = logging.getLogger(__name__)

# =========================================================================
# 1. ASYMMETRIC CONTRASTIVE LOSS
# =========================================================================

class AsymmetricContrastiveLoss(nn.Module):
    """
    Asymmetric Contrastive Loss:
    - Kéo (pull) các mẫu Real lại với nhau thành một lõi (core) cực đặc.
    - Đẩy (push) các mẫu Fake ra xa khỏi lõi Real.
    - Không ép các mẫu Fake phải co cụm lại với nhau.
    """
    def __init__(self, temperature=0.07):
        super(AsymmetricContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Cần ít nhất 2 mẫu trong batch để tính loss
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Tính ma trận tương đồng Cosine (vì features đã được L2 Normalize từ trước)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Loại bỏ đường chéo chính (không so sánh ảnh với chính nó)
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(mask_self, -1e9) 

        # Tính Log-Softmax (Mẫu số đóng vai trò lực đẩy đẩy tất cả các vector ra xa nhau)
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Tạo mask: Chỉ kích hoạt lực kéo khi cả 2 ảnh đều là Real (label = 0)
        labels = labels.contiguous().view(-1, 1)
        is_real = (labels == 0).float() 
        pos_mask = torch.matmul(is_real, is_real.T).to(device)
        pos_mask.masked_fill_(mask_self, 0.0)

        real_indices = torch.where(is_real.squeeze() == 1)[0]
        if len(real_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_counts = pos_mask.sum(dim=1)
        valid_real_indices = real_indices[pos_counts[real_indices] > 0]
        
        if len(valid_real_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Trung bình hóa log_prob cho các cặp Positive (Real-Real)
        mean_log_prob_pos = (pos_mask[valid_real_indices] * log_prob[valid_real_indices]).sum(dim=1) / pos_counts[valid_real_indices]

        loss = -mean_log_prob_pos.mean()
        return loss

# =========================================================================
# 2. COSINE CLASSIFIER HEAD
# =========================================================================

class CosineClassifier(nn.Module):
    """
    Cosine Classifier Head:
    Chuẩn hóa L2 cho trọng số W. 
    Lợi ích: Khóa chặt độ lớn (Magnitude), đo lường 100% bằng góc (Angle), 
    đồng bộ tuyệt đối với mặt cầu L2 của AsymSupCon.
    """
    def __init__(self, in_features, num_classes):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x được giả định là đã qua F.normalize(p=2)
        # Chuẩn hóa luôn cả trọng số W lên mặt cầu
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-6)
        
        # Tích vô hướng của 2 vector đã normalize chính là Cosine Similarity
        cosine_sim = F.linear(x, w_norm)
        return cosine_sim

# =========================================================================
# 3. MAIN DETECTOR CLASS (GEND: LN + BITFIT + ASYM-SUPCON + COSINE HEAD)
# =========================================================================

@DETECTOR.register_module(module_name='gend_effort')
class GenDEffortDetector(AbstractDetector):
    def __init__(self, config=None):
        super(GenDEffortDetector, self).__init__()
        self.config = config or {}
        
        self.temperature = self.config.get('temperature', 0.07)
        self.lambda_supcon = float(self.config.get('lambda_supcon', 0.3))
        
        logger.info("Loading CLIP ViT-L/14 for GenD-Effort (LN + BitFit + AsymSupCon + CosineHead)...")
        self.backbone = self.build_backbone(self.config)
        
        # Thay thế nn.Linear bằng CosineClassifier
        self.head = CosineClassifier(1024, 2)
        
        # Logit Scale (Giữ vai trò bán kính/nhiệt độ học)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / self.temperature))
        
        self.build_loss(self.config)
        
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        self._setup_trainable_params()

    def build_backbone(self, config):
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        except Exception:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)

        return clip_model.vision_model

    def build_loss(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        weight_real = float(config.get('weight_real', 1.0))
        weight_fake = float(config.get('weight_fake', 2.0))
        class_weights = torch.tensor([weight_real, weight_fake], device=device)
        
        label_smoothing = float(config.get('label_smoothing', 0.1))

        # CrossEntropyLoss sẽ nhận logit = Scale * Cosine
        self.loss_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        self.loss_supcon = AsymmetricContrastiveLoss(temperature=self.temperature)

    def _setup_trainable_params(self):
        # 1. Đóng băng toàn bộ trọng số Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        count = 0
        
        # 2. Mở khóa Cosine Head và Logit Scale
        for p in self.head.parameters():
            p.requires_grad = True
            count += p.numel()
        self.logit_scale.requires_grad = True
            
        # 3. KỸ THUẬT KÉP: Mở khóa LayerNorm VÀ Bias (BitFit)
        for name, p in self.backbone.named_parameters():
            if 'layer_norm' in name or 'layernorm' in name or 'bias' in name:
                p.requires_grad = True
                count += p.numel()

        logger.info(f"Hybrid LN-Tuning + BitFit Initialized. Trainable params: {count}")

    def features(self, data_dict: dict) -> torch.tensor:
        outputs = self.backbone(data_dict['image'])
        feat = outputs.pooler_output 
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # Nhận L2-normalized features, nhân với Cosine Head, rồi scale lên
        return self.head(features) * self.logit_scale.exp()

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)
        
        # L2 Normalization là BẮT BUỘC cho cả AsymSupCon và Cosine Head
        norm_features = F.normalize(raw_features, p=2, dim=1, eps=1e-6)
        
        # Truyền norm_features vào classifier
        pred = self.classifier(norm_features)
        
        # Softmax để lấy xác suất của class 1 (Fake)
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        return {'cls': pred, 'prob': prob, 'feat': raw_features, 'feat_norm': norm_features}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        feat_norm = pred_dict['feat_norm'] 

        # Tính Cross Entropy Loss
        loss_ce = self.loss_ce(pred, label)
        loss_supcon_val = torch.tensor(0.0, device=pred.device)

        # Tính AsymSupCon Loss (Chỉ tính khi Training)
        if self.training:
            loss_supcon_val = self.loss_supcon(feat_norm, label)
            if torch.isnan(loss_supcon_val):
                loss_supcon_val = torch.nan_to_num(loss_supcon_val)

        # Trộn tổng Loss
        total_loss = loss_ce + (self.lambda_supcon * loss_supcon_val)

        loss_dict = {
            'overall': total_loss,
            'loss_ce': loss_ce,
            'loss_supcon': loss_supcon_val
        }
        
        # Tính Loss riêng lẻ để tiện theo dõi log
        with torch.no_grad():
            mask_real = (label == 0)
            mask_fake = (label == 1)
            loss_dict['real_loss'] = self.loss_ce(pred[mask_real], label[mask_real]) if mask_real.sum() > 0 else torch.tensor(0.0, device=pred.device)
            loss_dict['fake_loss'] = self.loss_ce(pred[mask_fake], label[mask_fake]) if mask_fake.sum() > 0 else torch.tensor(0.0, device=pred.device)

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
