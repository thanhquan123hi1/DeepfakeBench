import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train

# ==========================================
# 1. LOSS FUNCTIONS (Uniformity & Alignment)
# ==========================================
# Dựa trên file src/losses/unifalign.py của tác giả
def alignment_loss(embeddings, labels, alpha=2):
    """
    Label-aware Alignment loss: Kéo các mẫu cùng nhãn lại gần nhau.
    """
    if embeddings.size(0) < 2: return torch.tensor(0.0, device=embeddings.device)
    
    # Tạo ma trận so sánh nhãn (chỉ lấy cặp cùng nhãn, khác mẫu)
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    
    if positive_indices.numel() == 0: return torch.tensor(0.0, device=embeddings.device)
    
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity_loss(x, t=2, clip_value=1e-6):
    """
    Uniformity loss: Đẩy các mẫu phân bố đều trên mặt cầu.
    """
    if x.size(0) < 2: return torch.tensor(0.0, device=x.device)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()

# ==========================================
# 2. GenD DETECTOR
# ==========================================
@DETECTOR.register_module(module_name='genD')
class GenDDetector(nn.Module):
    def __init__(self, config=None):
        super(GenDDetector, self).__init__()
        self.config = config
        
        # 1. Load Backbone (CLIP ViT-L/14)
        # Lưu ý: Cập nhật đường dẫn local nếu cần
        print("Loading CLIP ViT-L/14 for GenD...")
        self.backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model
        
        # 2. Setup LN-Tuning (Chỉ train LayerNorm)
        self._setup_training_params()
        
        # 3. Classifier Head (Linear Layer)
        # Input: 1024 (ViT-L hidden size), Output: 2 (Real/Fake)
        self.head = nn.Linear(1024, 2)
        
        # 4. Loss Weights (Tham khảo từ paper/code gốc)
        self.lambda_align = config.get('lambda_align', 1.0) if config else 1.0
        self.lambda_unif = config.get('lambda_unif', 1.0) if config else 1.0
        
        self.loss_ce = nn.CrossEntropyLoss()

    def _setup_training_params(self):
        """
        Đóng băng toàn bộ backbone, CHỈ mở khóa (requires_grad=True)
        cho các lớp Layer Normalization.
        """
        # Đóng băng toàn bộ trước
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Mở khóa các lớp LayerNorm
        trainable_params = 0
        for name, param in self.backbone.named_parameters():
            # CLIP dùng 'layer_norm' hoặc 'layernorm' trong tên biến
            if 'layer_norm' in name or 'layernorm' in name:
                param.requires_grad = True
                trainable_params += param.numel()
        
        print(f"GenD Initialized. Trainable params (LayerNorm only): {trainable_params}")

    def features(self, data_dict: dict) -> torch.tensor:
        # Lấy feature từ backbone (CLS token / pooler_output)
        # CLIP vision model trả về pooler_output (đã qua LN + projection) hoặc last_hidden_state
        # GenD dùng CLS token sau khi qua encoder
        outputs = self.backbone(data_dict['image'])
        feat = outputs.pooler_output # Shape: [B, 1024]
        
        # Quan trọng: GenD yêu cầu L2 Normalization feature trước khi vào classifier
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # Features đã được normalize ở bước trên
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        # 1. Extract Features (Normalized)
        features = self.features(data_dict)
        
        # 2. Classification
        pred = self.classifier(features)
        
        # 3. Probability
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        features = pred_dict['feat'] # Features này đã L2-normalized
        
        # 1. Cross Entropy Loss
        loss_cls = self.loss_ce(pred, label)
        
        # 2. Metric Learning Losses (Chỉ tính khi training)
        loss_align = torch.tensor(0.0, device=pred.device)
        loss_unif = torch.tensor(0.0, device=pred.device)
        
        if self.training:
            loss_align = alignment_loss(features, label)
            loss_unif = uniformity_loss(features)
            
        # Tổng hợp Loss
        overall_loss = loss_cls + (self.lambda_align * loss_align) + (self.lambda_unif * loss_unif)
        
        loss_dict = {
            'overall': overall_loss,
            'ce_loss': loss_cls,
            'align_loss': loss_align,
            'unif_loss': loss_unif
        }
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict