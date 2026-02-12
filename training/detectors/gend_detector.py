import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train

# ==========================================
# 1. LOSS FUNCTIONS
# ==========================================
def alignment_loss(embeddings, labels, alpha=2):
    if embeddings.size(0) < 2: return torch.tensor(0.0, device=embeddings.device)
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0: return torch.tensor(0.0, device=embeddings.device)
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity_loss(x, t=2, clip_value=1e-6):
    if x.size(0) < 2: return torch.tensor(0.0, device=x.device)
    sq_dist = torch.pdist(x, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().clamp(min=clip_value).log()

# ==========================================
# 2. GenD DETECTOR (FINAL MATCHING VERSION)
# ==========================================
@DETECTOR.register_module(module_name='gend')
class GenDDetector(nn.Module):
    def __init__(self, config=None):
        super(GenDDetector, self).__init__()
        self.config = config
        
        print("Loading CLIP ViT-L/14 for GenD...")
        # Load CLIP Model
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # Chỉ lấy phần Vision Model (Bỏ qua Text Model và Projection)
        self.backbone = clip_model.vision_model
        
        self._setup_training_params()
        
        # Classifier Head
        self.head = nn.Linear(1024, 2)
        
        # Loss config
        self.lambda_align = config.get('lambda_align', 0.1) if config else 0.1
        self.lambda_unif = config.get('lambda_unif', 0.5) if config else 0.5
        self.loss_ce = nn.CrossEntropyLoss()

    def _setup_training_params(self):
        # Đóng băng toàn bộ backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Mở khóa LayerNorm (Bao gồm cả post_layernorm trong pooler_output)
        trainable_params = 0
        for name, param in self.backbone.named_parameters():
            if 'layer_norm' in name or 'layernorm' in name:
                param.requires_grad = True
                trainable_params += param.numel()
        print(f"GenD Backbone Initialized. Trainable params (LN): {trainable_params}")

    def features(self, data_dict: dict) -> torch.tensor:
        """
        Dùng pooler_output theo đúng source code gốc.
        Trong HF, pooler_output = CLS Token + Post LayerNorm.
        """
        # Không cần output_hidden_states=True nữa vì pooler_output là mặc định
        outputs = self.backbone(data_dict['image'])
        
        # Lấy pooler_output (Shape: [Batch, 1024])
        # Đây chính là CLS token đã qua lớp LN cuối cùng.
        feat = outputs.pooler_output 
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        # 1. Raw Features (Từ pooler_output)
        raw_features = self.features(data_dict)
        
        # 2. Luồng Classification: Dùng Raw Features (Khớp Effort/Checkpoint)
        
        
        # 3. Luồng Metric Learning: Dùng Normalized Features (Khớp Paper)
        norm_features = F.normalize(raw_features, p=2, dim=1)
        pred = self.classifier(norm_features)
        
        # 4. Xác suất
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        return {'cls': pred, 'prob': prob, 'feat': raw_features, 'feat_norm': norm_features}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        feat_norm = pred_dict['feat_norm'] 
        
        loss_cls = self.loss_ce(pred, label)
        
        loss_align = torch.tensor(0.0, device=pred.device)
        loss_unif = torch.tensor(0.0, device=pred.device)
        
        if self.training:
            loss_align = alignment_loss(feat_norm, label)
            loss_unif = uniformity_loss(feat_norm)
            
        overall_loss = loss_cls + (self.lambda_align * loss_align) + (self.lambda_unif * loss_unif)
        
        return {
            'overall': overall_loss,
            'ce_loss': loss_cls,
            'align_loss': loss_align,
            'unif_loss': loss_unif
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
