import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig, PreTrainedModel, PretrainedConfig
from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train

# ==========================================
# 1. CẤU HÌNH VÀ ENCODER (KHÔNG ĐƯỢC LẶP BIẾN)
# ==========================================

class GenDConfig(PretrainedConfig):
    model_type = "GenD" 
    def __init__(self, backbone="openai/clip-vit-large-patch14", head="LinearNorm", **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head

class CLIPEncoder(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        # SỬA TẠI ĐÂY: Thay vì bọc CLIPVisionModel vào self.vision_model, 
        # ta load thẳng các layer của nó ra để tránh cấu trúc lồng nhau.
        full_vision_model = CLIPVisionModel.from_pretrained(backbone_name)
        self.embeddings = full_vision_model.vision_model.embeddings
        self.pre_layrnorm = full_vision_model.vision_model.pre_layrnorm
        self.encoder = full_vision_model.vision_model.encoder
        self.post_layernorm = full_vision_model.vision_model.post_layernorm
        self.config = full_vision_model.config

    def forward(self, x):
        # Tái tạo lại luồng forward của CLIP Vision
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        encoder_outputs = self.encoder(x)
        pooled_output = encoder_outputs[0][:, 0, :] # Lấy CLS token
        pooled_output = self.post_layernorm(pooled_output)
        return pooled_output

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        if self.use_norm:
            x = F.normalize(x, p=2, dim=1)
        return self.linear(x)

# ==========================================
# 2. CLASS GEND (CẤU TRÚC PHẲNG TUYỆT ĐỐI)
# ==========================================

class GenD(PreTrainedModel):
    config_class = GenDConfig

    def __init__(self, config):
        super().__init__(config)
        # SỬA TẠI ĐÂY: Sử dụng trực tiếp CLIPVisionModel để Transformers tự map keys
        self.feature_extractor = CLIPVisionModel.from_pretrained(config.backbone)
        
        features_dim = self.feature_extractor.config.hidden_size
        
        head_type = getattr(config, 'head', 'LinearNorm')
        if head_type == "LinearNorm":
            self.model = LinearProbe(features_dim, 2, use_norm=True)
        else:
            self.model = LinearProbe(features_dim, 2, use_norm=False)

    def forward(self, inputs):
        # CLIPVisionModel của HF trả về pooler_output mặc định
        outputs = self.feature_extractor(pixel_values=inputs)
        features = outputs.pooler_output
        return self.model(features)

# ==========================================
# 3. DETECTOR WRAPPER (DEEPFAKEBENCH)
# ==========================================

@DETECTOR.register_module(module_name='gend')
class GenDDetector(nn.Module):
    def __init__(self, config=None):
        super(GenDDetector, self).__init__()
        self.config = config or {}
        
        print("Đang nạp mô hình GenD từ yermandy/GenD_CLIP_L_14...")
        self.gend = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")
        
        self._setup_training_params()
        
        self.lambda_align = self.config.get('lambda_align', 0.1)
        self.lambda_unif = self.config.get('lambda_unif', 0.5)
        self.loss_ce = nn.CrossEntropyLoss()

    def _setup_training_params(self):
        for param in self.gend.parameters():
            param.requires_grad = False
        
        for name, param in self.gend.named_parameters():
            # Mở khóa LayerNorm (LN-tuning)
            if 'layer_norm' in name or 'layernorm' in name or 'model.linear' in name:
                param.requires_grad = True

    def forward(self, data_dict, inference=False):
        # GenD hoạt động trên từng frame đơn lẻ
        output = self.gend(data_dict['image'])
        prob = torch.softmax(output, dim=1)[:, 1]
        
        # Lấy features để tính loss nếu cần
        raw_features = self.gend.feature_extractor(data_dict['image']).pooler_output
        norm_features = F.normalize(raw_features, p=2, dim=1)
        
        return {
            'cls': output, 
            'prob': prob, 
            'feat': raw_features, 
            'feat_norm': norm_features
        }

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        feat_norm = pred_dict['feat_norm']
        
        loss_cls = self.loss_ce(pred, label)
        # Alignment và Uniformity losses giúp tăng cường khả năng tổng quát
        loss_align = alignment_loss(feat_norm, label) if self.training else torch.tensor(0.0, device=pred.device)
        loss_unif = uniformity_loss(feat_norm) if self.training else torch.tensor(0.0, device=pred.device)
        
        overall_loss = loss_cls + (self.lambda_align * loss_align) + (self.lambda_unif * loss_unif)
        return {'overall': overall_loss, 'ce_loss': loss_cls, 'align_loss': loss_align, 'unif_loss': loss_unif}

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

# ==========================================
# 4. HÀM LOSS PHỤ TRỢ
# ==========================================

def alignment_loss(embeddings, labels, alpha=2):
    # Thu hẹp cụm tính năng của cùng một lớp trên mặt cầu
    if embeddings.size(0) < 2: return torch.tensor(0.0, device=embeddings.device)
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0: return torch.tensor(0.0, device=embeddings.device)
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity_loss(x, t=2):
    # Khuyến khích tính năng phân bổ đều để tránh shortcut learning
    if x.size(0) < 2: return torch.tensor(0.0, device=x.device)
    sq_dist = torch.pdist(x, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().clamp(min=1e-6).log()
