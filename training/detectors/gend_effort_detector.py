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
# 1. HELPER FUNCTIONS (STABILIZED)
# =========================================================================

def slerp(val, low, high):
    """
    Spherical Linear Interpolation (Numerically Stable Version)
    """
    # Normalize inputs
    low_norm = F.normalize(low, p=2, dim=1, eps=1e-6)
    high_norm = F.normalize(high, p=2, dim=1, eps=1e-6)
    
    # Compute cosine similarity with clamping to avoid acos(>1) error
    dot_prod = (low_norm * high_norm).sum(dim=1, keepdim=True)
    dot_prod = dot_prod.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    
    omega = torch.acos(dot_prod)
    so = torch.sin(omega)
    
    # CRITICAL FIX: Add epsilon to denominator to prevent NaN gradient in the 'else' branch
    # torch.where calculates gradients for BOTH branches.
    so_safe = so + 1e-8
    
    res = (torch.sin((1.0 - val) * omega) / so_safe) * low + (torch.sin(val * omega) / so_safe) * high
    
    # Use Linear interpolation if vectors are too close (sin(omega) ~ 0)
    return torch.where(so.abs() < 1e-4, (1.0 - val) * low + val * high, res)

def alignment_loss(embeddings, labels, alpha=2):
    """GenD Alignment Loss (Stable)"""
    if embeddings.size(0) < 2: 
        return torch.tensor(0.0, device=embeddings.device)
    
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    
    if positive_indices.numel() == 0: 
        return torch.tensor(0.0, device=embeddings.device)
    
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    
    # CRITICAL FIX: Replace norm().pow(2) with pow(2).sum()
    # Gradient of norm() at 0 is unstable (NaN). 
    # (x-y)^2 sum is stable (gradient is 2(x-y)).
    if alpha == 2:
        return (x - y).pow(2).sum(dim=1).mean()
    else:
        # Fallback for alpha != 2: add epsilon inside norm
        return (x - y).norm(p=2, dim=1).add(1e-8).pow(alpha).mean()

def uniformity_loss(x, t=2):
    """GenD Uniformity Loss"""
    if x.size(0) < 2: 
        return torch.tensor(0.0, device=x.device)
    # pdist computes pairwise euclidean distance
    # We use stable exponential logic
    sq_dist = torch.pdist(x, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()

# =========================================================================
# 2. SVD RESIDUAL MODULE (STABILIZED)
# =========================================================================

class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Main Weight (Buffer)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        self.register_buffer('weight_original_fnorm', torch.norm(self.weight_main.data, p='fro'))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # Residuals
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None
        
        self.register_buffer('U_r', None)
        self.register_buffer('V_r', None)

    def forward(self, x):
        weight = self.weight_main
        if self.S_residual is not None:
            # W = W_fixed + U * diag(S) * V
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = weight + residual_weight
        
        return F.linear(x, weight, self.bias)

    def compute_orthogonal_loss(self):
        """Stable Orthogonal Loss"""
        if self.S_residual is not None and self.U_r is not None:
            U_cat = torch.cat((self.U_r, self.U_residual), dim=1) 
            V_cat = torch.cat((self.V_r, self.V_residual), dim=0)

            # Gram Matrices
            gram_U = U_cat.t() @ U_cat
            gram_V = V_cat @ V_cat.t()

            I_U = torch.eye(gram_U.size(0), device=gram_U.device)
            I_V = torch.eye(gram_V.size(0), device=gram_V.device)

            # CRITICAL FIX: Use Mean Squared Error logic directly instead of norm(p='fro')**2
            # Avoids sqrt gradient instability
            loss_u = ((gram_U - I_U) ** 2).sum()
            loss_v = ((gram_V - I_V) ** 2).sum()

            return 0.5 * (loss_u + loss_v)
            
        return torch.tensor(0.0, device=self.weight_main.device)

    def compute_keepsv_loss(self):
        if self.S_residual is not None:
            weight_current = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight_current_fnorm = torch.norm(weight_current, p='fro')
            
            # Use MSE style
            return (weight_current_fnorm**2 - self.weight_original_fnorm**2).abs()
        return torch.tensor(0.0, device=self.weight_main.device)

# --- SVD Injection Logic ---
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        new_module = SVDResidualLinear(
            module.in_features, module.out_features, r, 
            bias=(module.bias is not None), 
            init_weight=module.weight.data.clone()
        )
        if module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        W = module.weight.data.float()
        device = W.device
        new_module.weight_original_fnorm = torch.norm(W, p='fro')
        
        # SVD
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError:
            # Fallback to CPU if GPU SVD fails
            W_cpu = W.cpu()
            U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
            U, S, Vh = U.to(device), S.to(device), Vh.to(device)

        rank = min(r, len(S))
        
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        
        new_module.weight_main.data.copy_(U_r @ torch.diag(S_r) @ Vh_r)
        new_module.U_r = U_r
        new_module.V_r = Vh_r

        if len(S) > rank:
            new_module.U_residual = nn.Parameter(U[:, rank:].clone())
            new_module.S_residual = nn.Parameter(S[rank:].clone())
            new_module.V_residual = nn.Parameter(Vh[rank:, :].clone())

        return new_module.to(device)
    return module

def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        if 'self_attn' in name or 'attn' in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    path = sub_name.split('.')
                    parent = module
                    for node in path[:-1]:
                        parent = getattr(parent, node)
                    target_name = path[-1]
                    if hasattr(parent, target_name):
                        setattr(parent, target_name, replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    return model

# =========================================================================
# 3. DETECTOR CLASS
# =========================================================================

@DETECTOR.register_module(module_name='gend_effort')
class GenDEffortDetector(AbstractDetector):
    def __init__(self, config=None):
        super(GenDEffortDetector, self).__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        
        self.build_loss(config)
        
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        self._setup_trainable_params()

    def build_backbone(self, config):
        logger.info("Loading CLIP ViT-L/14...")
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        except Exception:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)

        backbone = clip_model.vision_model
        
        svd_rank = config.get('svd_rank', 1023)
        logger.info(f"Applying Effort SVD (rank={svd_rank}) to backbone...")
        backbone = apply_svd_residual_to_self_attn(backbone, svd_rank)
        
        return backbone

    def build_loss(self, config):
        self.lambda_align = config.get('lambda_align', 1.0)
        self.lambda_unif = config.get('lambda_unif', 1.0)
        self.lambda_ortho = config.get('lambda_ortho', 1.0)
        self.lambda_ksv = config.get('lambda_ksv', 1.0)
        self.loss_func = nn.CrossEntropyLoss()

    def _setup_trainable_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        count = 0
        # Head
        for p in self.head.parameters():
            p.requires_grad = True
            count += p.numel()
            
        # LayerNorm (GenD)
        for name, p in self.backbone.named_parameters():
            if 'layer_norm' in name or 'layernorm' in name:
                p.requires_grad = True
                count += p.numel()

        # SVD Residuals (Effort)
        for name, p in self.backbone.named_parameters():
            if any(k in name for k in ['S_residual', 'U_residual', 'V_residual']):
                p.requires_grad = True
                count += p.numel()
        
        # Freeze SVD Bias if present
        for m in self.backbone.modules():
            if isinstance(m, SVDResidualLinear) and m.bias is not None:
                m.bias.requires_grad = False

        logger.info(f"GenD-Effort Initialized. Trainable params: {count}")

    def features(self, data_dict: dict) -> torch.tensor:
        out = self.backbone(data_dict['image'])['pooler_output']
        # Stable Normalize
        return F.normalize(out, p=2, dim=1, eps=1e-6)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        feat = pred_dict['feat']

        loss_ce = self.loss_func(pred, label)
        
        loss_align = torch.tensor(0.0, device=pred.device)
        loss_unif = torch.tensor(0.0, device=pred.device)
        loss_ortho = torch.tensor(0.0, device=pred.device)
        loss_ksv = torch.tensor(0.0, device=pred.device)

        if self.training:
            # Slerp Augmentation
            feat_aug_list = []
            label_aug_list = []
            
            unique_labels = torch.unique(label)
            for l in unique_labels:
                mask = (label == l)
                class_feats = feat[mask]
                if class_feats.size(0) >= 2:
                    idx = torch.randperm(class_feats.size(0), device=feat.device)
                    lam = torch.rand(class_feats.size(0), 1, device=feat.device)
                    
                    feat_interp = slerp(lam, class_feats, class_feats[idx])
                    
                    feat_aug_list.append(feat_interp)
                    label_aug_list.append(label[mask])
            
            if len(feat_aug_list) > 0:
                feat_aug = torch.cat(feat_aug_list, dim=0)
                label_aug = torch.cat(label_aug_list, dim=0)
                feat_total = torch.cat([feat, feat_aug], dim=0)
                label_total = torch.cat([label, label_aug], dim=0)
                
                loss_align = alignment_loss(feat_total, label_total)
                loss_unif = uniformity_loss(feat_total)
            else:
                loss_align = alignment_loss(feat, label)
                loss_unif = uniformity_loss(feat)

            ortho_list = []
            ksv_list = []
            for m in self.backbone.modules():
                if isinstance(m, SVDResidualLinear):
                    ortho_list.append(m.compute_orthogonal_loss())
                    ksv_list.append(m.compute_keepsv_loss())
            
            if len(ortho_list) > 0:
                loss_ortho = torch.mean(torch.stack(ortho_list))
                loss_ksv = torch.mean(torch.stack(ksv_list))

        # Check for NaNs before summation (Optional debugging)
        if torch.isnan(loss_align) or torch.isnan(loss_unif) or torch.isnan(loss_ortho):
            logger.warning("NaN detected in aux losses! Zeroing them to prevent crash.")
            loss_align = torch.nan_to_num(loss_align)
            loss_unif = torch.nan_to_num(loss_unif)
            loss_ortho = torch.nan_to_num(loss_ortho)

        total_loss = loss_ce + \
                     (self.lambda_align * loss_align) + \
                     (self.lambda_unif * loss_unif) + \
                     (self.lambda_ortho * loss_ortho) + \
                     (self.lambda_ksv * loss_ksv)

        loss_dict = {
            'overall': total_loss,
            'loss_ce': loss_ce,
            'loss_align': loss_align,
            'loss_unif': loss_unif,
            'loss_ortho': loss_ortho,
            'loss_ksv': loss_ksv
        }
        
        with torch.no_grad():
            mask_real = (label == 0)
            mask_fake = (label == 1)
            loss_dict['real_loss'] = self.loss_func(pred[mask_real], label[mask_real]) if mask_real.sum() > 0 else torch.tensor(0.0)
            loss_dict['fake_loss'] = self.loss_func(pred[mask_fake], label[mask_fake]) if mask_fake.sum() > 0 else torch.tensor(0.0)

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}