import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="asymmetric_contrastive")
class AsymmetricContrastiveLoss(AbstractLossClass):
    """
    Asymmetric contrastive loss

    Mục tiêu:
    - real-real: kéo gần
    - fake-real: đẩy xa
    - fake-fake: không kéo gần
    """

    def __init__(self, temperature=0.07, lambda_fake=1.0, real_label=0):
        super().__init__()
        self.temperature = temperature
        self.lambda_fake = lambda_fake
        self.real_label = real_label

    def forward(self, features, labels):
        """
        Args:
            features: tensor shape [B, V, D] hoặc [B, D]
            labels: tensor shape [B]
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)

        device = features.device
        B, V, D = features.shape

        features = F.normalize(features, dim=-1)
        features = features.reshape(B * V, D)

        labels = labels.view(B, 1).repeat(1, V).reshape(-1)

        sim = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(sim, device=device)
        logits_mask.fill_diagonal_(0)

        labels_row = labels.unsqueeze(0)
        labels_col = labels.unsqueeze(1)

        same_class = (labels_row == labels_col).float()

        real_mask = (labels == self.real_label).float()
        fake_mask = 1.0 - real_mask

        real_row = real_mask.unsqueeze(0)
        real_col = real_mask.unsqueeze(1)
        fake_row = fake_mask.unsqueeze(0)

        # positive chỉ là real-real
        pos_mask = same_class * real_row * real_col
        pos_mask = pos_mask * logits_mask

        # ===== real loss =====
        # dùng SupCon cho real anchors
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = pos_mask.sum(dim=1)
        real_loss = -((pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8))

        real_loss = (real_loss * real_mask).sum() / (real_mask.sum() + 1e-8)

        # ===== fake loss =====
        # fake chỉ repel với real
        fake_real_mask = fake_row * real_col * logits_mask

        fake_pair_count = fake_real_mask.sum(dim=1)
        fake_loss = ((fake_real_mask * sim).sum(dim=1) / (fake_pair_count + 1e-8))

        fake_loss = (fake_loss * fake_mask).sum() / (fake_mask.sum() + 1e-8)

        loss = real_loss + self.lambda_fake * fake_loss
        return loss
