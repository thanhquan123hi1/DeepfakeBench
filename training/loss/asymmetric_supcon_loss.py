import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="asymmetric_contrastive")
class AsymmetricSupConLoss(AbstractLossClass):
    """
    Asymmetric Supervised Contrastive Loss for Deepfake Detection.

    Ý tưởng:
    - Real-Real: kéo gần bằng SupCon
    - Fake-Real: đẩy xa bằng cosine margin
    - Fake-Fake: bỏ qua, không kéo gần fake với fake

    Phù hợp cho Deepfake Detection vì:
    - Real thường có phân phối ổn định hơn.
    - Fake rất đa dạng, không nên ép toàn bộ fake gom thành một cụm.
    """

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        real_label=0,
        lambda_fake=0.1,
        fake_margin=0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.real_label = real_label
        self.lambda_fake = lambda_fake
        self.fake_margin = fake_margin

    def forward(self, features, labels):
        """
        Args:
            features: tensor [B, V, D] hoặc [B, D]
                B: batch size
                V: số view / augmentation view
                D: embedding dimension

            labels: tensor [B]
                real_label mặc định = 0
                fake label thường = 1

        Returns:
            loss scalar
        """

        device = features.device

        # Cho phép input dạng [B, D]
        if features.dim() == 2:
            features = features.unsqueeze(1)

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [B, V, D] or [B, D]."
            )

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")

        # Normalize để dot product = cosine similarity
        features = F.normalize(features, dim=-1)

        contrast_count = features.shape[1]

        # [B, V, D] -> [B * V, D]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown contrast_mode: {}".format(self.contrast_mode))

        # Label cho contrast feature: [B * V]
        labels_contrast = labels.repeat(contrast_count, 1).view(-1)

        # Label cho anchor feature
        if self.contrast_mode == "one":
            labels_anchor = labels.view(-1)
        else:
            labels_anchor = labels_contrast

        # ================================================================
        # 1. REAL-REAL SUPCON LOSS
        # ================================================================

        # Cosine / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        # Log-sum-exp trick chỉ dùng cho SupCon
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask bỏ self-contrast
        logits_mask = torch.ones_like(logits, device=device)

        if self.contrast_mode == "one":
            self_index = torch.arange(batch_size, device=device).view(-1, 1)
        else:
            self_index = torch.arange(batch_size * anchor_count, device=device).view(-1, 1)

        logits_mask = torch.scatter(
            logits_mask,
            1,
            self_index,
            0,
        )

        labels_anchor_col = labels_anchor.view(-1, 1)
        labels_contrast_row = labels_contrast.view(1, -1)

        same_class = torch.eq(labels_anchor_col, labels_contrast_row).float().to(device)

        real_anchor_mask = (labels_anchor == self.real_label).float().view(-1, 1)
        real_contrast_mask = (labels_contrast == self.real_label).float().view(1, -1)

        # Positive chỉ là real-real
        real_pos_mask = same_class * real_anchor_mask * real_contrast_mask
        real_pos_mask = real_pos_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        real_pos_count = real_pos_mask.sum(dim=1)

        safe_real_pos_count = torch.where(
            real_pos_count < 1e-6,
            torch.ones_like(real_pos_count),
            real_pos_count,
        )

        mean_log_prob_real_pos = (
            real_pos_mask * log_prob
        ).sum(dim=1) / safe_real_pos_count

        real_loss_per_anchor = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_real_pos

        valid_real_anchor = (
            (real_anchor_mask.view(-1) > 0) & (real_pos_count > 0)
        ).float()

        real_loss = (
            real_loss_per_anchor * valid_real_anchor
        ).sum() / (valid_real_anchor.sum() + 1e-8)

        # ================================================================
        # 2. FAKE-REAL REPULSION LOSS
        # ================================================================
        # Quan trọng:
        # Không dùng logits ở đây, vì logits đã chia temperature và trừ max.
        # Dùng cosine similarity gốc để margin có ý nghĩa hình học.

        cos_sim = torch.matmul(anchor_feature, contrast_feature.T)

        fake_anchor_mask = (labels_anchor != self.real_label).float().view(-1, 1)
        real_contrast_mask = (labels_contrast == self.real_label).float().view(1, -1)

        # Row = fake anchor, column = real contrast
        fake_real_mask = fake_anchor_mask * real_contrast_mask * logits_mask

        fake_pair_count = fake_real_mask.sum(dim=1)

        safe_fake_pair_count = torch.where(
            fake_pair_count < 1e-6,
            torch.ones_like(fake_pair_count),
            fake_pair_count,
        )

        # Margin-based repulsion:
        # Nếu cos(fake, real) > margin thì phạt.
        # Nếu cos(fake, real) <= margin thì không phạt.
        fake_loss_per_anchor = (
            fake_real_mask * F.relu(cos_sim - self.fake_margin)
        ).sum(dim=1) / safe_fake_pair_count

        valid_fake_anchor = (
            (fake_anchor_mask.view(-1) > 0) & (fake_pair_count > 0)
        ).float()

        fake_loss = (
            fake_loss_per_anchor * valid_fake_anchor
        ).sum() / (valid_fake_anchor.sum() + 1e-8)

        # ================================================================
        # TOTAL LOSS
        # ================================================================

        loss = real_loss + self.lambda_fake * fake_loss

        return loss
