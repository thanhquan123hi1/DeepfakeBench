import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo các đường dẫn import này khớp với cấu trúc thư mục của bạn
from model.clip.clip import load
from model.adapters.adapter import Adapter
from model.attn import RecAttnClip
from model.layer import PostClipProcess, MaskPostXrayProcess
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='forada')
class ForADADetector(AbstractDetector):
    def __init__(self, config=None):
        super(ForADADetector, self).__init__()
        self.config = config or {}
        
        # Thiết lập các tham số cho Model
        self.clip_name = self.config.get('clip_model_name', 'ViT-L/14')
        self.adapter_vit_name = self.config.get('vit_name', 'vit_tiny_patch16_224')
        self.num_quires = self.config.get('num_quires', 128)
        self.fusion_map = self.config.get('fusion_map', {1: 1, 2: 8, 3: 15})
        self.mlp_dim = self.config.get('mlp_dim', 256)
        self.mlp_out_dim = self.config.get('mlp_out_dim', 128)
        self.head_num = self.config.get('head_num', 16)
        
        # Lấy device từ config
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info(f"Loading ForADA with CLIP ({self.clip_name}) and Adapter ({self.adapter_vit_name})...")
        
        # 1. Khởi tạo CLIP backbone
        # Lưu ý: Sửa 'download_root' phù hợp với server của bạn hoặc lấy từ config
        download_root = self.config.get('clip_download_root', './weights')
        self.clip_model, self.processor = load(self.clip_name, device=self.device, download_root=download_root)
        
        # 2. Khởi tạo các Modules của ForADA
        self.adapter = Adapter(vit_name=self.adapter_vit_name, num_quires=self.num_quires, 
                               fusion_map=self.fusion_map, mlp_dim=self.mlp_dim,
                               mlp_out_dim=self.mlp_out_dim, head_num=self.head_num, device=self.device)
                               
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, self.num_quires, device=self.device)
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=self.num_quires).to(self.device)
        self.clip_post_process = PostClipProcess(num_quires=self.num_quires, embed_dim=768).to(self.device)

        # Trọng số tính Loss
        self.lambda_cls = self.config.get('lambda_cls', 10.0)
        self.lambda_mse = self.config.get('lambda_mse', 200.0)
        self.lambda_intra = self.config.get('lambda_intra', 20.0)
        self.lambda_clip = self.config.get('lambda_clip', 10.0)
        
        self.criterion_ce = nn.CrossEntropyLoss()
        
        # Theo dõi các biến như model gốc (để lưu test metrics nếu cần)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        self._freeze()

    def _freeze(self):
        """Đóng băng toàn bộ tham số của CLIP gốc theo thiết kế của ForADA"""
        count_frozen = 0
        for name, param in self.named_parameters():
            if 'clip_model' in name:
                param.requires_grad = False
                count_frozen += param.numel()
        logger.info(f"Frozen CLIP parameters: {count_frozen}")

    def forward(self, data_dict: dict, inference=False) -> dict:
        images = data_dict['image']
        
        # Tiền xử lý ảnh cho CLIP (Resize về 224x224)
        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )

        # Trích xuất đặc trưng CLIP
        clip_features = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())

        # Forward qua Adapter
        attn_biases, xray_preds, loss_adapter_intra = self.adapter(data_dict, clip_features, inference)
        
        # Forward qua RecAttnClip
        clip_output, loss_clip = self.rec_attn_clip(data_dict, clip_features, attn_biases[-1], inference, normalize=True)

        # Xử lý X-ray boundary mask
        if 'if_boundary' in data_dict:
            if_boundary = data_dict['if_boundary'].to(self.device)
            xray_preds = [self.masked_xray_post_process(xray_pred, if_boundary) for xray_pred in xray_preds]

        # Classification logits
        clip_cls_output = self.clip_post_process(clip_output.float()).squeeze()
        
        # Trường hợp batch size = 1, squeeze() có thể làm mất chiều batch, cần handle an toàn:
        if clip_cls_output.dim() == 1:
            clip_cls_output = clip_cls_output.unsqueeze(0)

        # Xác suất Class 1 (Fake)
        prob = torch.softmax(clip_cls_output, dim=1)[:, 1]

        # Đóng gói dữ liệu đầu ra
        pred_dict = {
            'cls': clip_cls_output,
            'prob': prob,
            'xray_pred': xray_preds[-1] if len(xray_preds) > 0 else None,
            'loss_intra': loss_adapter_intra,
            'loss_clip': loss_clip
        }

        if inference:
            self.prob.append(prob.detach().squeeze().cpu().numpy())
            self.label.append(data_dict['label'].detach().squeeze().cpu().numpy())
            _, prediction_class = torch.max(clip_cls_output, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

        return pred_dict

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        xray = data_dict.get('xray', None)

        pred = pred_dict['cls']
        xray_pred = pred_dict['xray_pred']
        loss_intra = pred_dict['loss_intra']
        loss_clip = pred_dict['loss_clip']

        # 1. Tính CrossEntropy Loss
        loss_cls = self.criterion_ce(pred.float(), label)

        loss_dict = {
            'loss_cls': loss_cls,
            'loss_intra': loss_intra if isinstance(loss_intra, torch.Tensor) else torch.tensor(0.0, device=self.device),
            'loss_clip': loss_clip if isinstance(loss_clip, torch.Tensor) else torch.tensor(0.0, device=self.device)
        }

        total_loss = self.lambda_cls * loss_cls

        # 2. Tính X-ray MSE Loss (nếu có dữ liệu X-ray)
        if xray is not None and xray_pred is not None:
            loss_mse = F.mse_loss(xray_pred.squeeze().float(), xray.squeeze().float())
            loss_dict['loss_xray'] = loss_mse
            total_loss += self.lambda_mse * loss_mse
        
        # 3. Tính tổng Loss
        if isinstance(loss_intra, torch.Tensor):
            total_loss += self.lambda_intra * loss_intra
        if isinstance(loss_clip, torch.Tensor):
            total_loss += self.lambda_clip * loss_clip

        loss_dict['overall'] = total_loss

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
