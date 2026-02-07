# File: core/eupg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import einops
from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList, Instances

# --- Giữ nguyên các class phụ trợ (Không thay đổi) ---
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        device = x.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        pe = pe.permute(2, 0, 1)
        return pe.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class MultiScaleGenerator(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        self.up_p2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        self.p2_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p3_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p4_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        p4 = self.p4_out(x)
        p3_feat = self.up_p3(x)
        p3 = self.p3_out(p3_feat)
        p2_feat = self.up_p2(x)
        p2 = self.p2_out(p2_feat)
        return {"p2": p2, "p3": p3, "p4": p4}

class PromptEncoder(nn.Module):
    def __init__(self, in_channels, pooler_resolution, embed_dim=256, num_points=5):
        super().__init__()
        self.num_points = num_points
        self.embed_dim = embed_dim
        
        # 1. Downsample feature giống code tham khảo
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        reduced_resolution = pooler_resolution // 2 
        flatten_dim = in_channels * reduced_resolution * reduced_resolution
        
        # Output dim x2 để tách làm sin/cos (giả lập logic của code tham khảo)
        output_dim = embed_dim * num_points * 2 
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, in_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, output_dim) 
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x)
        
        # Generate raw points: [B, num_points * embed_dim * 2]
        points = self.mlp(x)
        
        # Reshape: [B, num_points, embed_dim * 2]
        points = points.view(-1, self.num_points, self.embed_dim * 2)
        
        # --- SIN/COS TRICK (Giống Anchor.py) ---
        # "Giả lập" Positional Encoding bằng cách trộn Sin và Feature
        # Logic: sin(features[even]) + features[odd]
        p_sin = torch.sin(points[..., 0::2]) 
        p_other = points[..., 1::2]
        
        # [B, num_points, embed_dim] - Visual Embedding đã được hình học hóa
        prompt_embeddings = p_sin + p_other
        
        return prompt_embeddings

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, pooler_resolution, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.flatten_dim = in_channels * pooler_resolution * pooler_resolution
        self.num_outputs = num_classes + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.num_outputs) 
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="linear")
        if self.conv.bias is not None: nn.init.constant_(self.conv.bias, 0)
    def forward(self, x):
        x = self.conv(x) 
        x = x.flatten(start_dim=1) 
        x = self.mlp(x)            
        return x

class BoxRegressionHead(nn.Module):
    def __init__(self, in_channels, pooler_resolution, box_dim=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.flatten_dim = in_channels * pooler_resolution * pooler_resolution
        
        # [NEW] Tách layer để dễ init
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.output_layer = nn.Linear(1024, box_dim) 
        
        self.mlp = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            self.fc2,
            nn.ReLU(inplace=True),
            self.output_layer
        )
        
        # Init weights cơ bản
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        
        # [FIX QUAN TRỌNG] Init lớp cuối cùng cực nhỏ để Loss ban đầu ~ 0
        nn.init.normal_(self.output_layer.weight, std=0.001)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x

class EUPG(nn.Module):
    def __init__(self, cfg: CfgNode, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        in_features = list(input_shape.keys()) 
        self.in_feature_name = in_features[0]
        in_channels = input_shape[self.in_feature_name].channels
        
        self.channel_attention = ChannelAttention(in_channels)
        self.multiscale_gen = MultiScaleGenerator(out_channels=in_channels)
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=in_channels // 2)
        
        rpn_in_features = ["p2", "p3", "p4"]
        rpn_input_shapes = {
            "p2": ShapeSpec(channels=in_channels, stride=4),
            "p3": ShapeSpec(channels=in_channels, stride=8),
            "p4": ShapeSpec(channels=in_channels, stride=16),
        }
        
        anchor_input_shapes = [rpn_input_shapes[f] for f in rpn_in_features]
        anchor_sizes = [[8, 16, 32], [32, 64, 128], [64, 128, 256]]
        
        self.anchor_generator = DefaultAnchorGenerator(
            cfg, 
            input_shape=anchor_input_shapes, 
            sizes=anchor_sizes, 
            aspect_ratios=[[0.5, 1.0, 2.0]] * len(rpn_in_features)
        )
        
        self.rpn_head = StandardRPNHead(
            in_channels=in_channels,
            num_anchors=self.anchor_generator.num_cell_anchors[0],
            box_dim=4
        )
        
        rpn_cfg = cfg.clone()
        rpn_cfg.defrost()
        rpn_cfg.MODEL.RPN.IN_FEATURES = rpn_in_features
        rpn_cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_sizes 
        rpn_cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        rpn_cfg.freeze()
        
        self.rpn = RPN(
            rpn_cfg, 
            input_shape=rpn_input_shapes, 
            in_features=rpn_in_features, 
            head=self.rpn_head,
            anchor_generator=self.anchor_generator,
            anchor_matcher=Matcher(
                cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
            ),
            box2box_transform=Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            positive_fraction=cfg.MODEL.RPN.POSITIVE_FRACTION,
            pre_nms_topk=(cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST),
            post_nms_topk=(cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        self.pooler_resolution = 14
        self.pooler = ROIPooler(
            output_size=(self.pooler_resolution, self.pooler_resolution),
            scales=[1.0 / rpn_input_shapes[k].stride for k in rpn_in_features],
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        
        # [UPDATE] Sử dụng PromptEncoder mới (5 điểm, Sin/Cos)
        self.prompt_head = PromptEncoder(
            in_channels=in_channels, 
            pooler_resolution=self.pooler_resolution,
            num_points=5 
        )
        
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cls_head = ClassificationHead(
            in_channels=in_channels,
            pooler_resolution=self.pooler_resolution,
            num_classes=num_classes
        )
        self.box_head = BoxRegressionHead(
            in_channels=in_channels, pooler_resolution=self.pooler_resolution
        )

    def forward(self, images: ImageList, features: Dict[str, torch.Tensor], gt_instances: List[Instances] = None):
        feat = features[self.in_feature_name]
        feat_attended = self.channel_attention(feat)
        multiscale_features = self.multiscale_gen(feat_attended)
        
        # 1. RPN sinh Proposals
        proposals, losses = self.rpn(images, multiscale_features, gt_instances)
        
        # 2. [FIX] TRỘN GT VÀO PROPOSALS
        if self.training and gt_instances is not None:
            proposals_with_gt = []
            for p, gt in zip(proposals, gt_instances):
                # Tạo Instances giả lập từ GT
                gt_as_prop = Instances(p.image_size)
                gt_as_prop.proposal_boxes = gt.gt_boxes
                
                # --- [SỬA LỖI TẠI ĐÂY] ---
                # Thay vì dùng p.device (gây lỗi), ta lấy device từ box hoặc features
                # Cách 1: Lấy từ feature map (chắc chắn đúng device)
                device = feat.device 
                
                # Hoặc Cách 2: Lấy từ proposal boxes
                # device = p.proposal_boxes.device
                
                # Gán điểm số cao tuyệt đối cho GT
                gt_as_prop.objectness_logits = torch.ones(len(gt), device=device) * 10.0 
                # -------------------------
                
                # Nối: [RPN Proposals, GT Boxes]
                merged_prop = Instances.cat([p, gt_as_prop])
                proposals_with_gt.append(merged_prop)
            
            proposals = proposals_with_gt
            
        # 3. Lấy Box từ danh sách đã trộn
        proposal_boxes = [p.proposal_boxes for p in proposals]
        
        features_with_pe = []
        feature_levels = ["p2", "p3", "p4"]
        for level in feature_levels:
            f_img = multiscale_features[level]
            f_pe = self.pe_layer(f_img)
            features_with_pe.append(f_img + f_pe)
            
        # 4. RoI Align
        roi_features = self.pooler(features_with_pe, proposal_boxes)
        
        # 5. Heads
        prompt_embeddings = self.prompt_head(roi_features) 
        pred_class_logits = self.cls_head(roi_features)
        pred_proposal_deltas = self.box_head(roi_features)

        return proposals, losses, prompt_embeddings, pred_class_logits, pred_proposal_deltas