# File: core/uwsam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from detectron2.config import CfgNode
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import ImageList, Instances, BitMasks, PolygonMasks
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from .eupg import EUPG
from .utils import PositionEmbeddingRandom

class UWSAMMaskHead(nn.Module):
    """
    Tương đương class USISPrompterAnchorMaskHead trong anchor.py
    Sinh prompt từ RoI features và decode mask.
    """
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Config khớp với anchor.py
        roi_feat_size = 14
        per_pointset_point = 5
        self.with_sincos = True
        num_sincos = 2 if self.with_sincos else 1
        
        # 1. Point Embedding Network (Prompt Generator)
        # Conv -> Flatten -> Linear -> Linear
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels * (roi_feat_size // 2) ** 2, in_channels), # stride 2 nên size giảm 1 nửa
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)
        )
        
        # 2. SAM Components
        embed_dim = 256
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=embed_dim, mlp_dim=2048, num_heads=8)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3, transformer=self.transformer, transformer_dim=embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
        )
        # Dummy prompt encoder để lấy embedding "no_mask"
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16,
        )
        
        if cfg.MODEL.SAM.CHECKPOINT:
            self._load_sam_weights(cfg.MODEL.SAM.CHECKPOINT)
            
        self.no_mask_embed = self.prompt_encoder.no_mask_embed
        self.per_pointset_point = per_pointset_point

    def _load_sam_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        decoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith("mask_decoder."):
                decoder_dict[k.replace("mask_decoder.", "")] = v
        self.mask_decoder.load_state_dict(decoder_dict, strict=True)
        # Prompt encoder weights (chỉ cần no_mask_embed)
        self.prompt_encoder.load_state_dict(
            {k.replace("prompt_encoder.", ""): v for k, v in state_dict.items() if "prompt_encoder" in k}, 
            strict=False
        )

    def forward(self, mask_feats, image_embeddings, image_pe):
        """
        Args:
            mask_feats: [N_RoI, 256, 14, 14]
            image_embeddings: [N_RoI, 256, 64, 64] (Đã repeat theo số roi mỗi ảnh)
            image_pe: [N_RoI, 256, 64, 64]
        """
        roi_bs = mask_feats.shape[0]
        
        # 1. Generate Visual Prompts (Points)
        # [N, C, 7, 7] -> [N, num_points * C * 2]
        point_embeddings = self.point_emb(mask_feats)
        
        # Reshape: [N, num_points, C]
        point_embeddings = einops.rearrange(
            point_embeddings, 'b (n c) -> b n c', n=self.per_pointset_point
        )
        
        # Sin/Cos Encoding (Logic từ anchor.py)
        if self.with_sincos:
            # sin(even) + odd
            point_embeddings = torch.sin(point_embeddings[..., ::2]) + point_embeddings[..., 1::2]
            
        # [N, 1, num_points, C] -> Sparse Embeddings
        sparse_embeddings = point_embeddings.unsqueeze(1)
        
        # 2. Dense Embeddings (No Mask)
        # [N, 256, 64, 64]
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            roi_bs, -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )
        
        # 3. Decode
        low_res_masks, iou_preds = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return low_res_masks, iou_preds

@META_ARCH_REGISTRY.register()
class UWSAM(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        
        # 1. EUPG (Proposal Generator + Feature Adapter)
        self.proposal_generator = EUPG(cfg, self.backbone.output_shape())
        
        # 2. RoI Heads Setup
        # Box Head (Standard Detectron2)
        pooler_resolution = 14 # Khớp với roi_feat_size
        self.box_pooler = ROIPooler(
            output_size=(7, 7), # Box head thường dùng 7x7
            scales=[1.0 / 16], # Input feature stride 16
            sampling_ratio=0,
            pooler_type="ROIAlignV2"
        )
        
        in_channels = 256
        self.box_head = FastRCNNConvFCHead(
            ShapeSpec(channels=in_channels, height=7, width=7),
            conv_dims=[], fc_dims=[1024, 1024]
        )
        self.box_predictor = FastRCNNOutputLayers(
            cfg, self.box_head.output_shape
        )
        
        # 3. Mask Head (Custom UWSAM)
        self.mask_pooler = ROIPooler(
            output_size=(14, 14), # Mask head cần 14x14
            scales=[1.0 / 16],
            sampling_ratio=0,
            pooler_type="ROIAlignV2"
        )
        self.mask_head = UWSAMMaskHead(cfg, in_channels)
        
        # Utils
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=128) # 128*2 = 256
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(img - self.pixel_mean) / self.pixel_std for img in images]
        return ImageList.from_tensors(images, self.backbone.size_divisibility)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        
        # 1. Backbone
        with torch.no_grad():
            features = self.backbone(images.tensor)
        
        # 2. EUPG (Attention -> MultiScale -> RPN)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            
        proposals, proposal_losses, feat_attended = self.proposal_generator(images, features, gt_instances)
        
        # 3. RoI Heads Logic
        # Box Branch
        if self.training:
            # Match proposals to GT
            targets = self.box_predictor.label_and_sample_proposals(proposals, gt_instances) 
            proposals = targets # update proposals with gt fields
        
        # Lấy boxes
        proposal_boxes = [p.proposal_boxes for p in proposals]
        
        # Pool features cho Box Head (Branch 1: feat_attended)
        box_features = self.box_pooler([feat_attended], proposal_boxes)
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        
        losses = {}
        losses.update(proposal_losses)
        
        if self.training:
            loss_cls, loss_box_reg = self.box_predictor.losses(
                (pred_class_logits, pred_proposal_deltas), proposals
            )
            losses.update({"loss_cls": loss_cls, "loss_box_reg": loss_box_reg})
            
            # 4. Mask Branch (Training)
            # Filter positive proposals for mask training
            pos_mask_features, pos_instances = self._get_mask_training_samples(
                feat_attended, proposals
            )
            
            if len(pos_instances) > 0:
                # Prepare Image Embeddings & PE for SAM
                image_embeddings_raw = features["feature_map"] # [B, 256, 64, 64]
                
                # Repeat image embeddings cho mỗi ROI thuộc ảnh đó
                img_inds = []
                for idx, p in enumerate(pos_instances):
                    img_inds.extend([idx] * len(p))
                img_inds = torch.tensor(img_inds, device=self.device)
                
                # [N_pos, 256, 64, 64]
                image_embeddings_roi = image_embeddings_raw[img_inds]
                
                # Generate PE [1, 256, 64, 64] -> Repeat
                image_pe = self.pe_layer((64, 64), self.device)
                image_pe_roi = image_pe.repeat(len(img_inds), 1, 1, 1)
                
                # Forward Mask Head
                pred_masks, pred_ious = self.mask_head(
                    pos_mask_features, image_embeddings_roi, image_pe_roi
                )
                
                # Calculate Loss
                loss_mask = self._mask_loss(pred_masks, pred_ious, pos_instances)
                losses.update(loss_mask)
            else:
                losses["loss_mask"] = sum(p.sum() for p in self.mask_head.parameters()) * 0.0
                
            return losses
        else:
            # Inference logic
            results = self._inference(
                box_features, pred_class_logits, pred_proposal_deltas, 
                proposals, feat_attended, features["feature_map"]
            )
            return results

    def _get_mask_training_samples(self, features, proposals):
        # Lọc các proposal dương tính
        pos_proposals = []
        for p in proposals:
            pos_inds = torch.nonzero(p.gt_classes != self.box_predictor.num_classes).squeeze(1)
            pos_proposals.append(p[pos_inds])
            
        # RoI Align cho mask (Size 14x14)
        pos_boxes = [p.proposal_boxes for p in pos_proposals]
        mask_features = self.mask_pooler([features], pos_boxes)
        
        return mask_features, pos_proposals

    def _mask_loss(self, pred_masks, pred_ious, proposals):
        # Lấy GT Masks
        gt_masks = []
        for p in proposals:
            gt_masks.append(p.gt_masks.tensor)
        gt_masks = torch.cat(gt_masks).to(self.device).float()
        
        # Upsample pred_mask to 1024 (ảnh gốc) or 256 (GT)?
        # Detectron2 thường tính loss trên size 28x28 hoặc size ảnh gốc.
        # Ở đây output SAM là 256x256 (low res).
        # Ta resize pred về 1024x1024 để khớp GT (hoặc resize GT về 256)
        # Cách chuẩn: Resize Pred lên 1024x1024
        
        pred_masks_up = F.interpolate(
            pred_masks, size=(1024, 1024), mode="bilinear", align_corners=False
        ).squeeze(1)
        
        # Focal Loss + Dice Loss logic (như cũ)
        loss_focal = F.binary_cross_entropy_with_logits(pred_masks_up, gt_masks, reduction='mean')
        
        # IoU Loss (Predicted IoU vs Real IoU)
        with torch.no_grad():
            pred_binary = (pred_masks_up > 0).float()
            inter = (pred_binary * gt_masks).sum((1, 2))
            union = pred_binary.sum((1, 2)) + gt_masks.sum((1, 2)) - inter
            true_ious = (inter + 1e-6) / (union + 1e-6)
            
        loss_iou = F.mse_loss(pred_ious.flatten(), true_ious)
        
        return {"loss_mask": loss_focal * 20.0 + loss_iou}

    def _inference(self, box_features, class_logits, box_deltas, proposals, feat_attended, image_embeddings_raw):
        # 1. Apply Box Deltas & NMS
        results = self.box_predictor.inference(
            (class_logits, box_deltas), proposals
        )
        
        # 2. Mask Inference cho các box còn lại
        final_results = []
        for i, (res, img_emb_i) in enumerate(zip(results, image_embeddings_raw)):
            if len(res) == 0:
                final_results.append(res)
                continue
                
            # RoI Align cho mask
            mask_boxes = [res.pred_boxes]
            mask_features = self.mask_pooler([feat_attended[i:i+1]], mask_boxes)
            
            # Prepare Embeddings
            img_emb_roi = img_emb_i.unsqueeze(0).repeat(len(res), 1, 1, 1)
            image_pe = self.pe_layer((64, 64), self.device).repeat(len(res), 1, 1, 1)
            
            # Predict
            pred_masks, _ = self.mask_head(mask_features, img_emb_roi, image_pe)
            
            # Resize & Threshold
            pred_masks_up = F.interpolate(
                pred_masks, size=(1024, 1024), mode="bilinear", align_corners=False
            ).squeeze(1)
            pred_masks_bin = pred_masks_up > 0.0
            
            res.pred_masks = pred_masks_bin[:, None, :, :] # [N, 1, H, W]
            final_results.append(res)
            
        return final_results