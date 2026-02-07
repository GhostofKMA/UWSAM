# File: core/uwsam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import CfgNode
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import ImageList, Instances, BitMasks, PolygonMasks, pairwise_iou, Boxes
from detectron2.modeling.matcher import Matcher 
from detectron2.modeling.box_regression import Box2BoxTransform
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

# Import components
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from .eupg import EUPG, PositionEmbeddingRandom

# --- 1. Sửa Criterion: Gộp loss thành 'loss_mask' ---
class UWSAMCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.dice_weight = 1.0
        self.focal_weight = 20.0 
        self.iou_weight = 1.0
        self.smooth = 1e-6 

    def compute_dice_loss(self, inputs, targets):
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        loss = 1 - (numerator + self.smooth) / (denominator + self.smooth)
        return loss.mean()

    def compute_iou_loss(self, pred_ious, true_ious):
        return F.mse_loss(pred_ious, true_ious)

    def forward(self, pred_masks, gt_masks, pred_iou_scores):
        if pred_masks is None or pred_masks.numel() == 0:
            device = pred_masks.device if pred_masks is not None else torch.device("cpu")
            zero = torch.tensor(0., device=device, requires_grad=True)
            return {"loss_mask": zero} # Chỉ trả về 1 key duy nhất

        loss_focal = sigmoid_focal_loss_jit(
            pred_masks, gt_masks, 
            alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean"
        )
        loss_dice = self.compute_dice_loss(pred_masks, gt_masks)

        with torch.no_grad():
            pred_masks_binary = (pred_masks.sigmoid() > 0.5).float()
            intersection = (pred_masks_binary * gt_masks).sum(dim=(1, 2))
            union = pred_masks_binary.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2)) - intersection
            true_ious = (intersection + 1e-6) / (union + 1e-6)
        
        loss_iou = self.compute_iou_loss(pred_iou_scores, true_ious)

        # GỘP LẠI THÀNH MỘT
        total_loss = (loss_focal * self.focal_weight + 
                      loss_dice * self.dice_weight + 
                      loss_iou * self.iou_weight)
        
        return {"loss_mask": total_loss}

@META_ARCH_REGISTRY.register()
class UWSAM(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = EUPG(cfg, self.backbone.output_shape())
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        # Loss cho 2nd stage
        self.loss_cls = nn.CrossEntropyLoss(reduction="mean")
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        embed_dim = 256
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=embed_dim, mlp_dim=2048, num_heads=8)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3, transformer=self.transformer, transformer_dim=embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16,
        )
        
        if cfg.MODEL.SAM.CHECKPOINT:
            self._load_sam_decoder_weights(cfg.MODEL.SAM.CHECKPOINT)
            
        for param in self.mask_decoder.parameters(): param.requires_grad = True 
        for param in self.prompt_encoder.parameters(): param.requires_grad = False
        for param in self.backbone.parameters(): param.requires_grad = False
            
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        self.matcher = Matcher(thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False)
        self.criterion = UWSAMCriterion(cfg)

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(self.device))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(self.device))
        self.to(self.device)

    def _load_sam_decoder_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        decoder_dict = {}
        prompt_dict = {}
        for k, v in state_dict.items():
            if k.startswith("mask_decoder."):
                decoder_dict[k.replace("mask_decoder.", "")] = v
            elif k.startswith("prompt_encoder."):
                prompt_dict[k.replace("prompt_encoder.", "")] = v
        self.mask_decoder.load_state_dict(decoder_dict, strict=True)
        self.prompt_encoder.load_state_dict(prompt_dict, strict=True)

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(img - self.pixel_mean) / self.pixel_std for img in images]
        return ImageList.from_tensors(images, self.backbone.size_divisibility)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        with torch.no_grad():
            features = self.backbone(images.tensor)
        image_embeddings = features["feature_map"]
        if isinstance(image_embeddings, list): image_embeddings = image_embeddings[0]
        if isinstance(image_embeddings, torch.Tensor): image_embeddings = image_embeddings.detach()
        else: image_embeddings = [x.detach() for x in image_embeddings]

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            
        proposals, proposal_losses, all_visual_prompts, pred_class_logits, pred_proposal_deltas = self.proposal_generator(images, features, gt_instances)
        
        num_proposals_per_img = [len(p) for p in proposals]
        visual_prompts_split = all_visual_prompts.split(num_proposals_per_img)
        class_logits_split = pred_class_logits.split(num_proposals_per_img)
        deltas_split = pred_proposal_deltas.split(num_proposals_per_img)

        # ---------------- TRAINING ----------------
        if self.training:
            losses = {}
            losses.update(proposal_losses) 
            
            mask_loss_acc = torch.tensor(0., device=self.device)
            loss_cls_acc = torch.tensor(0., device=self.device)
            loss_box_reg_acc = torch.tensor(0., device=self.device)
            
            total_pos = 0
            total_samples = 0

            for i, (p_inst, visual_emb, p_logits, p_deltas, targets) in enumerate(zip(
                proposals, visual_prompts_split, class_logits_split, deltas_split, gt_instances
            )):
                if len(targets) == 0: continue

                # --- 1. Matcher & Force Positive (QUAN TRỌNG) ---
                with torch.no_grad():
                    match_quality_matrix = pairwise_iou(p_inst.proposal_boxes, targets.gt_boxes)
                    matched_idxs, matched_labels = self.matcher(match_quality_matrix)
                    
                    # [FORCE MATCH] Gán nhãn Positive cho các GT đã được trộn vào cuối danh sách
                    num_gt = len(targets)
                    if num_gt > 0:
                        safe_idx = min(len(matched_labels), num_gt)
                        if safe_idx > 0:
                            # Index của các GT nằm ở cuối danh sách proposals
                            matched_labels[-safe_idx:] = 1
                            # Map chúng với chính các target tương ứng (0, 1, ... num_gt-1)
                            gt_indices_forced = torch.arange(num_gt, device=self.device)[-safe_idx:]
                            matched_idxs[-safe_idx:] = gt_indices_forced

                    pos_inds = torch.nonzero(matched_labels == 1).squeeze(1)

                # --- 2. Loss CLS (Safe Version) ---
                # Fallback: Tự động thêm cột background nếu thiếu
                if p_logits.shape[1] == self.num_classes:
                    padding = torch.full((p_logits.shape[0], 1), -1e9, device=self.device)
                    p_logits = torch.cat([p_logits, padding], dim=1)
                
                bg_class_ind = p_logits.shape[1] - 1 
                gt_classes_target = torch.full((len(p_inst),), bg_class_ind, dtype=torch.long, device=self.device)
                
                if len(pos_inds) > 0:
                    gt_idx_mapped = matched_idxs[pos_inds]
                    target_labels = targets.gt_classes[gt_idx_mapped]
                    # Kẹp nhãn để không bao giờ vượt quá background
                    target_labels = torch.clamp(target_labels, min=0, max=bg_class_ind - 1)
                    gt_classes_target[pos_inds] = target_labels

                loss_cls_img = self.loss_cls(p_logits, gt_classes_target)
                loss_cls_acc += loss_cls_img
                total_samples += 1

                # --- 3. Loss Box Reg ---
                if len(pos_inds) > 0:
                    gt_inds = matched_idxs[pos_inds]
                    
                    # Lấy GT Box và Proposal Box tương ứng
                    # [DIRECT INDEXING] Detectron2 hỗ trợ indexing trực tiếp, không cần loop CPU
                    gt_boxes = targets.gt_boxes[gt_inds]
                    src_boxes = p_inst.proposal_boxes[pos_inds]
                    
                    # Tính GT Deltas
                    gt_deltas = self.box2box_transform.get_deltas(src_boxes.tensor, gt_boxes.tensor)
                    pred_deltas_pos = p_deltas[pos_inds]
                    
                    loss_box_reg_img = smooth_l1_loss(pred_deltas_pos, gt_deltas, beta=0.0, reduction="sum")
                    loss_box_reg_acc += loss_box_reg_img / max(len(pos_inds), 1.0)

                # --- 4. Loss Mask ---
                if len(pos_inds) == 0: continue
                
                # Sampling đơn giản để tránh OOM
                if len(pos_inds) > 64:
                    perm = torch.randperm(len(pos_inds))[:64]
                    pos_inds = pos_inds[perm]
                
                # Cập nhật lại gt_inds theo pos_inds đã sample
                gt_inds = matched_idxs[pos_inds]
                
                # Create Hybrid Prompt
                rpn_boxes_pos = p_inst.proposal_boxes[pos_inds].tensor
                with torch.no_grad():
                    box_embeddings = self.prompt_encoder(points=None, boxes=rpn_boxes_pos, masks=None)[0] 
                visual_embeddings_pos = visual_emb[pos_inds] 
                sparse_embeddings = torch.cat([box_embeddings, visual_embeddings_pos], dim=1)
                
                with torch.no_grad():
                    _, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None)
                dense_embeddings = dense_embeddings.repeat(len(pos_inds), 1, 1, 1)

                curr_emb = image_embeddings[i]
                while curr_emb.dim() > 3: curr_emb = curr_emb.squeeze(0)
                curr_emb = curr_emb.unsqueeze(0)
                curr_pe = self.pe_layer(curr_emb)

                # Get GT Masks [DIRECT INDEXING]
                gt_masks_orig = targets.gt_masks[gt_inds]
                if isinstance(gt_masks_orig, PolygonMasks):
                    h_img, w_img = targets.image_size
                    bitmasks = BitMasks.from_polygon_masks(gt_masks_orig.polygons, h_img, w_img)
                    gt_masks_tensor = bitmasks.tensor.to(self.device).float()
                else:
                    gt_masks_tensor = gt_masks_orig.tensor.float().to(self.device)

                low_res_masks, iou_preds = self.mask_decoder(
                    image_embeddings=curr_emb, image_pe=curr_pe,
                    sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                if low_res_masks.dim() == 5: low_res_masks = low_res_masks.squeeze(2)
                pred_masks_up = F.interpolate(
                    low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False
                ).squeeze(1)
                
                loss_dict = self.criterion(pred_masks_up, gt_masks_tensor, iou_preds.flatten())
                mask_loss_acc += loss_dict["loss_mask"] * len(pos_inds)
                total_pos += len(pos_inds)

            # Normalize losses
            if total_samples > 0:
                losses["loss_cls"] = loss_cls_acc / total_samples
                losses["loss_box_reg"] = loss_box_reg_acc / total_samples
            else:
                losses["loss_cls"] = torch.tensor(0., device=self.device, requires_grad=True)
                losses["loss_box_reg"] = torch.tensor(0., device=self.device, requires_grad=True)

            if total_pos > 0:
                losses["loss_mask"] = mask_loss_acc / total_pos
            else:
                dummy_loss = 0.0
                # Cộng các tham số của mask_decoder
                for p in self.mask_decoder.parameters():
                    dummy_loss += p.sum()
                # Cộng các tham số của prompt_encoder (nếu có train)
                for p in self.prompt_encoder.parameters():
                    dummy_loss += p.sum()
                    
                # Nhân với 0 để không ảnh hưởng kết quả, nhưng có Gradient
                losses["loss_mask"] = dummy_loss * 0.0

            return losses
            
        # ... (Phần inference giữ nguyên) ...
        else:
            final_results = []
            for i, (p_inst, p_logits, p_deltas, visual_emb) in enumerate(zip(proposals, class_logits_split, deltas_split, visual_prompts_split)):
                if len(p_inst) == 0: final_results.append(p_inst); continue
                
                scores, classes = F.softmax(p_logits, dim=-1).max(dim=-1)
                valid_mask = classes < self.num_classes
                
                pred_boxes = self.box2box_transform.apply_deltas(p_deltas, p_inst.proposal_boxes.tensor)
                refined_boxes = Boxes(pred_boxes)
                p_inst.proposal_boxes = refined_boxes
                
                # Logic Inference SAM như cũ (Hybrid Prompt)
                rpn_boxes = p_inst.proposal_boxes.tensor
                with torch.no_grad():
                    box_embeddings = self.prompt_encoder(points=None, boxes=rpn_boxes, masks=None)[0]
                
                visual_embeddings = visual_emb
                sparse_embeddings = torch.cat([box_embeddings, visual_embeddings], dim=1)
                
                with torch.no_grad():
                    _, dense = self.prompt_encoder(points=None, boxes=None, masks=None)
                dense = dense.repeat(sparse_embeddings.size(0), 1, 1, 1)

                curr_emb = image_embeddings[i]
                while curr_emb.dim() > 3: curr_emb = curr_emb.squeeze(0)
                curr_emb = curr_emb.unsqueeze(0)
                curr_pe = self.pe_layer(curr_emb)

                # Batch Inference
                chunk_size = 16
                preds_parts = []
                num_props = sparse_embeddings.size(0)
                for st in range(0, num_props, chunk_size):
                    en = min(st + chunk_size, num_props)
                    s_chunk = sparse_embeddings[st:en]
                    d_chunk = dense[st:en]
                    masks_chunk, _ = self.mask_decoder(curr_emb, curr_pe, s_chunk, d_chunk, multimask_output=False)
                    if masks_chunk.dim() == 5: masks_chunk = masks_chunk.squeeze(2)
                    mask_pred_chunk = F.interpolate(masks_chunk, (1024, 1024), mode="bilinear", align_corners=False).squeeze(1)
                    mask_pred_chunk = (mask_pred_chunk > 0.0) 
                    preds_parts.append(mask_pred_chunk)

                if len(preds_parts) > 0:
                    mask_pred = torch.cat(preds_parts, dim=0)
                else:
                    mask_pred = torch.zeros((0, 1024, 1024), dtype=torch.bool, device=self.device)

                # Filtering
                final_instances = p_inst[valid_mask]
                if len(final_instances) > 0:
                    final_instances.pred_masks = mask_pred[valid_mask].unsqueeze(1)
                    final_instances.pred_classes = classes[valid_mask]
                    final_instances.scores = scores[valid_mask]
                    final_instances.pred_boxes = final_instances.proposal_boxes
                    if final_instances.has("proposal_boxes"): final_instances.remove("proposal_boxes")
                else:
                    final_instances.pred_masks = torch.zeros((0, 1, 1024, 1024), dtype=torch.bool, device=self.device)
                    final_instances.pred_classes = torch.tensor([], dtype=torch.long, device=self.device)
                    final_instances.scores = torch.tensor([], device=self.device)
                    final_instances.pred_boxes = Boxes(torch.tensor([], device=self.device))

                final_results.append(final_instances)

            return self._postprocess(final_results, batched_inputs, images.image_sizes)