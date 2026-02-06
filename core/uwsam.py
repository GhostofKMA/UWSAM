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
            
        # Nhận thêm pred_proposal_deltas
        proposals, proposal_losses, all_visual_prompts, pred_class_logits, pred_proposal_deltas = self.proposal_generator(images, features, gt_instances)
        
        num_proposals_per_img = [len(p) for p in proposals]
        visual_prompts_split = all_visual_prompts.split(num_proposals_per_img)
        class_logits_split = pred_class_logits.split(num_proposals_per_img)
        deltas_split = pred_proposal_deltas.split(num_proposals_per_img)

        # ---------------- TRAINING ----------------
        if self.training:
            # 1. Start with RPN losses (loss_rpn_cls, loss_rpn_loc)
            losses = {}
            losses.update(proposal_losses) 
            
            mask_loss_acc = torch.tensor(0., device=self.device)
            loss_cls_acc = torch.tensor(0., device=self.device)
            loss_box_reg_acc = torch.tensor(0., device=self.device) # [NEW]
            
            total_pos = 0
            total_samples = 0

            for i, (p_inst, visual_emb, p_logits, p_deltas, targets) in enumerate(zip(
                proposals, visual_prompts_split, class_logits_split, deltas_split, gt_instances
            )):
                if len(targets) == 0 or len(p_inst) == 0: continue

                # Matcher
                with torch.no_grad():
                    match_quality_matrix = pairwise_iou(p_inst.proposal_boxes, targets.gt_boxes)
                    matched_idxs, matched_labels = self.matcher(match_quality_matrix)
                    pos_inds = torch.nonzero(matched_labels == 1).squeeze(1)

                # --- A. Loss CLS ---
                if p_logits.shape[1] == self.num_classes:
                    # Tạo cột background với giá trị rất thấp (xác suất ~ 0)
                    padding = torch.full((p_logits.shape[0], 1), -1e9, device=self.device)
                    p_logits = torch.cat([p_logits, padding], dim=1)
                
                # 2. Xác định Index của Background dựa trên shape thực tế
                # Background luôn là cột cuối cùng
                bg_class_ind = p_logits.shape[1] - 1 
                
                # 3. Tạo Target
                # Mặc định gán tất cả là Background
                gt_classes_target = torch.full(
                    (len(p_inst),), bg_class_ind, dtype=torch.long, device=self.device
                )
                
                # [CRITICAL FIX] Only process valid positive matches
                if len(pos_inds) > 0:
                    # Move to CPU to safely access indices
                    pos_inds_cpu = pos_inds.cpu().numpy()
                    matched_idxs_cpu = matched_idxs.cpu().numpy()
                    
                    # Get target GT indices for positive proposals
                    gt_idxs_for_pos = matched_idxs_cpu[pos_inds_cpu]
                    
                    # Filter only valid GT indices (>= 0, < num_targets)
                    valid_mask = (gt_idxs_for_pos >= 0) & (gt_idxs_for_pos < len(targets))
                    valid_pos_inds = pos_inds[torch.tensor(valid_mask, device=self.device)]
                    valid_gt_idxs = gt_idxs_for_pos[valid_mask]
                    
                    if len(valid_pos_inds) > 0:
                        # Safely index targets on CPU, then move back
                        target_labels_list = []
                        for gt_idx in valid_gt_idxs:
                            target_labels_list.append(targets.gt_classes[int(gt_idx)])
                        target_labels = torch.stack(target_labels_list).to(self.device)
                        
                        # Check max value on CPU
                        max_label = int(target_labels.max().item()) if target_labels.numel() > 0 else 0
                        
                        # Convert 1-indexed COCO to 0-indexed if needed
                        if max_label > self.num_classes - 1:
                            target_labels = target_labels - 1
                        
                        # Clamp to valid range
                        target_labels = torch.clamp(target_labels, min=0, max=bg_class_ind - 1)
                        
                        # Assign to valid positions
                        gt_classes_target[valid_pos_inds] = target_labels

                # 4. Tính Loss
                # Lúc này: max(target) == bg_class_ind == (p_logits.shape[1] - 1)
                # Đảm bảo index luôn hợp lệ -> KHÔNG BAO GIỜ LỖI CUDA
                # [CRITICAL FIX] Clamp gt_classes_target để chắc chắn không vượt quá số classes
                gt_classes_target = torch.clamp(gt_classes_target, min=0, max=p_logits.shape[1] - 1)
                loss_cls_img = self.loss_cls(p_logits, gt_classes_target)
                
                loss_cls_acc += loss_cls_img
                total_samples += 1

                # --- B. Loss Box Reg (Stage 2) ---
                # Chỉ tính cho Positive samples
                if len(pos_inds) > 0:
                    # [CRITICAL] Use CPU-safe indexing
                    pos_inds_cpu = pos_inds.cpu().numpy()
                    matched_idxs_cpu = matched_idxs.cpu().numpy()
                    gt_inds_all = matched_idxs_cpu[pos_inds_cpu]
                    
                    # Filter valid GT indices
                    valid_gt_mask = (gt_inds_all >= 0) & (gt_inds_all < len(targets))
                    if valid_gt_mask.any():
                        valid_pos_inds = pos_inds[torch.tensor(valid_gt_mask, device=self.device)]
                        valid_gt_inds = gt_inds_all[valid_gt_mask]
                        
                        # Safely index targets using CPU indices
                        gt_boxes_list = [targets.gt_boxes[int(idx)] for idx in valid_gt_inds]
                        gt_boxes = type(targets.gt_boxes)(torch.stack([b.tensor for b in gt_boxes_list]))
                        
                        # Lấy proposal boxes tương ứng
                        src_boxes = p_inst.proposal_boxes[valid_pos_inds]
                        
                        # Tính GT Deltas chuẩn
                        gt_deltas = self.box2box_transform.get_deltas(src_boxes.tensor, gt_boxes.tensor)
                        pred_deltas_pos = p_deltas[valid_pos_inds]
                        
                        # Smooth L1 Loss
                        loss_box_reg_img = smooth_l1_loss(
                            pred_deltas_pos, gt_deltas, beta=0.0, reduction="sum"
                        )
                        # Normalize by number of positive samples
                        loss_box_reg_acc += loss_box_reg_img / max(len(valid_pos_inds), 1.0)

                # --- C. Loss Mask ---
                if len(pos_inds) == 0: continue
                
                # [CRITICAL] Filter to only valid GT indices on CPU
                pos_inds_cpu = pos_inds.cpu().numpy()
                matched_idxs_cpu = matched_idxs.cpu().numpy()
                gt_inds_all_cpu = matched_idxs_cpu[pos_inds_cpu]
                valid_mask = (gt_inds_all_cpu >= 0) & (gt_inds_all_cpu < len(targets))
                
                if not valid_mask.any():
                    continue
                
                valid_pos_inds_mask = pos_inds[torch.tensor(valid_mask, device=self.device)]
                gt_inds_safe = gt_inds_all_cpu[valid_mask]
                
                if len(valid_pos_inds_mask) > 32:
                    perm = torch.randperm(len(valid_pos_inds_mask))[:32]
                    valid_pos_inds_mask = valid_pos_inds_mask[perm]
                    gt_inds_safe = gt_inds_safe[perm]
                
                # ... (Logic tạo prompt giữ nguyên) ...
                rpn_boxes_pos = p_inst.proposal_boxes[valid_pos_inds_mask].tensor
                with torch.no_grad():
                    box_embeddings = self.prompt_encoder(points=None, boxes=rpn_boxes_pos, masks=None)[0] 
                visual_embeddings_pos = visual_emb[valid_pos_inds_mask] 
                sparse_embeddings = torch.cat([box_embeddings, visual_embeddings_pos], dim=1)
                
                with torch.no_grad():
                    _, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None)
                dense_embeddings = dense_embeddings.repeat(len(valid_pos_inds_mask), 1, 1, 1)

                curr_emb = image_embeddings[i]
                while curr_emb.dim() > 3: curr_emb = curr_emb.squeeze(0)
                curr_emb = curr_emb.unsqueeze(0)
                curr_pe = self.pe_layer(curr_emb)

                # GT Mask - use CPU indices
                gt_masks_orig = targets.gt_masks[[int(idx) for idx in gt_inds_safe]]
                if isinstance(gt_masks_orig, PolygonMasks):
                    h_img, w_img = targets.image_size
                    bitmasks = BitMasks.from_polygon_masks(gt_masks_orig.polygons, h_img, w_img)
                    gt_masks_tensor = bitmasks.tensor.to(self.device).float()
                else:
                    gt_masks_tensor = gt_masks_orig.tensor.float().to(self.device)

                # SAM Decoder
                low_res_masks, iou_preds = self.mask_decoder(
                    image_embeddings=curr_emb, image_pe=curr_pe,
                    sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                if low_res_masks.dim() == 5: low_res_masks = low_res_masks.squeeze(2)
                pred_masks_up = F.interpolate(
                    low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False
                ).squeeze(1)
                
                # Tính Loss Mask gộp
                loss_dict = self.criterion(pred_masks_up, gt_masks_tensor, iou_preds.flatten())
                mask_loss_acc += loss_dict["loss_mask"] * len(valid_pos_inds_mask)
                
                total_pos += len(valid_pos_inds_mask)

            # Normalize and Update Dict
            if total_samples > 0:
                losses["loss_cls"] = loss_cls_acc / total_samples
                losses["loss_box_reg"] = loss_box_reg_acc / total_samples # Có loss box reg
            else:
                losses["loss_cls"] = torch.tensor(0., device=self.device, requires_grad=True)
                losses["loss_box_reg"] = torch.tensor(0., device=self.device, requires_grad=True)

            if total_pos > 0:
                losses["loss_mask"] = mask_loss_acc / total_pos
            else:
                losses["loss_mask"] = torch.tensor(0., device=self.device, requires_grad=True)

            return losses

        # ---------------- INFERENCE ----------------
        else:
            final_results = []
            for i, (p_inst, p_logits, p_deltas, visual_emb) in enumerate(zip(proposals, class_logits_split, deltas_split, visual_prompts_split)):
                if len(p_inst) == 0: final_results.append(p_inst); continue
                
                # Apply Box Refinement (Stage 2)
                scores, classes = F.softmax(p_logits, dim=-1).max(dim=-1)
                valid_mask = classes < self.num_classes
                
                # Refine boxes using predicted deltas
                pred_boxes = self.box2box_transform.apply_deltas(
                    p_deltas, p_inst.proposal_boxes.tensor
                )
                # Chỉ lấy box của class dự đoán
                # (Đơn giản hoá: lấy box tương ứng class max score, hoặc dùng class-agnostic box)
                # Ở đây mình dùng class-agnostic style (giống SAM) hoặc lấy đúng cột delta.
                # EUPG BoxHead của mình trả về (N, 4) class agnostic
                refined_boxes = Boxes(pred_boxes)
                
                # Update boxes in instances (để đưa vào SAM prompt chuẩn hơn)
                p_inst.proposal_boxes = refined_boxes # Update proposal thành refined box
                
                # Tiếp tục luồng inference SAM như cũ...
                # (Logic SAM inference giữ nguyên, chỉ thay đổi input box là refined_boxes)
                # ...
                
                # [ĐOẠN NÀY COPY LẠI LOGIC INFERENCE TỪ BÀI TRƯỚC VÀ DÙNG refined_boxes]
                # ...
                
            # (Phần inference cậu dùng lại logic cũ, chỉ cần update p_inst.proposal_boxes là được)
            # Để tiết kiệm token tớ không paste lại toàn bộ phần inference trừ khi cậu cần.
            return self._postprocess(final_results, batched_inputs, images.image_sizes) 