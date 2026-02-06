import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import CfgNode
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import ImageList, Instances, BitMasks, PolygonMasks, pairwise_iou
from detectron2.modeling.matcher import Matcher 
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import paste_masks_in_image
# Import components
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from .eupg import EUPG, PositionEmbeddingRandom

# ---------------------------------------------------------------------------- #
# 1. LOSS FUNCTIONS (Giữ nguyên vì đã tốt)
# ---------------------------------------------------------------------------- #
# Thay thế class UWSAMCriterion trong core/uwsam.py

class UWSAMCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.dice_weight = 1.0
        self.focal_weight = 20.0 
        self.iou_weight = 1.0
        self.smooth = 1e-6 # Epsilon nhỏ để tránh chia cho 0

    def compute_dice_loss(self, inputs, targets):
        # inputs đang là logits -> sigmoid
        inputs = inputs.sigmoid()
        
        # Flatten
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(1) + targets.sum(1)
        
        # Thêm self.smooth vào cả tử và mẫu để tránh NaN khi cả 2 đều bằng 0
        loss = 1 - (numerator + self.smooth) / (denominator + self.smooth)
        return loss.mean()

    def compute_iou_loss(self, pred_ious, true_ious):
        return F.mse_loss(pred_ious, true_ious)

    def forward(self, pred_masks, gt_masks, pred_iou_scores):
        if pred_masks is None or pred_masks.numel() == 0:
            device = pred_masks.device if pred_masks is not None else torch.device("cpu")
            zero = torch.tensor(0., device=device, requires_grad=True)
            return {
                "loss_mask_focal": zero,
                "loss_mask_dice": zero,
                "loss_mask_iou": zero
            }

        # [FIX NAN] Clamp logits vào khoảng an toàn (-10, 10) trước khi tính toán
        # Để tránh sigmoid bị bão hòa hoặc exp bị nổ
        pred_masks = torch.clamp(pred_masks, min=-10.0, max=10.0)

        loss_focal = sigmoid_focal_loss_jit(
            pred_masks, gt_masks, 
            alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean"
        )
        
        loss_dice = self.compute_dice_loss(pred_masks, gt_masks)

        with torch.no_grad():
            pred_masks_binary = (pred_masks.sigmoid() > 0.5).float()
            intersection = (pred_masks_binary * gt_masks).sum(dim=(1, 2))
            union = pred_masks_binary.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2)) - intersection
            # Thêm epsilon vào mẫu số
            true_ious = (intersection + 1e-6) / (union + 1e-6)
        
        loss_iou = self.compute_iou_loss(pred_iou_scores, true_ious)

        return {
            "loss_mask_focal": loss_focal * self.focal_weight,
            "loss_mask_dice": loss_dice * self.dice_weight,
            "loss_mask_iou": loss_iou * self.iou_weight
        }

# ---------------------------------------------------------------------------- #
# 2. MAIN MODEL
# ---------------------------------------------------------------------------- #
@META_ARCH_REGISTRY.register()
class UWSAM(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = EUPG(cfg, self.backbone.output_shape())
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        embed_dim = 256
        self.transformer = TwoWayTransformer(
            depth=2, embedding_dim=embed_dim, mlp_dim=2048, num_heads=8,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=self.transformer,
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )
        
        if cfg.MODEL.SAM.CHECKPOINT:
            self._load_sam_decoder_weights(cfg.MODEL.SAM.CHECKPOINT)
            
        # Freeze SAM components logic
        for param in self.mask_decoder.parameters(): param.requires_grad = True # Nên để True nếu muốn fine-tune decoder
        for param in self.prompt_encoder.parameters(): param.requires_grad = False
        for param in self.backbone.parameters(): param.requires_grad = False
            
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        self.matcher = Matcher(
            thresholds=[0.3], 
            labels=[0, 1],        
            allow_low_quality_matches=True                
        )
        
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
        # 1. Preprocess
        images = self.preprocess_image(batched_inputs)
        
        with torch.no_grad():
            features = self.backbone(images.tensor)
            
        image_embeddings = features["feature_map"]
        if isinstance(image_embeddings, list): image_embeddings = image_embeddings[0]
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.detach()
        else:
            image_embeddings = [x.detach() for x in image_embeddings]

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            
        # 2. RPN / EUPG Forward
        proposals, proposal_losses, all_prompt_embeddings, pred_class_logits = self.proposal_generator(images, features, gt_instances)
        
        num_proposals_per_img = [len(p) for p in proposals]
        prompt_embeddings_list = all_prompt_embeddings.split(num_proposals_per_img)

        # ---------------- TRAINING PHASE ----------------
        if self.training:
            losses = {}
            losses.update(proposal_losses) # Luôn có Loss RPN (để RPN học)
            
            mask_count = 0
            mask_loss_acc = {k: torch.tensor(0., device=self.device) for k in ["loss_mask_focal", "loss_mask_dice", "loss_mask_iou"]}
            # Debug accumulators to inspect why mask losses may be zero
            pred_mean_accum = 0.0
            gt_mean_accum = 0.0
            processed_images_with_masks = 0

            for i, (p_inst, p_emb, targets) in enumerate(zip(proposals, prompt_embeddings_list, gt_instances)):
                # Per-image debug counters
                try:
                    get_event_storage().put_scalar("debug/num_proposals_per_image", float(len(p_inst)))
                except Exception:
                    pass

                if len(targets) == 0 or len(p_inst) == 0:
                    try:
                        get_event_storage().put_scalar("debug/skip_no_targets_or_proposals", 1.0)
                    except Exception:
                        pass
                    continue

                with torch.no_grad():
                    # Logic Matcher chuẩn của Detectron2
                    match_quality_matrix = pairwise_iou(p_inst.proposal_boxes, targets.gt_boxes)
                    matched_idxs, matched_labels = self.matcher(match_quality_matrix)
                    
                    # Lấy các Positive samples (Label = 1)
                    pos_inds = torch.nonzero(matched_labels == 1).squeeze(1)

                    if len(pos_inds) == 0:
                        # Fallback: nếu không có mẫu dương từ matcher, dùng trực tiếp GT boxes
                        # như prompts để đảm bảo Mask Decoder luôn được training (theo chiến lược in-paper).
                        gt_box_t = targets.gt_boxes.tensor.to(self.device)
                        if gt_box_t.numel() == 0:
                            try:
                                get_event_storage().put_scalar("debug/fallback_no_gt_boxes", 1.0)
                            except Exception:
                                pass
                            continue
                        # Lấy embeddings từ prompt_encoder cho GT boxes (mỗi box 1 batch item)
                        sparse_gt, dense_gt = self.prompt_encoder(points=None, boxes=gt_box_t, masks=None)

                        # Giới hạn số GT nếu quá nhiều để tránh OOM
                        max_gt_for_training = 64
                        if sparse_gt.size(0) > max_gt_for_training:
                            keep = torch.randperm(sparse_gt.size(0), device=self.device)[:max_gt_for_training]
                            sparse_gt = sparse_gt[keep]
                            dense_gt = dense_gt[keep]
                            gt_inds = keep
                        else:
                            gt_inds = torch.arange(sparse_gt.size(0), device=self.device)

                        # Mark that we'll use GT prompts downstream
                        use_gt_prompts = True

                        try:
                            get_event_storage().put_scalar("debug/use_gt_prompts", float(sparse_gt.size(0)))
                        except Exception:
                            pass
                    else:
                        use_gt_prompts = False

                    try:
                        get_event_storage().put_scalar("debug/pos_inds_count", float(len(pos_inds)))
                    except Exception:
                        pass

                    # Sample bớt nếu quá nhiều (tiết kiệm VRAM)
                    if len(pos_inds) > 64:
                        perm = torch.randperm(len(pos_inds))[:64]
                        pos_inds = pos_inds[perm]

                    gt_inds = matched_idxs[pos_inds]

                    # --- [SAFETY CHECK - BẮT BUỘC ĐỂ KHÔNG CRASH] ---
                    # Đảm bảo index nằm trong khoảng [0, len(targets)-1]
                    valid_mask = (gt_inds >= 0) & (gt_inds < len(targets))
                    if not valid_mask.all():
                        pos_inds = pos_inds[valid_mask]
                        gt_inds = gt_inds[valid_mask]
                    
                    if len(pos_inds) == 0: continue
                    # -----------------------------------------------

                # --- CHUẨN BỊ INPUT ---
                # Nếu fallback dùng GT prompts => sparse_gt/dense_gt đã được tạo
                if 'use_gt_prompts' in locals() and use_gt_prompts:
                    sparse_embeddings = sparse_gt
                    dense_embeddings = dense_gt
                else:
                    # Lấy đúng Prompt Embedding tương ứng với box được chọn
                    sparse_embeddings = p_emb[pos_inds]
                    with torch.no_grad():
                        _, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None)
                    dense_embeddings = dense_embeddings.repeat(len(pos_inds), 1, 1, 1)

                curr_emb = image_embeddings[i]
                while curr_emb.dim() > 3: curr_emb = curr_emb.squeeze(0)
                curr_emb = curr_emb.unsqueeze(0)
                curr_pe = self.pe_layer(curr_emb)

                # Lấy GT Mask
                gt_masks_orig = targets.gt_masks[gt_inds] 
                
                if isinstance(gt_masks_orig, PolygonMasks):
                    h_img, w_img = targets.image_size
                    bitmasks = BitMasks.from_polygon_masks(gt_masks_orig.polygons, h_img, w_img)
                    gt_masks_tensor = bitmasks.tensor.to(self.device).float()
                else:
                    gt_masks_tensor = gt_masks_orig.tensor.float().to(self.device)

                # --- MASK DECODER FORWARD ---
                num_prompts = sparse_embeddings.size(0)
                chunk_size = 4 # Giảm chunk nếu OOM
                
                for st in range(0, num_prompts, chunk_size):
                    en = min(st + chunk_size, num_prompts)
                    
                    low_res_masks, iou_preds = self.mask_decoder(
                        image_embeddings=curr_emb,
                        image_pe=curr_pe,
                        sparse_prompt_embeddings=sparse_embeddings[st:en],
                        dense_prompt_embeddings=dense_embeddings[st:en],
                        multimask_output=False,
                    )

                    if low_res_masks.dim() == 5: low_res_masks = low_res_masks.squeeze(2)
                    ph, pw = low_res_masks.size(-2), low_res_masks.size(-1)
                    
                    pred_masks_chunk = low_res_masks.reshape(-1, ph, pw)
                    pred_ious_chunk = iou_preds.flatten()

                    gt_raw_slice = gt_masks_tensor[st:en]
                    gt_down_chunk = F.interpolate(
                        gt_raw_slice.unsqueeze(1), size=(ph, pw), mode="nearest"
                    ).squeeze(1)

                    # Compute areas at both downsampled and original resolutions
                    gt_down_area = gt_down_chunk.flatten(1).sum(1)
                    gt_orig_area = gt_raw_slice.flatten(1).sum(1)

                    # Large masks available at downsampled resolution
                    valid_large = gt_down_area > 0
                    # Small masks that vanished after downsampling but exist in original
                    valid_small = (gt_down_area == 0) & (gt_orig_area > 0)
                    skipped = (~valid_large & ~valid_small).sum().item()
                    if skipped > 0:
                        try:
                            get_event_storage().put_scalar("debug/skipped_small_masks", skipped)
                        except Exception:
                            pass

                    try:
                        get_event_storage().put_scalar("debug/valid_large_count", float(valid_large.sum().item()))
                        get_event_storage().put_scalar("debug/valid_small_count", float(valid_small.sum().item()))
                        get_event_storage().put_scalar("debug/skipped_count", float(skipped))
                    except Exception:
                        pass

                    # If nothing valid at all, skip
                    if (valid_large.sum() + valid_small.sum()).item() == 0:
                        continue

                    # Prepare tensors for loss: handle large (use low-res) and small (upsample preds)
                    preds_for_loss = []
                    gts_for_loss = []
                    ious_for_loss = []

                    if valid_large.sum().item() > 0:
                        plarge = pred_masks_chunk[valid_large]
                        glarge = gt_down_chunk[valid_large]
                        ilarge = pred_ious_chunk[valid_large]
                        preds_for_loss.append(plarge)
                        gts_for_loss.append(glarge)
                        ious_for_loss.append(ilarge)

                    if valid_small.sum().item() > 0:
                        # Upsample small preds to original resolution to compute loss
                        psmall = pred_masks_chunk[valid_small]
                        # psmall: (K, h, w) -> upsample to (K, 1024, 1024)
                        psmall_up = F.interpolate(
                            psmall.unsqueeze(1), size=(1024, 1024), mode="bilinear", align_corners=False
                        ).squeeze(1)
                        gsmall = gt_raw_slice[valid_small]
                        ism = pred_ious_chunk[valid_small]
                        preds_for_loss.append(psmall_up)
                        gts_for_loss.append(gsmall)
                        ious_for_loss.append(ism)

                    # Compute loss separately for large and small masks to avoid mixing different
                    # spatial sizes (low-res vs full-res). This avoids cat() errors when shapes differ.
                    loss_n = 0

                    # Large masks: use low-res preds and downsampled GT
                    if valid_large.sum().item() > 0:
                        plarge = pred_masks_chunk[valid_large]
                        glarge = gt_down_chunk[valid_large]
                        ilarge = pred_ious_chunk[valid_large]

                        loss_dict_large = self.criterion(plarge, glarge, ilarge)
                        n_large = plarge.shape[0]
                        loss_n += n_large

                        # Debug stats for large
                        try:
                            get_event_storage().put_scalar("debug/pred_masks_mean_large", plarge.sigmoid().mean().item())
                            get_event_storage().put_scalar("debug/gt_masks_mean_large", glarge.mean().item())
                        except Exception:
                            pass

                        for k, v in loss_dict_large.items():
                            mask_loss_acc[k] += v * n_large

                    # Small masks: upsample preds and use original GT resolution
                    if valid_small.sum().item() > 0:
                        psmall = pred_masks_chunk[valid_small]
                        psmall_up = F.interpolate(
                            psmall.unsqueeze(1), size=(1024, 1024), mode="bilinear", align_corners=False
                        ).squeeze(1)
                        gsmall = gt_raw_slice[valid_small]
                        ism = pred_ious_chunk[valid_small]

                        loss_dict_small = self.criterion(psmall_up, gsmall, ism)
                        n_small = psmall_up.shape[0]
                        loss_n += n_small

                        # Debug stats for small
                        try:
                            get_event_storage().put_scalar("debug/pred_masks_mean_small", psmall_up.sigmoid().mean().item())
                            get_event_storage().put_scalar("debug/gt_masks_mean_small", gsmall.mean().item())
                        except Exception:
                            pass

                        for k, v in loss_dict_small.items():
                            mask_loss_acc[k] += v * n_small

                    if loss_n == 0:
                        # nothing valid, skip
                        continue

                    # Update counts and means
                    n_valid = loss_n
                    mask_count += n_valid

                    try:
                        # Weighted accumulations
                        if valid_large.sum().item() > 0:
                            pred_mean_accum += plarge.sigmoid().mean().item() * plarge.shape[0]
                            gt_mean_accum += glarge.mean().item() * glarge.shape[0]
                        if valid_small.sum().item() > 0:
                            pred_mean_accum += psmall_up.sigmoid().mean().item() * psmall_up.shape[0]
                            gt_mean_accum += gsmall.mean().item() * psmall_up.shape[0]
                    except Exception:
                        pass

                    processed_images_with_masks += 1


            # Normalize Loss
            if mask_count > 0:
                averaged = {k: (mask_loss_acc[k] / mask_count) for k in mask_loss_acc}

                # Add debug scalars
                try:
                    avg_pred = pred_mean_accum / mask_count
                    avg_gt = gt_mean_accum / mask_count
                except Exception:
                    avg_pred = 0.0
                    avg_gt = 0.0

                averaged["debug/mask_count"] = torch.tensor(float(mask_count), device=self.device)
                averaged["debug/processed_images_with_masks"] = torch.tensor(float(processed_images_with_masks), device=self.device)
                averaged["debug/pred_mean"] = torch.tensor(float(avg_pred), device=self.device)
                averaged["debug/gt_mean"] = torch.tensor(float(avg_gt), device=self.device)

                # Warn if all mask losses are still zero
                try:
                    if all((averaged[k].item() == 0.0) for k in ["loss_mask_focal", "loss_mask_dice", "loss_mask_iou"]):
                        averaged["debug/warning_all_zero"] = torch.tensor(1.0, device=self.device)
                    else:
                        averaged["debug/warning_all_zero"] = torch.tensor(0.0, device=self.device)
                except Exception:
                    averaged["debug/warning_all_zero"] = torch.tensor(0.0, device=self.device)

                losses.update(averaged)
            else:
                # Nếu không có mask nào (do RPN chưa tốt), trả về 0 để không lỗi backward
                zero = torch.tensor(0., device=self.device, requires_grad=True)
                losses.update({"loss_mask_focal": zero, "loss_mask_dice": zero, "loss_mask_iou": zero,
                               "debug/mask_count": torch.tensor(0., device=self.device),
                               "debug/processed_images_with_masks": torch.tensor(0., device=self.device),
                               "debug/pred_mean": torch.tensor(0., device=self.device),
                               "debug/gt_mean": torch.tensor(0., device=self.device),
                               "debug/warning_all_zero": torch.tensor(1.0, device=self.device)})
            
            return losses

        # ... (Phần Inference giữ nguyên) ...

        # ---------------- INFERENCE PHASE ----------------
        else:
            final_results = []
            class_logits_split = pred_class_logits.split(num_proposals_per_img)
            
            # Zip thêm p_emb (prompt embeddings)
            for i, (p_inst, p_logits, p_emb) in enumerate(zip(proposals, class_logits_split, prompt_embeddings_list)):
                if len(p_inst) == 0:
                    final_results.append(p_inst); continue
                
                curr_emb = image_embeddings[i]
                while curr_emb.dim() > 3: curr_emb = curr_emb.squeeze(0)
                curr_emb = curr_emb.unsqueeze(0)
                curr_pe = self.pe_layer(curr_emb)
                
                # --- PREDICT MASKS ---
                sparse = p_emb 
                with torch.no_grad():
                    _, dense = self.prompt_encoder(points=None, boxes=None, masks=None)
                dense = dense.repeat(sparse.size(0), 1, 1, 1)

                num_props = sparse.size(0)
                chunk_size = 16
                preds_parts = []
                
                for st in range(0, num_props, chunk_size):
                    en = min(st + chunk_size, num_props)
                    s_chunk = sparse[st:en]
                    d_chunk = dense[st:en]
                    
                    # Forward Decoder
                    masks_chunk, _ = self.mask_decoder(curr_emb, curr_pe, s_chunk, d_chunk, multimask_output=False)
                    
                    if masks_chunk.dim() == 5: masks_chunk = masks_chunk.squeeze(2)
                    mh, mw = masks_chunk.size(-2), masks_chunk.size(-1)
                    
                    # Upsample về 1024x1024
                    mask_pred_chunk = F.interpolate(
                        masks_chunk.reshape(-1, 1, mh, mw),
                        (1024, 1024),
                        mode="bilinear", align_corners=False,
                    )
                    mask_pred_chunk = (mask_pred_chunk > 0.0) # Binary Mask
                    preds_parts.append(mask_pred_chunk)

                if len(preds_parts) > 0:
                    mask_pred = torch.cat(preds_parts, dim=0)
                else:
                    mask_pred = torch.zeros((0, 1, 1024, 1024), dtype=torch.bool, device=self.device)

                # --- FILTER & ASSIGN RESULT (FIXED ERROR) ---
                scores, classes = F.softmax(p_logits, dim=-1).max(dim=-1)
                
                # 1. Tạo mask lọc bỏ background
                valid_mask = classes < self.num_classes
                
                # 2. Cắt (Slice) Instances trước!
                # Dòng này sẽ tạo ra một Instance mới chỉ chứa các box hợp lệ (Length = 571)
                final_instances = p_inst[valid_mask]
                
                if len(final_instances) > 0:
                    # 3. Gán các thuộc tính vào Instance mới (Length khớp nhau)
                    final_instances.pred_masks = mask_pred[valid_mask]
                    final_instances.pred_classes = classes[valid_mask]
                    final_instances.scores = scores[valid_mask]
                    
                    # Detectron2 yêu cầu field 'pred_boxes' cho inference
                    # 'proposal_boxes' đã được slice tự động ở bước 2, giờ ta gán sang 'pred_boxes'
                    final_instances.pred_boxes = final_instances.proposal_boxes
                    
                    # Xóa field thừa cho nhẹ
                    if final_instances.has("proposal_boxes"):
                        final_instances.remove("proposal_boxes")
                else:
                    # Xử lý trường hợp rỗng (không có object nào)
                    final_instances.pred_masks = torch.zeros((0, 1, 1024, 1024), dtype=torch.bool, device=self.device)
                    final_instances.pred_classes = torch.tensor([], dtype=torch.long, device=self.device)
                    final_instances.scores = torch.tensor([], device=self.device)
                    final_instances.pred_boxes = Boxes(torch.tensor([], device=self.device))

                final_results.append(final_instances)

            return self._postprocess(final_results, batched_inputs, images.image_sizes)
    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale kết quả output về kích thước ảnh gốc.
        """
        processed_results = []
        for results, input_per_image, image_size in zip(instances, batched_inputs, image_sizes):
            # Lấy chiều cao/rộng gốc của ảnh
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            
            # Hàm detector_postprocess của Detectron2 sẽ tự động resize boxes/masks
            r = detector_postprocess(results, height, width)
            processed_results.append({"instances": r})
            
        return processed_results