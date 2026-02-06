# File: core/backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from segment_anything.modeling import ImageEncoderViT
from .lora import inject_lora_sam 

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        def run_forward(input_tensor): return self.block(input_tensor)
        if self.training and x.requires_grad:
            return checkpoint(run_forward, x, use_reentrant=False)
        return self.block(x)

@BACKBONE_REGISTRY.register()
class SAMBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        
        model_type = cfg.MODEL.SAM.TYPE
        
        # 1. Cấu hình động dựa trên model_type (vit_b, vit_l, vit_h)
        if model_type == "vit_b":
            embed_dim = 768
            depth = 12
            num_heads = 12
            global_attn_indexes = [2, 5, 8, 11]
            self.distillation_indices = [2, 5, 8, 11] # Lấy feature tại các layer Global Attention
            print("[SAMBackbone] Initializing ViT-Base...")
            
        elif model_type == "vit_l":
            embed_dim = 1024
            depth = 24
            num_heads = 16
            global_attn_indexes = [5, 11, 17, 23]
            self.distillation_indices = [5, 11, 17, 23]
            print("[SAMBackbone] Initializing ViT-Large...")
            
        elif model_type == "vit_h":
            embed_dim = 1280
            depth = 32
            num_heads = 16
            global_attn_indexes = [7, 15, 23, 31]
            self.distillation_indices = [7, 15, 23, 31]
            print("[SAMBackbone] Initializing ViT-Huge...")
            
        else:
            raise ValueError(f"Unsupported SAM model type: {model_type}")

        # 2. Khởi tạo ImageEncoderViT
        self.vit = ImageEncoderViT(
            depth=depth, 
            embed_dim=embed_dim, 
            img_size=1024, 
            mlp_ratio=4,
            norm_layer=torch.nn.LayerNorm, 
            num_heads=num_heads, 
            patch_size=16,
            qkv_bias=True, 
            use_rel_pos=True, 
            global_attn_indexes=global_attn_indexes,
            window_size=14, 
            out_chans=256, 
        )

        # 3. Load Pretrained Weights
        if cfg.MODEL.SAM.CHECKPOINT:
            print(f"[SAMBackbone] Loading weights from {cfg.MODEL.SAM.CHECKPOINT}")
            state_dict = torch.load(cfg.MODEL.SAM.CHECKPOINT, map_location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("image_encoder."):
                    new_state_dict[k[len("image_encoder."):]] = v
            
            # Load weight (strict=True để đảm bảo khớp hoàn toàn)
            msg = self.vit.load_state_dict(new_state_dict, strict=False)
            print(f"[SAMBackbone] Weights loaded. Missing keys (expect prompt/mask encoder keys): {len(msg.missing_keys)}")

        # 4. Freeze logic
        if cfg.MODEL.SAM.FREEZE:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("[SAMBackbone] Backbone frozen.")

        # 5. Inject LoRA
        if cfg.MODEL.SAM.LORA.ENABLED:
            print(f"[SAMBackbone] Injecting LoRA (Rank={cfg.MODEL.SAM.LORA.RANK})...")
            inject_lora_sam(
                self.vit, 
                r=cfg.MODEL.SAM.LORA.RANK, 
                alpha=cfg.MODEL.SAM.LORA.ALPHA,
                dropout=cfg.MODEL.SAM.LORA.DROPOUT
            )
            
            # Gradient Checkpointing (Giúp tiết kiệm VRAM)
            for i in range(len(self.vit.blocks)):
                self.vit.blocks[i] = CheckpointedBlock(self.vit.blocks[i])

        self._out_feature_channels = {"feature_map": 256}
        self._out_feature_strides = {"feature_map": 16} 

    def forward(self, x):
        x = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed

        distillation_features = []

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.distillation_indices:
                distillation_features.append(x)

        x = x.permute(0, 3, 1, 2) 
        final_feature = self.vit.neck(x)

        return {
            "feature_map": final_feature,
            "distill_feats": distillation_features
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_feature_strides
        }