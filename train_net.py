import torch
import os
import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

from detectron2.engine import DefaultTrainer, default_argument_parser, launch, default_setup
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances

# Import model của cậu
from core import backbone, uwsam 

def add_uwsam_config(cfg):
    cfg.MODEL.SAM = CN()
    # --- [Config ViT-B như yêu cầu] ---
    cfg.MODEL.SAM.TYPE = "vit_h"
    # Đường dẫn weight SAM gốc (cần đảm bảo file này tồn tại)
    cfg.MODEL.SAM.CHECKPOINT = "/home/hoangnv/MaskRCNN-LoRA-/weights/sam_vit_h_4b8939.pth" 
    cfg.MODEL.SAM.IMAGE_SIZE = 1024
    cfg.MODEL.SAM.FREEZE = True
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000 
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.] 
    cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
    cfg.MODEL.SAM.LORA = CN()
    cfg.MODEL.SAM.LORA.ENABLED = True
    cfg.MODEL.SAM.LORA.RANK = 8
    cfg.MODEL.SAM.LORA.ALPHA = 8
    cfg.MODEL.SAM.LORA.DROPOUT = 0.05

    cfg.MODEL.EUPG = CN()
    cfg.MODEL.EUPG.NUM_PROPOSALS = 64
    cfg.MODEL.EUPG.SCORE_THRESH = 0.5
    cfg.MODEL.EUPG.NMS_THRESH = 0.5
    return cfg

def setup(args):
    cfg = get_cfg()
    add_uwsam_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.META_ARCHITECTURE = "UWSAM"
    cfg.MODEL.BACKBONE.NAME = "SAMBackbone"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    # Input size cố định cho SAM
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.CROP.ENABLED = False 
    
    cfg.MODEL.DEVICE = "cuda"
    DATA_ROOT = "/data/bailab_data/hoangnv/UIIS10K/"
    register_coco_instances("uiis10k_train", {}, os.path.join(DATA_ROOT, "annotations/multiclass_train.json"), os.path.join(DATA_ROOT, "img"))
    register_coco_instances("uiis10k_test", {}, os.path.join(DATA_ROOT, "annotations/multiclass_test.json"), os.path.join(DATA_ROOT, "img"))
    
    cfg.DATASETS.TRAIN = ("uiis10k_train",)
    cfg.DATASETS.TEST = ("uiis10k_test",)
    cfg.DATALOADER.NUM_WORKERS = 4 # Tăng lên 4 nếu CPU khỏe để load ảnh nhanh hơn
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.ACCUMULATE_STEPS = 4  # Giả lập batch size lớn hơn bằng cách tích lũy Gradients
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WARMUP_ITERS = 1000 # Giảm warmup xuống chút vì dataset cũng ko quá lớn
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.BIAS_LR_FACTOR = 1.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    
    # Tính toán số iter dựa trên batch size
    dicts = DatasetCatalog.get("uiis10k_train")
    num_images = len(dicts)
    batch_size = cfg.SOLVER.IMS_PER_BATCH  
    one_epoch_iters = int(num_images / batch_size)
    
    cfg.SOLVER.MAX_ITER = one_epoch_iters * 24
    cfg.SOLVER.STEPS = (one_epoch_iters * 15, one_epoch_iters * 20)
    cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters 
    cfg.TEST.EVAL_PERIOD = one_epoch_iters *24
    cfg.SOLVER.AMP.ENABLED = True
    
    # Gradient Clipping là cần thiết cho Transformer
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    cfg.OUTPUT_DIR = "./output/uwsam_vit_h_lora_standard_bs"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # Data augmentation
        augs = [
            T.RandomFlip(),
            T.ResizeScale(
                min_scale=0.1, max_scale=2.0, target_height=1024, target_width=1024
            ), # Sửa min/max scale hợp lý hơn chút cho training ổn định
            T.FixedSizeCrop(crop_size=(1024, 1024), pad=True, pad_value=128.0), # Pad value 0 hoặc mean pixel tùy ý
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs, use_instance_mask=True, recompute_boxes=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Resize về 1024x1024 cứng cho test để khớp input SAM
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[
            T.Resize((1024, 1024)) 
        ])
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # Logic lọc tham số requires_grad (quan trọng cho LoRA/Freeze backbone)
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if weight_decay is None:
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
        # AdamW thường tốt hơn SGD cho Transformer
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return optimizer

    # [XÓA] Đã xóa hàm run_step() để dùng Default Loop của Detectron2

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    trainer = Trainer(cfg)
    model = trainer.model
    
    # In thông tin model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*40}")
    print(f"[MODEL INFO] Architecture: UWSAM ({cfg.MODEL.SAM.TYPE.upper()} + LoRA)")
    print(f"[MODEL INFO] Total Params:      {total_params / 1e6:.2f} M")
    print(f"[MODEL INFO] Trainable Params:  {trainable_params / 1e6:.2f} M")
    print(f"[MODEL INFO] Trainable Ratio:   {(trainable_params/total_params)*100:.2f} %")
    print(f"{'='*40}\n")
    
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        dist_url=args.dist_url,
        args=(args,),
    )