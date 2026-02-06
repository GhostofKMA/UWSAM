import torch
import torch.nn as nn
import math

class LoRA_SAM_QKV(nn.Module):
    def __init__(self, original_qkv_module, r=8, alpha=8, dropout=0.05):
        super().__init__()
        self.qkv = original_qkv_module
        for param in self.qkv.parameters():
            param.requires_grad = False
        self.dim = original_qkv_module.in_features 
        self.lora_A_q = nn.Linear(self.dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, self.dim, bias=False) 
        self.lora_A_v = nn.Linear(self.dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, self.dim, bias=False) 
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        qkv_out = self.qkv(x)
        x_drop = self.dropout(x)
        delta_q = self.lora_B_q(self.lora_A_q(x_drop)) * self.scaling
        delta_v = self.lora_B_v(self.lora_A_v(x_drop)) * self.scaling
        qkv_out[..., :self.dim] += delta_q
        qkv_out[..., -self.dim:] += delta_v
        return qkv_out

def inject_lora_sam(model, r=8, alpha=8, dropout=0.05):
    for name, module in model.named_children():
        if name == "qkv" and isinstance(module, nn.Linear):
            lora_layer = LoRA_SAM_QKV(module, r=r, alpha=alpha, dropout=dropout)
            setattr(model, name, lora_layer) 
        else:
            inject_lora_sam(module, r=r, alpha=alpha, dropout=dropout)