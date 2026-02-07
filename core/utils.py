# File: core/utils.py
import torch
import torch.nn as nn
import numpy as np

class ColorAttentionAdapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, change=False):
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(embedding_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, embedding_dim, 1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.change_channel = change

    def forward(self, x):
        # Logic từ common.py
        if self.change_channel:
            x_perm = x.permute(0, 3, 1, 2).contiguous()
            avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x_perm))))
            max_out = self.fc2(self.act(self.fc1(self.max_pool(x_perm))))
            out = self.Sigmoid(avg_out + max_out).view(x.shape[0], 1, 1, -1)
            return out
        else:
            avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
            return self.Sigmoid(avg_out + max_out) * x

class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.GELU):
        super().__init__()
        self.act = act_layer()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        # Multi-scale convolutions
        self.conv3 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(output_dim, output_dim, 5, padding=2)
        self.conv7 = nn.Conv2d(output_dim, output_dim, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        # Cộng gộp các scale lại
        x = self.conv3(x) + self.conv5(x) + self.conv7(x)
        return self.act(self.bn2(x))

class PositionEmbeddingRandom(nn.Module):
    """
    Gaussian Positional Encoding như mô tả trong paper/code anchor.py
    """
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        # coords range [-1, 1]
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size, device):
        h, w = size
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        # [H, W, 2]
        coords = torch.stack([x_embed, y_embed], dim=-1)
        # [H, W, C] -> [C, H, W]
        pe = self._pe_encoding(coords).permute(2, 0, 1) 
        return pe.unsqueeze(0) # [1, C, H, W]