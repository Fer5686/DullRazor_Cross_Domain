"""
=========================================================================================
MODULE: Swin Transformer Architecture Wrapper (models/swin.py)
=========================================================================================
DESCRIPTION:
    Wraps the hierarchical Swin Transformer architecture ('swin_base_patch4_window7_224' 
    via timm, ~88M parameters) for 7-class dermoscopic skin lesion classification.
    
    Methodological Specifications (see Manuscript Section 'Selection of Neural 
    Architectures' & Table 3):
    1. Hierarchical Shifted Windows: Utilizes shifted window self-attention across four 
       sequential resolution stages, capturing multi-scale contextual dependencies across 
       dermoscopic patterns.
    2. Dynamic Method Injection (Monkey Patching): Binds custom phase-control methods 
       to the pre-trained timm instance dynamically at instantiation.
    3. Three-Phase Progressive Unfreezing:
       - Phase 1 (Warmup Head): Freezes all transformer blocks except the linear head.
       - Phase 2 (Stage 4 Unfreezing): Unfreezes Stage 4 ('layers.3') and the head.
       - Phase 3 (Full Model): Unfreezes the entire Swin Transformer hierarchy.
=========================================================================================
"""

import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PatchMerging(nn.Module):
    """
    Swin Transformer Patch Merging layer: reduces spatial resolution by half 
    while quadrupling channel dimensions.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)


class WindowAttention(nn.Module):
    """
    Swin Transformer Window-based Multi-head Self-Attention (W-MSA) with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """Basic Swin Transformer Block combining Shifted Window Attention and MLP."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.shift_size = shift_size

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x) 
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


def freeze_backbone_fn(self):
    """Freezes the entire Swin backbone, leaving only the classification head trainable."""
    for name, param in self.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    print("❄️ Swin Backbone frozen (Only HEAD is trainable).")


def unfreeze_backbone_fn(self, last_blocks=True):
    """
    Unfreezes the Swin Transformer.
    If last_blocks=True (Phase 2), selectively unfreezes Stage 4 ('layers.3') and the head.
    If last_blocks=False (Phase 3), unfreezes the entire model.
    """
    if last_blocks:
        for name, param in self.named_parameters():
            if "layers.3" in name or "head" in name:
                param.requires_grad = True
        print("🔥 Swin Stage 4 and Head unfrozen.")
    else:
        for param in self.parameters():
            param.requires_grad = True
        print("🔥 Full Swin Transformer architecture unfrozen.")


def create_model(num_classes=7, device='cuda'):
    """
    Instantiates the Swin-Base pre-trained model via timm and injects phase control 
    methods dynamically via monkey patching.
    """
    print("📥 Downloading and instantiating Swin-Base pre-trained model ('swin_base_patch4_window7_224')...")
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    
    # Inject phase control methods dynamically
    model.freeze_backbone = types.MethodType(freeze_backbone_fn, model)
    model.unfreeze_backbone = types.MethodType(unfreeze_backbone_fn, model)
    
    print("✅ Swin model instantiated and methods bound successfully.")
    return model.to(device)


def count_parameters(model):
    """Returns the total number of currently trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)