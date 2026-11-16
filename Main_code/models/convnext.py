"""
=========================================================================================
MODULE: Native ConvNeXt-Small Architecture Definition (models/convnext.py)
=========================================================================================
DESCRIPTION:
    Implements the native ConvNeXt-Small architecture (~50M parameters) adapted for 
    7-class dermoscopic skin lesion classification, loading official Facebook pre-trained 
    weights via timm.
    
    Methodological Specifications (see Manuscript Section 'Selection of Neural 
    Architectures' & Table 3):
    1. Modernized Convolutional Design: Combines depthwise separable convolutions (7x7 
       kernels), LayerNorm2d, GELU activations, and inverted bottlenecks, bridging the 
       architectural gap between traditional CNNs and Vision Transformers.
    2. Stochastic Depth (DropPath): Integrates path dropout across sequential stages 
       to regularize training and prevent overfitting.
    3. Three-Phase Progressive Unfreezing:
       - Phase 1 (Warmup Head): Freezes Stem and all Stages, training only the classification head.
       - Phase 2 (Stage 3 Unfreezing): Selectively unfreezes the final blocks of Stage 3 and the head.
       - Phase 3 (Full Model): Unfreezes the entire ConvNeXt hierarchy.
=========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DropPath(nn.Module):
    """
    Stochastic Depth / Drop Path.
    Randomly zeroes entire residual paths during training to improve generalization.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm layer operating on 2D NCHW image tensors."""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    """Utility tensor dimension permutation module."""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(self.dims)


class Mlp(nn.Module):
    """ConvNeXt Inverted Bottleneck MLP block."""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """ConvNeXt Residual Block with 7x7 depthwise convolution and LayerNorm."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.permute_nchw_to_nhwc = Permute([0, 2, 3, 1])
        self.permute_nhwc_to_nchw = Permute([0, 3, 1, 2])
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.permute_nchw_to_nhwc(x)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.permute_nhwc_to_nchw(x)
        
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)

        return shortcut + self.drop_path(x)


class ConvNeXtStage(nn.Module):
    """Sequential grouping of ConvNeXt blocks with optional spatial downsampling."""
    def __init__(self, in_dim, out_dim, depth, dp_rates, downsample=True):
        super().__init__()
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim, eps=1e-6),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
            )
            
        blocks_list = [Block(out_dim, drop_path=dp_rates[j]) for j in range(depth)]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXtHead(nn.Module):
    """Global average pooling and classification head for ConvNeXt."""
    def __init__(self, dim, num_classes, dropout_rate=0.):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = LayerNorm2d(dim, eps=1e-6) 
        self.flatten = nn.Flatten(1)
        self.drop = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class ConvNeXtManualNative(nn.Module):
    """
    Native ConvNeXt-Small architecture with 4 hierarchical stages and progressive 
    unfreezing controls.
    """
    def __init__(self, num_classes=7, drop_path_rate=0.4, head_dropout=0.3):
        super().__init__()
        
        dims = [96, 192, 384, 768]
        depths = [3, 3, 27, 3]
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        
        # Linear drop path schedule across all blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        curr = 0
        
        self.stages = nn.ModuleList()
        # Stage 0
        self.stages.append(ConvNeXtStage(dims[0], dims[0], depths[0], dp_rates[curr:curr+depths[0]], downsample=False))
        curr += depths[0]
        
        # Stages 1, 2, 3
        for i in range(3):
            self.stages.append(ConvNeXtStage(dims[i], dims[i+1], depths[i+1], dp_rates[curr:curr+depths[i+1]], downsample=True))
            curr += depths[i+1]
            
        self.head = ConvNeXtHead(dims[-1], num_classes, head_dropout)
        
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

    def _freeze_stem(self):
        print("❄️ Freezing ConvNeXt Stem...")
        for param in self.stem.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freezes entire backbone (Stem + Stages), leaving only the head trainable (Phase 1)."""
        print("❄️ Freezing complete backbone (Stem + Stages)...")
        for param in self.stem.parameters():
            param.requires_grad = False
            
        for param in self.stages.parameters():
            param.requires_grad = False
            
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self, last_n_blocks=None):
        """
        Unfreezes the backbone.
        If last_n_blocks is specified (Phase 2), selectively unfreezes 
        the final N blocks of Stage 3 and the classification head.
        If last_n_blocks is None (Phase 3), unfreezes the entire model.
        """
        if last_n_blocks is None:
            print("🔥 Unfreezing entire ConvNeXt model (Full Fine-Tuning)...")
            for param in self.parameters():
                param.requires_grad = True
        else:
            print(f"🔓 Partially unfreezing ConvNeXt: Final {last_n_blocks} blocks of Stage 3 and Classification Head released...")
            
            # Ensure head remains trainable
            for param in self.head.parameters():
                param.requires_grad = True
            
            # Unfreeze the last N blocks of Stage 3 (index 3)
            last_stage = self.stages[3] 
            total_blocks = len(last_stage.blocks)
            
            start_idx = total_blocks - last_n_blocks
            if start_idx < 0:
                start_idx = 0
                
            print(f"    -> Enabling gradients from block {start_idx} to {total_blocks-1} in Stage 3")
            for i in range(start_idx, total_blocks):
                for param in last_stage.blocks[i].parameters():
                    param.requires_grad = True


def create_model(num_classes=7, dropout_rate=0.3, drop_path_rate=0.4, freeze_stem=False, device='cuda'):
    """
    Instantiates ConvNeXt-Small and loads official pre-trained weights from Facebook via timm.
    """
    print(f"🏗️ Instantiating Native ConvNeXt-Small (Head Drop={dropout_rate}, Path Drop={drop_path_rate})...")
    
    model = ConvNeXtManualNative(num_classes=num_classes, drop_path_rate=drop_path_rate, head_dropout=dropout_rate)
    
    print("📥 Downloading Facebook pre-trained weights...")
    timm_model = timm.create_model('convnext_small.fb_in1k', pretrained=True)
    state_dict = timm_model.state_dict()
    
    clean_dict = {}
    for k, v in state_dict.items():
        if "head.fc" in k: 
            continue
        clean_dict[k] = v
        
    model.load_state_dict(clean_dict, strict=False)
    print(f"✅ Successfully loaded {len(clean_dict)} tensors directly.")
        
    print("🧠 Initializing classification head...")
    nn.init.trunc_normal_(model.head.fc.weight, std=0.02)
    nn.init.constant_(model.head.fc.bias, 0)
    
    if freeze_stem:
        model._freeze_stem()
        
    return model.to(device)


def count_parameters(model):
    """Returns the total number of currently trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)