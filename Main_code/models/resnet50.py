"""
=========================================================================================
MODULE: Adapted ResNet-50 Architecture Definition (models/resnet50.py)
=========================================================================================
DESCRIPTION:
    Implements the customized ResNet-50 architecture equipped with residual bottleneck 
    blocks and a multi-layer deep fully connected classification head.
    
    Methodological Specifications (see Manuscript Section 'Selection of Neural 
    Architectures' & Table 3):
    1. Deep Residual Backbone: Leverages standard ResNet-50 pre-trained on ImageNet-1K 
       to capture deep hierarchical spatial representations of dermoscopic lesions.
    2. Deep Classification Head: Replaces the final linear layer with a multi-layer MLP 
       comprising Linear layers (2048 → 2048 → 512 → 128 → 7 classes), interleaved with 
       Batch Normalization, ReLU activations, and Dropout (p=0.5).
    3. Three-Phase Progressive Unfreezing:
       - Phase 1 (Warmup Head): Freezes all backbone parameters, training only the classification head.
       - Phase 2 (Layer 4 Unfreezing): Selectively unfreezes residual Layer 4 and the classification head.
       - Phase 3 (Full Model): Unfreezes the entire ResNet-50 architecture.
=========================================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Adapted(nn.Module):
    """
    Customized ResNet-50 topology with a deep multi-layer classification head and 
    granular stage-level unfreezing controls.
    """
    def __init__(self, dropout_rate=0.5, num_classes=7):
        super().__init__()
        
        # Load ResNet-50 backbone with ImageNet pre-trained weights
        try:
            self.backbone = models.resnet50(weights='DEFAULT')
        except TypeError:
            self.backbone = models.resnet50(pretrained=True)
        
        # Replace default fully connected layer with a deep multi-layer MLP head
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 2048),  
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.8),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    # ==========================================
    # PROGRESSIVE UNFREEZING CONTROL METHODS
    # ==========================================
    def freeze_backbone(self):
        """Freezes the entire ResNet-50 backbone, leaving only the classification head trainable (Phase 1)."""
        print("❄️ Freezing ResNet-50 backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Ensure custom classification head remains trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self, unfreeze_layer4=False, last_blocks=None):
        """
        Unfreezes the backbone.
        If last_blocks=True (Phase 2), selectively unfreezes residual Layer 4 and the head.
        If last_blocks=False (Phase 3), unfreezes the entire model.
        """
        if last_blocks:
            print("🔓 Partially unfreezing ResNet-50: Layer 4 and Classification Head released...")
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        else:
            print("🔥 Unfreezing entire ResNet-50 model (Full Fine-Tuning)...")
            for param in self.parameters():
                param.requires_grad = True


# Alias for backward compatibility
ResNet50Adaptado = ResNet50Adapted


def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def create_model(dropout_rate=0.5, init_weights='imagenet', **kwargs):
    """
    Instantiates the ResNet-50 adapted model with ImageNet pre-trained weights or 
    Kaiming initialization.
    """
    if init_weights == 'imagenet':
        model = ResNet50Adapted(dropout_rate=dropout_rate)
    elif init_weights in ['kaiming', 'XavierU', 'XavierN']:
        model = ResNet50Adapted(dropout_rate=dropout_rate)
        print(f"⚠️ Warning: Initialization '{init_weights}' overrides pre-trained ImageNet weights.")
        if init_weights == 'kaiming':
            model.apply(init_weights_kaiming)
    else:
        raise ValueError(f"Unrecognized init_weights option: '{init_weights}'.")
    
    return model


def count_parameters(model):
    """Returns the total number of currently trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)