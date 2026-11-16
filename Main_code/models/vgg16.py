"""
=========================================================================================
MODULE: Adapted VGG-16 Architecture Definition (models/vgg16.py)
=========================================================================================
DESCRIPTION:
    Implements the customized VGG-16 architecture adapted for 7-class dermoscopic 
    skin lesion classification.
    
    Methodological Specifications (see Manuscript Section 'Selection of Neural 
    Architectures' & Table 3):
    1. Sequential Feature Extractor: Structured into five convolutional blocks paired 
       with max-pooling operations, strictly matching official torchvision state dictionaries.
    2. Pre-trained Initialization: Injects official ImageNet-1K pre-trained weights 
       while replacing the final 1000-class linear layer with a 7-class diagnostic head 
       regularized via Dropout (p=0.5).
    3. Phase-Based Gradient Control: Provides explicit methods to freeze the convolutional 
       extractor (Phase 1), selectively unfreeze Block 5 (Phase 2), or unfreeze the entire 
       network (Phase 3).
=========================================================================================
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG16Manual(nn.Module):
    """
    Customized VGG-16 topology partitioned into 5 sequential convolutional blocks 
    with granular stage-level unfreezing controls.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # ==========================================
        # 1. FEATURE EXTRACTOR (Convolutional Backbone)
        # ==========================================
        # Structured into 5 convolutional blocks matching official torchvision state dictionaries.
        self.features = nn.Sequential(
            # --- Block 1 ---
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # index 0
            nn.ReLU(inplace=True),                       # index 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # index 2
            nn.ReLU(inplace=True),                       # index 3
            nn.MaxPool2d(kernel_size=2, stride=2),       # index 4
            
            # --- Block 2 ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# index 5
            nn.ReLU(inplace=True),                       # index 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# index 7
            nn.ReLU(inplace=True),                       # index 8
            nn.MaxPool2d(kernel_size=2, stride=2),       # index 9
            
            # --- Block 3 ---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# index 10
            nn.ReLU(inplace=True),                       # index 11
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# index 12
            nn.ReLU(inplace=True),                       # index 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# index 14
            nn.ReLU(inplace=True),                       # index 15
            nn.MaxPool2d(kernel_size=2, stride=2),       # index 16
            
            # --- Block 4 ---
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# index 17
            nn.ReLU(inplace=True),                       # index 18
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# index 19
            nn.ReLU(inplace=True),                       # index 20
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# index 21
            nn.ReLU(inplace=True),                       # index 22
            nn.MaxPool2d(kernel_size=2, stride=2),       # index 23
            
            # --- Block 5 ---
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# index 24
            nn.ReLU(inplace=True),                       # index 25
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# index 26
            nn.ReLU(inplace=True),                       # index 27
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# index 28
            nn.ReLU(inplace=True),                       # index 29
            nn.MaxPool2d(kernel_size=2, stride=2)        # index 30
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # ==========================================
        # 2. CLASSIFICATION HEAD (Classifier)
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        # Load official ImageNet pre-trained weights
        if pretrained:
            print("📥 Downloading and injecting ImageNet-1K pre-trained weights for VGG-16...")
            state_dict = load_state_dict_from_url(
                'https://download.pytorch.org/models/vgg16-397923af.pth', 
                progress=True
            )
            
            # Remove original 1000-class output layer weights to fit 7-class lesion categories
            state_dict.pop('classifier.6.weight', None)
            state_dict.pop('classifier.6.bias', None)
            
            self.load_state_dict(state_dict, strict=False)
            print("✅ Pre-trained weights injected successfully.")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # ==========================================
    # 3. PROGRESSIVE UNFREEZING CONTROL METHODS
    # ==========================================
    def freeze_backbone(self):
        """Freezes the entire convolutional feature extractor (Phase 1: Warmup Head)."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("❄️ VGG-16 Backbone frozen (Only CLASSIFIER head is trainable).")

    def unfreeze_backbone(self, last_blocks=True):
        """
        Unfreezes the convolutional backbone.
        If last_blocks=True (Phase 2), selectively unfreezes Block 5 (indices 24 to 30).
        If last_blocks=False (Phase 3), unfreezes the entire network.
        """
        if last_blocks:
            for i, layer in enumerate(self.features):
                if i >= 24:
                    for param in layer.parameters():
                        param.requires_grad = True
            print("🔥 VGG-16 Block 5 and Classifier head unfrozen.")
        else:
            for param in self.parameters():
                param.requires_grad = True
            print("🔥 Full VGG-16 architecture unfrozen.")


# Alias for backward compatibility
VGG16_Manual = VGG16Manual


def create_model(num_classes=7, device='cuda'):
    """Instantiates and allocates the pre-trained VGG-16 model on the target device."""
    model = VGG16Manual(num_classes=num_classes, pretrained=True)
    return model.to(device)


def count_parameters(model):
    """Returns the total number of currently trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)