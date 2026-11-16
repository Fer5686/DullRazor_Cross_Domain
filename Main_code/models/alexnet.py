"""
=========================================================================================
MODULE: Adapted AlexNet Architecture Definition (models/alexnet.py)
=========================================================================================
DESCRIPTION:
    Implements the customized AlexNet architecture (~61.1M parameters) adapted for 
    7-class dermoscopic skin lesion classification.
    
    Methodological Specifications (see Manuscript Section 'Additional Topological 
    Specifications' & Table 3):
    1. Dynamic GPU Spatial Padding: Integrates an on-the-fly asymmetric zero-padding 
       layer (nn.ZeroPad2d((1, 2, 1, 2))) at the network input to dynamically expand 
       224x224 tensors to the 227x227 spatial resolution required by the first 11x11 
       convolutional filter, avoiding interpolation artifacts from forced rescaling.
    2. Pre-trained Initialization: Injects official ImageNet-1K pre-trained weights 
       while replacing the final 1000-class linear layer with a 7-class diagnostic head 
       regularized via Dropout (p=0.5).
    3. Phase-Based Gradient Control: Provides explicit methods to freeze the convolutional 
       extractor (Phase 1), selectively unfreeze convolutional Blocks 3, 4, and 5 
       (Phase 2), or unfreeze the entire topology (Phase 3).
=========================================================================================
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class AlexNetManual(nn.Module):
    """
    Customized AlexNet topology with on-the-fly 224->227 spatial padding and 
    granular stage-level unfreezing controls.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # ==========================================
        # 0. ON-THE-FLY GPU SPATIAL PADDING
        # ==========================================
        # Expands 224x224 input tensors to 227x227 by adding zero-padded borders.
        # Padding tuple: (left=1, right=2, top=1, bottom=2) -> 224 + 3 = 227.
        self.gpu_padding = nn.ZeroPad2d((1, 2, 1, 2))

        # ==========================================
        # 1. FEATURE EXTRACTOR (Convolutional Backbone)
        # ==========================================
        self.features = nn.Sequential(
            # Block 1 (11x11 large spatial filter)
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # index 0
            nn.ReLU(inplace=True),                                  # index 1
            nn.MaxPool2d(kernel_size=3, stride=2),                  # index 2
            
            # Block 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # index 3
            nn.ReLU(inplace=True),                                  # index 4
            nn.MaxPool2d(kernel_size=3, stride=2),                  # index 5
            
            # Block 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # index 6
            nn.ReLU(inplace=True),                                  # index 7
            
            # Block 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # index 8
            nn.ReLU(inplace=True),                                  # index 9
            
            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # index 10
            nn.ReLU(inplace=True),                                  # index 11
            nn.MaxPool2d(kernel_size=3, stride=2),                  # index 12
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # ==========================================
        # 2. CLASSIFICATION HEAD (Classifier)
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if pretrained:
            print("📥 Downloading and injecting ImageNet-1K pre-trained weights for AlexNet...")
            state_dict = load_state_dict_from_url(
                'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', 
                progress=True
            )
            
            # Remove the original 1000-class output layer weights
            state_dict.pop('classifier.6.weight', None)
            state_dict.pop('classifier.6.bias', None)
            
            self.load_state_dict(state_dict, strict=False)
            print("✅ Pre-trained weights injected successfully.")

    def forward(self, x):
        x = self.gpu_padding(x)  # Dynamically adjusts 224x224 -> 227x227 on GPU
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
        print("❄️ Backbone frozen (Only CLASSIFIER head is trainable).")

    def unfreeze_backbone(self, last_blocks=True):
        """
        Unfreezes the convolutional backbone.
        If last_blocks=True (Phase 2), selectively unfreezes from Block 3 onwards (indices 6 to 12).
        If last_blocks=False (Phase 3), unfreezes the entire network.
        """
        if last_blocks:
            for i, layer in enumerate(self.features):
                if i >= 6:
                    for param in layer.parameters():
                        param.requires_grad = True
            print("🔥 Convolutional Blocks 3, 4, 5 and Classifier head unfrozen.")
        else:
            for param in self.parameters():
                param.requires_grad = True
            print("🔥 Full AlexNet architecture unfrozen.")


# Alias for backward compatibility
AlexNet_Manual = AlexNetManual


def create_model(num_classes=7, device='cuda'):
    """Instantiates and allocates the pre-trained AlexNet model on the target device."""
    model = AlexNetManual(num_classes=num_classes, pretrained=True)
    return model.to(device)


def count_parameters(model):
    """Returns the total number of currently trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)