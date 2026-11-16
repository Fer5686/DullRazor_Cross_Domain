"""
=========================================================================================
MODULE: Dynamic Dataset Pipeline with Curriculum Augmentation (dynamic_dataset.py)
=========================================================================================
DESCRIPTION:
    Implements a PyTorch Dataset class (`DynamicTransformations`) that loads pre-processed 
    lossless tensor partitions (uint8 [0-255]) and applies on-the-fly curriculum-driven 
    stochastic data augmentations via torchvision.transforms.v2.
    
    Methodological Specifications (see Manuscript Section 'Dynamic Augmentation and 
    Optimized Data Loading'):
    1. Zero-Copy Tensor Pipeline: Operates natively on uint8 tensor representations using 
       torchvision v2, avoiding the severe performance overhead of converting tensors to 
       PIL images and back.
    2. Curriculum Augmentation Control: Dynamically updates RandAugment severity and 
       probabilities per epoch via `set_intensities()`.
    3. Channel Reconstitution & Normalization: Automatically aligns color channels 
       (e.g., handling 3-channel RGB vs. 4-channel alpha configurations) before 
       converting to float32 [0, 1] scaling and applying dataset-level population normalization 
       (mean/std).
=========================================================================================
"""

import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Dict, Optional


class DynamicTransformations(Dataset):
    """
    Dataset loader supporting multi-version tensor sampling and curriculum-based 
    on-the-fly data augmentation for dermoscopic images.
    """
    def __init__(self, pt_file: str, data_params: Dict, norm_params: Optional[Dict] = None):
        self.data_params = data_params
        
        # Load pre-processed partition tensors (stored as uint8 [0-255])
        data = torch.load(pt_file, weights_only=True)
        self.images = data['images'] 
        self.labels = data['labels']
        
        # Detect multi-version tensor shape (5D: [N, Versions, C, H, W] vs 4D: [N, C, H, W])
        self.has_versions = len(self.images.shape) == 5
        self.num_versions = self.images.shape[1] if self.has_versions else 1
        
        # Channel adjustment configuration
        self.channels_needed = data_params.get("channels", 3)
        self.channels_actual = self.images.shape[2 if self.has_versions else 1]
        
        self.adjust_channels = self._no_adjust
        if self.channels_needed == 4 and self.channels_actual == 3:
            self.adjust_channels = self._add_alpha
            self.dummy_alpha = torch.ones(1, *self.images.shape[-2:], dtype=torch.uint8)
        elif self.channels_needed == 3 and self.channels_actual == 4:
            self.adjust_channels = self._remove_alpha
        
        # Data augmentation state variables
        self.randaug_prob = 0.0
        self.augmentation = v2.Identity()
        
        self.scale_factor = 1.0 / 255.0

        # Population Normalization Configuration
        self.apply_norm = False
        if norm_params and norm_params.get("apply_norm"):
            self.apply_norm = True
            # Reshape mean and std to [3, 1, 1] for simultaneous broadcasted channel-wise operations
            self.mean = torch.tensor(norm_params["mean"]).view(3, 1, 1)
            self.std = torch.tensor(norm_params["std"]).view(3, 1, 1)

    def set_intensities(self, prob: float, magnitude: int, num_ops: int = 1):
        """Updates the stochastic augmentation pipeline intensity for the current epoch."""
        self.randaug_prob = prob
        
        if prob > 0.01:
            # v2.RandAugment operates natively on uint8 tensors
            self.augmentation = v2.RandAugment(num_ops=num_ops, magnitude=magnitude)
        else:
            self.augmentation = v2.Identity()
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 1. Stochastic version selection (if multiple pre-rendered augmentations exist)
        if self.has_versions:
            ver_idx = random.randint(0, self.num_versions - 1)
            image = self.images[idx, ver_idx] 
        else:
            image = self.images[idx]
        
        # 2. Channel adjustment (remains in uint8 format)
        image = self.adjust_channels(image)
        
        # 3. Apply RandAugment (v2)
        if self.randaug_prob > 0.01: 
             image = self.augmentation(image)

        # 4. Type conversion, float scaling [0, 1], and population normalization
        image = image.float().div(255.0)

        if self.apply_norm:
            image = (image - self.mean) / self.std

        return {
            'image': image, 
            'label': self.labels[idx]
        }
    
    # --- Internal Channel Helpers ---
    def _no_adjust(self, image):
        return image
    
    def _add_alpha(self, image):
        return torch.cat([image, self.dummy_alpha], dim=0)
    
    def _remove_alpha(self, image):
        return image[:3]