"""
=========================================================================================
MODULE: Static Dataset Pipeline for Pre-loaded Tensors (static_dataset.py)
=========================================================================================
DESCRIPTION:
    Implements a PyTorch Dataset class (`ImagesDatasetPT`) that loads pre-processed 
    tensor partitions from disk and applies channel adjustments and population normalization 
    efficiently during data loading.
    
    Methodological Specifications (see Manuscript Section 'Dynamic Augmentation and 
    Optimized Data Loading'):
    1. Pre-computed Dispatch Functions: Binds tensor version selection and channel 
       adjustment methods dynamically during initialization to eliminate conditional branching 
       overhead within the inner `__getitem__` loop.
    2. Population Normalization: Applies dataset-level channel-wise mean subtraction and 
       standard deviation scaling using pre-allocated tensor buffers.
=========================================================================================
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


class ImagesDatasetPT(Dataset):
    """
    Optimized PyTorch Dataset for loading pre-rendered tensor partitions (.pt files) 
    with on-the-fly channel alignment and population normalization.
    """
    def __init__(self, pt_file: str, data_params: Dict, norm_params: Optional[Dict] = None):
        data = torch.load(pt_file, weights_only=True)
        self.images = data['images']  
        self.labels = data['labels']
        
        # 1. Precompute dimension structure
        self._has_versions = len(self.images.shape) == 5
        self._num_versions = self.images.shape[1] if self._has_versions else 1
        
        # 2. Configure Population Normalization
        self.apply_norm = False
        if norm_params and norm_params.get("apply_norm"):
            self.apply_norm = True
            # Store mean and std as class attributes reshaped to [3, 1, 1] for broadcasted operations
            self.mean = torch.tensor(norm_params["mean"]).view(3, 1, 1)
            self.std = torch.tensor(norm_params["std"]).view(3, 1, 1)

        # 3. Precompute channel adjustments
        channels_needed = data_params.get("channels", 3)
        channels_actual = self.images.shape[2 if self._has_versions else 1]
        
        if channels_needed == 4 and channels_actual == 3:
            self._channel_mode = 'add_alpha'
            self._dummy_alpha = torch.ones(1, *self.images.shape[-2:])
        elif channels_needed == 3 and channels_actual == 4:
            self._channel_mode = 'remove_alpha'
        else:
            self._channel_mode = 'none'
        
        # Bind helper methods dynamically to avoid conditional overhead in __getitem__
        self._get_image = self._get_image_with_versions if self._has_versions else self._get_image_single
        
        if self._channel_mode == 'add_alpha':
            self._adjust_channels = self._add_alpha_channel
        elif self._channel_mode == 'remove_alpha':
            self._adjust_channels = self._remove_alpha_channel
        else:
            self._adjust_channels = self._no_channel_adjust
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self._get_image(idx)
        image = self._adjust_channels(image)
        
        if self.apply_norm:
            image = (image - self.mean) / self.std
            
        return {
            'image': image, 
            'label': self.labels[idx]
        }
    
    # --- Internal Helper Methods ---
    def _get_image_with_versions(self, idx):
        version_idx = random.randint(0, self._num_versions - 1)
        return self.images[idx, version_idx]
    
    def _get_image_single(self, idx):
        return self.images[idx]
    
    def _add_alpha_channel(self, image):
        return torch.cat([image, self._dummy_alpha], dim=0)
    
    def _remove_alpha_channel(self, image):
        return image[:3]
    
    def _no_channel_adjust(self, image):
        return image