"""
=========================================================================================
MODULE: Training Environment and Loss Function Configuration (configure_environment.py)
=========================================================================================
DESCRIPTION:
    Configures the training and evaluation environment, including Class-Balanced loss 
    functions and PyTorch DataLoaders for the TRAIN, VAL, and TEST partitions.
    
    Methodological Specifications (see Manuscript Section 'Common Training Parameters' 
    & Table 2):
    1. Class-Balanced (CB) Loss Strategy: Mitigates the severe natural class imbalance 
       of the HAM10000 dataset (where melanocytic nevi account for ~67% of samples) by 
       weighting the Cross-Entropy loss according to the effective number of samples per 
       class: w_i = (1 - beta) / (1 - beta^n_i), with beta = 0.9999. Weights are 
       normalized to a mean of 1.0 to preserve the effective learning rate scale.
    2. Additional Loss Strategies: Also supports square-root inverse frequency ('sqrt'), 
       melanoma-boosted Class-Balanced loss ('mel_boost'), soft Focal Loss ('focal_soft'), 
       and unweighted Cross-Entropy ('none').
    3. Unbiased Evaluation Criteria: While training utilizes the selected class-balancing 
       strategy, validation and test partitions are strictly evaluated using unweighted 
       (vanilla) Cross-Entropy Loss to ensure comparable, unbiased loss telemetry.
=========================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ============================================================================
# OFFICIAL HAM10000 / ISIC 2018 CLASS DISTRIBUTION COUNTS (Table 2 of Manuscript)
# ============================================================================
CLASS_COUNTS_DATASET = { # Total N = 10015, original HAM10000 partition
    'nv': 6705,
    'mel': 1113,
    'bkl': 1099,
    'bcc': 514,
    'akiec': 327,
    'vasc': 142,
    'df': 115
}

CLASS_COUNTS_TRAIN = { # Total N = 8518, train partition
    'nv': 5705,
    'mel': 945,
    'bkl': 942,
    'bcc': 437,
    'akiec': 276,
    'vasc': 120,
    'df': 93
}

CLASS_COUNTS_VAL = { # Total N = 1497, validation partition
    'nv': 1000,
    'mel': 168,
    'bkl': 157,
    'bcc': 77,
    'akiec': 51,
    'vasc': 22,
    'df': 22
}

CLASS_COUNTS_GROUNDTRUTH = {  # Total N = 1,511 (ISIC 2018 official test partition)
    'nv': 908,
    'mel': 171,
    'bkl': 217,
    'bcc': 93,
    'akiec': 43,
    'vasc': 35,
    'df': 44,
}


# ============================================================================
# 1. LOSS FUNCTION DEFINITIONS
# ============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with class weights computed from sample frequency counts.
    Supports 'cb' (Class-Balanced effective number of samples), 'sqrt' (inverse 
    square root), 'mel_boost' (CB + explicit melanoma multiplier), and 'none'.
    """
    def __init__(self, counts_dict, strategy='cb', beta=0.9999, mel_boost_factor=5.0, device='cuda'):
        super().__init__()
        
        self.strategy = strategy
        self.mel_boost_factor = mel_boost_factor
        self.counts_dict = counts_dict
        
        # Compute normalized class weights
        weights = self._calculate_weights(counts_dict, beta)
        
        # Register weights as a persistent tensor buffer on the target compute device
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32, device=device))
        
        # Console audit logging
        class_names = list(counts_dict.keys())
        print(f"   📊 WeightedCrossEntropyLoss initialized (strategy: '{strategy}')")
        if strategy == 'mel_boost':
            print(f"      -> Base: Class-Balanced (beta={beta}) + Melanoma Boost (x{mel_boost_factor})")
            
        for name, weight in zip(class_names, weights):
            print(f"      - {name}: {weight:.3f}")
        
    def _calculate_weights(self, counts_dict, beta):
        counts = list(counts_dict.values())
        class_names = list(counts_dict.keys())
        
        if self.strategy == 'sqrt':
            weights = [1.0 / np.sqrt(c) for c in counts]
            
        elif self.strategy == 'cb':
            # Class-Balanced formula based on effective number of samples: (1 - beta) / (1 - beta^n)
            weights = [(1.0 - beta) / (1.0 - (beta ** c)) for c in counts]
            
        elif self.strategy == 'mel_boost':
            # 1. Base Class-Balanced weights
            weights = [(1.0 - beta) / (1.0 - (beta ** c)) for c in counts]
            # 2. Explicit multiplier for the Melanoma ('mel') category
            if 'mel' in class_names:
                idx_mel = class_names.index('mel')
                weights[idx_mel] *= self.mel_boost_factor
            
        else:  # 'none'
            weights = [1.0] * len(counts)
            
        # Normalize weights so their mean equals 1.0 (preserving effective learning rate)
        weights = np.array(weights, dtype=np.float64)
        weights = weights / np.mean(weights)
        
        return weights.tolist()
    
    def forward(self, inputs, targets):
        return nn.functional.cross_entropy(inputs, targets, weight=self.weights)


class FocalLossSoft(nn.Module):
    """
    Soft Focal Loss formulation without static alpha weighting, ensuring full 
    compatibility with mixed/soft target distributions generated by CutMix.
    """
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        print(f"   📊 FocalLossSoft initialized (gamma={gamma})")
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


# ============================================================================
# 2. LOSS CONFIGURATION FACTORY
# ============================================================================

def setup_loss_functions(main_param, hardware_device):
    """
    Instantiates the training, validation, and test loss criteria based on main_param.
    - Training criterion: Uses the configured class-balancing strategy (default: 'cb').
    - Validation/Test criteria: Always use unweighted CrossEntropyLoss for unbiased evaluation.
    """
    print("=" * 60)
    print("⚙️ CONFIGURING LOSS FUNCTIONS")
    print("=" * 60)

    counts_dict = CLASS_COUNTS_TRAIN
    print("📊 Dataset Distribution: UNBALANCED (Natural Clinical Prevalence)")
    
    # 1. Determine active loss strategy
    loss_strategy = main_param.get("loss_strategy", "cb")
    
    if loss_strategy == "auto":
        counts = list(counts_dict.values())
        ratio = max(counts) / min(counts)
        if ratio < 2.0:
            loss_strategy = "none"
            print("⚠️ Balanced distribution detected -> using unweighted Cross-Entropy.")
        else:
            loss_strategy = "sqrt"
            print("⚠️ Unbalanced distribution detected -> using inverse square-root weights.")
    
    print(f"📊 Active Loss Strategy: {loss_strategy}")
    
    # 2. Extract optional hyperparameters (beta = 0.9999 standard for CB loss)
    loss_beta = main_param.get("loss_beta", 0.9999) 
    loss_gamma = main_param.get("loss_gamma", 1.5)
    mel_boost_factor = main_param.get("mel_boost_factor", 5.0)
    
    # 3. Instantiate Training Criterion
    if loss_strategy == "none":
        criterion_train = nn.CrossEntropyLoss()
    elif loss_strategy == "sqrt":
        criterion_train = WeightedCrossEntropyLoss(counts_dict, strategy='sqrt', device=hardware_device)
    elif loss_strategy == "cb":
        criterion_train = WeightedCrossEntropyLoss(counts_dict, strategy='cb', beta=loss_beta, device=hardware_device)
    elif loss_strategy == "focal_soft":
        criterion_train = FocalLossSoft(gamma=loss_gamma)
    elif loss_strategy == "mel_boost":
        criterion_train = WeightedCrossEntropyLoss(
            counts_dict, 
            strategy='mel_boost', 
            beta=loss_beta, 
            mel_boost_factor=mel_boost_factor, 
            device=hardware_device
        )
    else:
        raise ValueError(f"Invalid loss_strategy specified: '{loss_strategy}'.")
    
    # 4. Instantiate Validation and Test Criteria (always unweighted)
    criterion_val = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss()
    
    print("   ✅ Training Criterion: Configured with active balancing strategy.")
    print("   ✅ Validation/Test Criteria: Unweighted CrossEntropyLoss.")
    
    return criterion_train, criterion_val, criterion_test


# ============================================================================
# 3. MASTER ENVIRONMENT SETUP FUNCTION
# ============================================================================

def setup_training_environment(
    main_param, 
    data_param, 
    hardware_device, 
    dataset_train, 
    dataset_val, 
    dataset_test, 
    batch_size=64, 
    epochs=500
):
    """
    Configures and returns the complete training environment dictionary, including 
    loss criteria, PyTorch DataLoaders, epoch counts, and class distribution metadata.
    """
    print("=" * 60)
    print("⚙️ CONFIGURING TRAINING ENVIRONMENT")
    print("=" * 60)
    
    # Support both English ('debug_mode') and legacy Spanish ('prueba') flags
    debug_mode = main_param.get("debug_mode", main_param.get("prueba", False))
    num_workers = main_param.get("number_of_workers", 0)

    # 1. Configure loss functions
    criterion_train, criterion_val, criterion_test = setup_loss_functions(
        main_param, hardware_device
    )
    
    # 2. Adjust epoch limit if running in quick debug mode
    if debug_mode:
        epochs = 5
    
    # 3. Configure Training DataLoader
    is_multi_version_mode = (main_param.get("total_versions", 1) > 1) and (not debug_mode)
    
    if is_multi_version_mode:
        print("⚠️ Multi-version dynamic mode detected: fixed train_loader will not be pre-allocated.")
        train_loader = None
    else:
        if debug_mode:
            # Reduced random subset for rapid pipeline debugging
            num_samples = batch_size * 5
            subset_indices = torch.randperm(len(dataset_train))[:num_samples].tolist()
            subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_indices)
            train_loader = DataLoader(
                dataset_train,
                batch_size=batch_size,
                sampler=subset_sampler,
                num_workers=0,
                pin_memory=True
            )
        else:
            # Standard shuffled DataLoader for full training
            train_loader = DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

    # 4. Configure Validation and Test DataLoaders (deterministic ordering, shuffle=False)
    val_loader = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    ) if dataset_val is not None else None

    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    ) if dataset_test is not None else None
    
    print("✅ PyTorch DataLoaders configured successfully.")
    
    # 5. Compute per-epoch image throughput counts
    total_train_images = len(dataset_train) if dataset_train is not None else 0
    total_val_images = len(dataset_val) if (dataset_val is not None and main_param.get("validation") == 'train_val') else 0
    
    images_per_epoch = (batch_size * 5) if debug_mode else total_train_images
    if main_param.get("validation") == 'train_val':
        images_per_epoch += total_val_images
    
    return {
        'criterion_train': criterion_train,
        'criterion_val': criterion_val, 
        'criterion_test': criterion_test,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'epochs': epochs,
        'images_per_epoch': images_per_epoch,
        'class_counts': CLASS_COUNTS_TRAIN
    }