"""
=========================================================================================
MODULE: Dataset Instantiation and Routing (instantiate_datasets.py)
=========================================================================================
DESCRIPTION:
    Configures population normalization constants, diagnostic class mappings, and 
    dataset routing logic for the serialized binary tensor pipelines (.pt).
    
    Methodological Specifications (see Manuscript Section 'Tensor Preprocessing and 
    Bottleneck Mitigation', Figure 7, & Table 2):
    1. Population Normalization Constants: Derived strictly from the training partition 
       distribution (MEAN_TRAIN = [0.7635, 0.5427, 0.5665], STD_TRAIN = [0.1373, 0.1536, 0.1712]) 
       to standardize chromaticity across all partitions without data leakage.
    2. Hybrid Training Pipeline (uint8 -> DynamicTransformations): Routes training 
       requests to pre-augmented (x10 Random Resized Crop) 8-bit unsigned integer (.pt) 
       tensors, delegating in-place RandAugment and float conversion to runtime.
    3. Static Validation & Test Pipelines (float32 -> ImagesDatasetPT): Routes validation 
       (N=1,497) and official ISIC 2018 test (N=1,511) requests to deterministic 32-bit 
       floating-point tensors resized to architecture-specific resolutions (224x224 or 256x256).
    4. Dual Domain Support: Dynamically switches between raw occluded datasets and 
       DullRazor-inpainted datasets based on the 'hair_removed' configuration flag.
=========================================================================================
"""

import os
from static_dataset import ImagesDatasetPT
from dynamic_dataset import DynamicTransformations
from config_paths import DOCS_DIR

# =========================================================
# 1. POPULATION NORMALIZATION CONSTANTS & LABEL MAPPING
# =========================================================
# Population statistics computed across the training partition
MEAN_TRAIN = [0.7635, 0.5427, 0.5665]
STD_TRAIN  = [0.1373, 0.1536, 0.1712]

# Diagnostic category mapping across the 7 classes of HAM10000 / ISIC 2018
label_mapping = {
    'nv': 0,     # Melanocytic nevi
    'mel': 1,    # Melanoma
    'bkl': 2,    # Benign keratosis-like lesions
    'bcc': 3,    # Basal cell carcinoma
    'akiec': 4,  # Actinic keratoses / intraepidermal carcinoma
    'vasc': 5,   # Vascular lesions
    'df': 6      # Dermatofibroma
}

# =========================================================
# 2. MASTER DATASET CONFIGURATION DICTIONARY
# =========================================================
# Configured for hybrid dynamic data augmentation on unbalanced uint8 training tensors
data_param = {
    "ROI": False,
    "flipHV_rot": False,
    "affine": False,
    "jitter": False,
    "normalize": True,
    "masked": False,
    "masked_value": -1,
    "shift": 0,
    "max_cropped": False,
    "gaussian": False,
    "crop_type": 'random',    # 'resize', 'center', 'random'
    "channels": 3,
    "size": 224,               # Dynamically set to 224 or 256 by the orchestrator
    "balanced": False,
    "randaug": True,           # Enables on-the-fly RandAugment via DynamicTransformations
    "multiplier": 1,           # Version multiplier index for serialized .pt files
    "hair_removed": True       # True = DullRazor Clean Domain | False = Raw Occluded Domain
}


# =========================================================
# 3. DATASET ROUTING AND LOADING FUNCTION
# =========================================================
def load_dataset(data_param, dataset_type='train'):
    """
    Instantiates and returns the appropriate PyTorch Dataset object according to the 
    requested partition ('train', 'val', 'test'), spatial resolution (224 vs. 256), 
    and preprocessing domain (With Hair vs. Without Hair).
    """
    norm_params = {
        "mean": MEAN_TRAIN,
        "std": STD_TRAIN,
        "apply_norm": data_param.get("normalize", False)
    }

    if dataset_type not in ['train', 'val', 'test']:
        raise ValueError("Invalid dataset_type. Expected 'train', 'val', or 'test'.")

    # Support both English ('hair_removed') and legacy Spanish ('afeitado') keys
    hair_removed = data_param.get("hair_removed", data_param.get("afeitado", False))
    resolution = data_param["size"]

    if resolution not in [224, 256]:
        raise ValueError(f"Unsupported spatial resolution: {resolution}. Expected 224 or 256.")

    # ==========================
    # TRAINING PARTITION (uint8 x10 Static Crop + On-the-fly Dynamic Augmentation)
    # ==========================
    if dataset_type == 'train':
        if resolution == 256:
            folder_name = (
                "256x256_Rcrop_unbalanced_hair_removed_x10_uint8" 
                if hair_removed else 
                "256x256_Rcrop_unbalanced_x10_uint8"
            )
        else:  # resolution == 224
            folder_name = (
                "224x224_Rcrop_unbalanced_hair_removed_x10_uint8" 
                if hair_removed else 
                "224x224_Rcrop_unbalanced_x10_uint8"
            )

        pt_file_train = os.path.join(DOCS_DIR, "HAM10000", "preprocessed_tensors", folder_name, "version_0.pt")
        print(f"📦 Loading Training Partition: {pt_file_train} (Norm: {norm_params['apply_norm']})")
        
        if data_param.get('randaug', False):
            return DynamicTransformations(pt_file_train, data_param, norm_params)
        else:
            return ImagesDatasetPT(pt_file_train, data_param, norm_params)

    # ==========================
    # VALIDATION PARTITION (float32 Static)
    # ==========================
    elif dataset_type == 'val':
        if resolution == 256:
            folder_name = "256x256_val_hair_removed" if hair_removed else "256x256_val"
        else:  # resolution == 224
            folder_name = "224x224_val_hair_removed" if hair_removed else "224x224_val"

        pt_file_val = os.path.join(DOCS_DIR, "HAM10000", "preprocessed_tensors", folder_name, "version_0.pt")
        return ImagesDatasetPT(pt_file_val, data_param, norm_params)

    # ==========================
    # TEST PARTITION (ISIC 2018 Ground Truth, float32 Static)
    # ==========================
    elif dataset_type == 'test':
        if resolution == 256:
            folder_name = "groundtruth_256x256_hair_removed" if hair_removed else "groundtruth_256x256"
        else:  # resolution == 224
            folder_name = "groundtruth_224x224_hair_removed" if hair_removed else "groundtruth_224x224"

        pt_file_test = os.path.join(DOCS_DIR, "HAM10000", "preprocessed_tensors", folder_name, "version_0.pt")
        return ImagesDatasetPT(pt_file_test, data_param, norm_params)