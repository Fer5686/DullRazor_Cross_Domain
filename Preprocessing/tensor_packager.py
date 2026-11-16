"""
=========================================================================================
MODULE: Multi-Version Tensor Dataset Packager (tensor_packager.py)
=========================================================================================
DESCRIPTION:
    Converts raw or hair-removed (DullRazor TIFF) dermoscopic image directories into 
    serialized PyTorch tensor files (.pt) optimized for fast I/O during training.

TENSOR OUTPUT FORMATS & MODES:
    1. BALANCED MODE (balanced=True):
        - Shape: [N, C, H, W] where N = 7 * target_per_class (e.g., 7 * 6705 = 46,935).
        - Saves `num_versions` separate files (version_0.pt, version_1.pt, ...).
        - Automatically calculates required augmentations per class to match `target_per_class`.

    2. UNBALANCED SINGLE-VERSION MODE (balanced=False, num_versions=1):
        - Shape: [N, C, H, W] where N = original_images * augs_per_image.
        - Ideal for Validation, Testing, or Native-Resolution (600x450) on-the-fly DA in RAM.

    3. UNBALANCED CONSOLIDATED MULTI-VERSION MODE (balanced=False, num_versions > 1):
        - Shape: [N, V, C, H, W] where N = original_images, V = num_versions.
        - Consolidates V stochastic random-crop views per image into a single `version_0.pt`.
        - Ideal for training with Class-Balanced Loss: maximizes spatial diversity while 
          minimizing RAM/disk overhead (especially when stored as `uint8`).
=========================================================================================
"""

import os
import sys
import gc
import math
import time
import inspect
from datetime import datetime
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

# ==========================================================
# --- PATH CONFIGURATION IMPORT ---
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_paths import HAM10000_DIR, DOCS_DIR

DATASET_ROOT = HAM10000_DIR
TENSORS_OUT_ROOT = os.path.join(DOCS_DIR, "HAM10000", "preprocessed_tensors")

LABEL_MAPPING = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
}

NORMALIZATION_MEAN = [0.7635, 0.5461, 0.5705]
NORMALIZATION_STD = [0.0876, 0.1164, 0.1306]


# ==========================================================
# --- 1. CORE PACKAGING ENGINE ---
# ==========================================================

def save_code_copy(output_dir: str):
    """Saves a snapshot of the generation script inside the output directory for reproducibility."""
    try:
        current_frame = inspect.currentframe()
        frame = current_frame.f_back.f_back
        source_code = inspect.getsource(frame.f_globals['__file__'])
        code_copy_path = os.path.join(output_dir, "generation_code.py")
        with open(code_copy_path, 'w', encoding='utf-8') as f:
            f.write(source_code)
        print(f"📄 Reproducibility script snapshot saved at: {code_copy_path}")
    except Exception:
        pass


def create_torchvision_pipeline(data_param: dict, version_idx: int) -> transforms.Compose:
    """Builds the stochastic or deterministic torchvision transformation pipeline."""
    size = data_param['size']
    transform_list = []
    torch.manual_seed(version_idx + int(time.time() * 1000) % 100000)
    
    if data_param['crop_type'] == 'random':
        transform_list.append(transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.75, 1.33)))
    elif data_param['crop_type'] == 'center':
        transform_list.append(transforms.CenterCrop(size))
    elif data_param['crop_type'] == 'resize':
        transform_list.append(transforms.Resize((size, size) if isinstance(size, int) else size))
    # Note: crop_type == 'none' preserves the native resolution (e.g., 600x450)
        
    if data_param.get('flipHV', False):
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.3)
        ])
    
    if data_param.get('rotation90', False):
        transform_list.append(transforms.RandomChoice([
            transforms.RandomRotation(degrees=[0, 0]),
            transforms.RandomRotation(degrees=[90, 90]),
            transforms.RandomRotation(degrees=[180, 180]),
            transforms.RandomRotation(degrees=[270, 270])
        ]))
    
    if data_param.get('jitter', False):
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    if data_param.get('gaussian', False):
        transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
    
    transform_list.append(transforms.ToTensor())
    
    if data_param.get('normalize', True):
        transform_list.append(transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD))
    
    return transforms.Compose(transform_list)


def detect_image_extension(image_dir: str) -> str:
    """Validates directory existence and detects image file extension (.jpg, .png, .tiff)."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"❌ CRITICAL: Image directory does not exist: {image_dir}")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
    for f in os.listdir(image_dir):
        if f.lower().endswith(valid_exts):
            return os.path.splitext(f)[1]
            
    raise FileNotFoundError(f"❌ CRITICAL: No valid images {valid_exts} found inside: {image_dir}")


def generate_dataset_version(
    version_idx: int, df: pd.DataFrame, image_dir: str, data_param: dict, 
    balanced: bool = True, target_per_class: int = 6705
) -> dict:
    """Processes all images in a partition for a single dataset version."""
    detected_ext = detect_image_extension(image_dir)
    transform = create_torchvision_pipeline(data_param, version_idx)
    images, labels, original_ids = [], [], []
    missing_count = 0
    
    grouped = df.groupby('dx') if balanced else [('all', df)]
    
    for cls, cls_df in grouped:
        augs_needed = math.ceil(target_per_class / len(cls_df)) if balanced else data_param.get('augs_per_image', 1)
        for _, row in tqdm(cls_df.iterrows(), total=len(cls_df), desc=f"Processing [{cls}]"):
            img_path = os.path.join(image_dir, str(row['image_id']) + detected_ext)
            if not os.path.exists(img_path):
                missing_count += 1
                continue
            
            try:
                image = Image.open(img_path).convert('RGB')
                for aug_idx in range(augs_needed):
                    transformed_tensor = transform(image)
                    if data_param.get('dtype') == 'uint8':
                        transformed_tensor = (transformed_tensor * 255).clamp(0, 255).to(torch.uint8)
                    images.append(transformed_tensor)
                    labels.append(LABEL_MAPPING[row['dx']])
                    original_ids.append(row['image_id'] if not balanced else f"{row['image_id']}_aug{aug_idx}")
            except Exception as e:
                missing_count += 1
                continue
                
    if missing_count > 0:
        print(f"⚠️ WARNING: {missing_count}/{len(df)} images could not be loaded from {image_dir}")
    if not images:
        raise RuntimeError(f"❌ CRITICAL: 0 images were processed from {image_dir}. Check file names and CSV IDs.")
            
    return {'images': images, 'labels': labels, 'original_ids': original_ids}


def generate_and_save_all_versions(config: dict):
    """Orchestrates version generation and serializes `.pt` files to disk."""
    # Pre-flight validation BEFORE creating output folders
    if not os.path.isfile(config['csv_file']):
        raise FileNotFoundError(f"❌ CRITICAL: CSV metadata file not found: {config['csv_file']}")
    detect_image_extension(config['image_dir'])

    os.makedirs(config['output_dir'], exist_ok=True)
    save_code_copy(config['output_dir'])
    
    dtype_map = {
        'uint8': torch.uint8, 'float32': torch.float32, 
        'float16': torch.float16, 'int64': torch.int64
    }
    target_dtype = dtype_map.get(config.get('dtype'), torch.float32)

    df = pd.read_csv(config['csv_file']).sort_values('image_id').reset_index(drop=True)
    
    existing_versions = set()
    for file_name in os.listdir(config['output_dir']):
        if file_name.startswith("version_") and file_name.endswith(".pt"):
            try:
                v_num = int(file_name.split("_")[1].split(".")[0])
                existing_versions.add(v_num)
            except Exception:
                pass
    
    print(f"🔍 Existing versions found on disk: {existing_versions}")

    # 1. UNBALANCED CONSOLIDATED MULTI-VERSION MODE ([N, V, C, H, W])
    if not config['balanced'] and config['num_versions'] > 1:
        if 0 in existing_versions:
            print("⚠️ Consolidated file (version_0.pt) already exists. Skipping...")
            return

        print(f"\n🎯 MODE: UNBALANCED CONSOLIDATED ({config['num_versions']} versions)")
        all_v_imgs, all_v_data = [], []
        
        for v in range(config['num_versions']):
            print(f"\n{'='*60}\n🔄 Generating version {v+1}/{config['num_versions']} (for consolidation)\n{'='*60}")
            d = generate_dataset_version(v, df, config['image_dir'], config['data_params'], False)
            all_v_imgs.append(torch.stack(d['images']))
            all_v_data.append(d)
        
        if all_v_imgs:
            consolidated_tensor = torch.stack(all_v_imgs, dim=1).to(dtype=target_dtype)
            consolidated = {
                'images': consolidated_tensor,
                'labels': torch.tensor(all_v_data[0]['labels']),
                'original_ids': all_v_data[0]['original_ids'],
                'metadata': {
                    'description': config['dataset_description'],
                    'num_versions': config['num_versions'],
                    'dtype': str(target_dtype),
                    'consolidated': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
            out_pt = os.path.join(config['output_dir'], "version_0.pt")
            torch.save(consolidated, out_pt)
            print(f"✅ Consolidated tensor saved: {out_pt} | Shape: {tuple(consolidated_tensor.shape)}")
        return

    # 2. BALANCED MODE OR SINGLE-VERSION UNBALANCED MODE ([N, C, H, W])
    else:
        balance_status = "BALANCED" if config['balanced'] else "UNBALANCED"
        print(f"\n🎯 MODE: {balance_status} (Individual version files)")
        
        for v in range(config['num_versions']):
            if v in existing_versions:
                print(f"⚠️ Version {v} already exists on disk. Skipping...")
                continue
            
            print(f"\n{'='*60}\n🔄 Generating version {v+1}/{config['num_versions']}\n{'='*60}")
            d = generate_dataset_version(
                v, df, config['image_dir'], config['data_params'], 
                config['balanced'], config.get('target_per_class', 6705)
            )
            
            tensor_imgs = torch.stack(d['images']).to(dtype=target_dtype)
            dataset = {
                'images': tensor_imgs,
                'labels': torch.tensor(d['labels']),
                'original_ids': d['original_ids'],
                'metadata': {
                    'description': config['dataset_description'],
                    'version': v,
                    'dtype': str(target_dtype),
                    'balanced': config['balanced'],
                    'timestamp': datetime.now().isoformat()
                }
            }
            out_pt = os.path.join(config['output_dir'], f"version_{v}.pt")
            torch.save(dataset, out_pt)
            print(f"✅ Version {v} saved: {out_pt} | Shape: {tuple(tensor_imgs.shape)}")
            gc.collect()


def configure_and_generate_dataset(
    output_dir: str, dataset_description: str = "Custom Dataset", image_size = 224, 
    num_versions: int = 1, balanced: bool = False, target_per_class: int = 6705, 
    flipHV: bool = False, jitter: bool = False, gaussian: bool = False,
    rotation90: bool = False, crop_type: str = 'none', augs_per_image: int = 1, 
    normalize: bool = False, dtype: str = 'float32', csv_file: str = None, image_dir: str = None
):
    """High-level wrapper to configure and trigger tensor dataset generation."""
    config = {
        "csv_file": csv_file,
        "image_dir": image_dir,
        "output_dir": output_dir,
        "dataset_description": dataset_description,
        "num_versions": num_versions,
        "balanced": balanced,
        "target_per_class": target_per_class,
        "image_size": image_size,
        "dtype": dtype,
        "data_params": {
            "size": image_size, "flipHV": flipHV, "jitter": jitter, "gaussian": gaussian,
            "rotation90": rotation90, "crop_type": crop_type, "augs_per_image": augs_per_image,
            "normalize": normalize, "dtype": dtype
        }
    }
    print(f"\n🚀 Initializing configuration: {dataset_description}")
    generate_and_save_all_versions(config)


# ==========================================================
# --- 2. PRESET GENERATORS (PAPER REPRODUCIBILITY & TOOLS) ---
# ==========================================================

def build_standard_experiment_suite(image_size: int = 224, hair_removed: bool = False):
    """
    Generates the complete 3-partition suite (Train x10 uint8, Validation, and Test) 
    used in the study for a given spatial resolution (e.g., 224, 256, 384) and domain 
    condition (With Hair vs. DullRazor Hair-Removed).
    """
    suffix = "_hair_removed" if hair_removed else ""
    train_img_dir = (
        os.path.join(DATASET_ROOT, "HAM10000_images_dullrazor", "tiff_images")
        if hair_removed else os.path.join(DATASET_ROOT, "HAM10000_images")
    )
    test_img_dir = (
        os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_Images_dullrazor", "tiff_images")
        if hair_removed else os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_Images")
    )

    # 1. Training Partition (10 consolidated random-crop views in uint8)
    configure_and_generate_dataset(
        output_dir=os.path.join(TENSORS_OUT_ROOT, f"{image_size}x{image_size}_Rcrop_unbalanced{suffix}_x10_uint8"),
        dataset_description=f"Training set ({image_size}x{image_size}), 10 random-crop views, unbalanced, uint8 ({'Hair-Removed' if hair_removed else 'Raw'})",
        image_size=image_size,
        num_versions=10,
        balanced=False,
        crop_type='random',
        normalize=False,
        dtype='uint8',
        csv_file=os.path.join(DATASET_ROOT, "HAM10000_metadata_train.csv"),
        image_dir=train_img_dir
    )

    # 2. Validation Partition (Deterministic resize, 1 version)
    configure_and_generate_dataset(
        output_dir=os.path.join(TENSORS_OUT_ROOT, f"{image_size}x{image_size}_val{suffix}"),
        dataset_description=f"Validation set ({image_size}x{image_size}), deterministic resize, float32 ({'Hair-Removed' if hair_removed else 'Raw'})",
        image_size=image_size,
        num_versions=1,
        balanced=False,
        crop_type='resize',
        normalize=False,
        dtype='float32',
        csv_file=os.path.join(DATASET_ROOT, "HAM10000_metadata_val.csv"),
        image_dir=train_img_dir
    )

    # 3. Test Partition - ISIC 2018 Task 3 GroundTruth (Deterministic resize, 1 version)
    configure_and_generate_dataset(
        output_dir=os.path.join(TENSORS_OUT_ROOT, f"groundtruth_{image_size}x{image_size}{suffix}"),
        dataset_description=f"ISIC 2018 Test set ({image_size}x{image_size}), deterministic resize, float32 ({'Hair-Removed' if hair_removed else 'Raw'})",
        image_size=image_size,
        num_versions=1,
        balanced=False,
        crop_type='resize',
        normalize=False,
        dtype='float32',
        csv_file=os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_GroundTruth.csv"),
        image_dir=test_img_dir
    )


def build_native_600x450_unbalanced(hair_removed: bool = False):
    """
    Utility Preset: High-resolution native dataset (600x450, 1 version, uint8).
    Designed to be loaded into RAM for 100% on-the-fly Data Augmentation inside the DataLoader.
    """
    suffix = "_hair_removed" if hair_removed else ""
    train_img_dir = (
        os.path.join(DATASET_ROOT, "HAM10000_images_dullrazor", "tiff_images")
        if hair_removed else os.path.join(DATASET_ROOT, "HAM10000_images")
    )
    configure_and_generate_dataset(
        output_dir=os.path.join(TENSORS_OUT_ROOT, f"600x450_native_unbalanced{suffix}_uint8"),
        dataset_description="Native 600x450 resolution, unbalanced, single version uint8 for live on-the-fly DA.",
        image_size=(450, 600),
        num_versions=1,
        balanced=False,
        crop_type='none',
        normalize=False,
        dtype='uint8',
        csv_file=os.path.join(DATASET_ROOT, "HAM10000_metadata_train.csv"),
        image_dir=train_img_dir
    )


def build_static_balanced_full_da(image_size: int = 256):
    """
    Utility Preset: Statically balanced dataset via offline Data Augmentation (6,705 samples/class).
    Generates 10 separate version files with random crops, 90-deg rotations, flips, jitter, and blur.
    """
    configure_and_generate_dataset(
        output_dir=os.path.join(TENSORS_OUT_ROOT, f"{image_size}x{image_size}_Rcrop_fullDA_balanced"),
        dataset_description=f"Statically balanced dataset ({image_size}x{image_size}) with full offline DA across 10 separate versions.",
        image_size=image_size,
        num_versions=10,
        balanced=True,
        target_per_class=6705,
        flipHV=True,
        rotation90=True,
        jitter=True,
        gaussian=True,
        crop_type='random',
        normalize=False,
        dtype='float32',
        csv_file=os.path.join(DATASET_ROOT, "HAM10000_metadata_train.csv"),
        image_dir=os.path.join(DATASET_ROOT, "HAM10000_images")
    )


if __name__ == "__main__":
    # --- 1. Generate Paper Experiment Suites (224x224 & 256x256, Raw & Hair-Removed) ---
    build_standard_experiment_suite(image_size=224, hair_removed=False)
    build_standard_experiment_suite(image_size=224, hair_removed=True)
    build_standard_experiment_suite(image_size=256, hair_removed=False)
    build_standard_experiment_suite(image_size=256, hair_removed=True)
    
    # Uncomment for 384x384 architectures:
    # build_standard_experiment_suite(image_size=384, hair_removed=False)
    # build_standard_experiment_suite(image_size=384, hair_removed=True)

    # --- 2. Optional Utility Presets ---
    # build_native_600x450_unbalanced(hair_removed=False)
    # build_static_balanced_full_da(image_size=256)