"""
=========================================================================================
MODULE: Automated Checkpoint & Evaluation Recovery Tool (verify_models.py)
=========================================================================================
DESCRIPTION:
    Scans experiment directories for saved model checkpoints (.pth) and verifies 
    whether complete evaluation suites (Test, Validation, Train metrics and Grad-CAMs) 
    have been generated on disk.
=========================================================================================
"""

import os
import sys
import glob
import re
import time
import gc
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==============================================================================
# CROSS-DIRECTORY PATH RESOLUTION (Tools/ -> Root & Main_code/)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from device_config import hardware_device
from test import test_model
from instantiate_datasets import load_dataset, data_param
from configure_environment import (
    CLASS_COUNTS_GROUNDTRUTH as class_counts_groundtruth,
    CLASS_COUNTS_VAL as class_counts_val,
    CLASS_COUNTS_TRAIN as class_counts_train
)
import config_paths

MODELS_DIR = getattr(config_paths, "MODELS_DIR", getattr(config_paths, "REPO_DIR"))

# ==============================================================================
# MODEL ARCHITECTURE IMPORTS
# ==============================================================================
from models.resnet50 import create_model as create_resnet
from models.convnext import create_model as create_convnext
from models.swin import create_model as create_swin
from models.vgg16 import create_model as create_vgg
from models.alexnet import create_model as create_alexnet

try:
    from models.vision_mamba import create_model as create_vim
except ImportError:
    create_vim = None


# ==============================================================================
# AUXILIARY CLASSES & HELPER FUNCTIONS
# ==============================================================================

class ManualVerificationLogger:
    """Redirects standard output to both the console terminal and a persistent log file."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def discover_checkpoint_files(folder_path: str) -> list:
    """Finds all standard, final, and legacy checkpoint .pth files in a directory."""
    patterns = (
        "model_weights_*.pth",
        "final_model_weights*.pth",
        "pesos_modelo_*.pth",
        "pesos_modelo_final*.pth"
    )
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(folder_path, pat)))
    return sorted(list(set(found)))


def detect_training_hair_condition(base_save_dir: str):
    """
    Infers whether the model was originally trained on Hair-Removed (True) or 
    Raw With-Hair (False) images by inspecting log filenames or parameter dumps.
    """
    log_files = glob.glob(os.path.join(base_save_dir, "log_*.txt"))
    for log_file in log_files:
        log_name = os.path.basename(log_file).lower()
        if any(k in log_name for k in ['withouthair', 'without_hair', 'hair_removed', 'sinpelos', 'sin_pelos']):
            return True
        if any(k in log_name for k in ['withhair', 'with_hair', 'conpelos', 'con_pelos']):
            return False

    info_files = (
        glob.glob(os.path.join(base_save_dir, "info_param*.txt")) +
        glob.glob(os.path.join(base_save_dir, "param_info*.txt")) +
        glob.glob(os.path.join(base_save_dir, "initial_info*.txt")) +
        glob.glob(os.path.join(base_save_dir, "info_inicio*.txt"))
    )
    for info_path in info_files:
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                content = f.read().lower().replace(" ", "")
                if any(k in content for k in ['"hair_removed":true', "'hair_removed':true", '"afeitado":true', "'afeitado':true"]):
                    return True
                if any(k in content for k in ['"hair_removed":false', "'hair_removed':false", '"afeitado":false', "'afeitado':false"]):
                    return False
        except Exception:
            pass
            
    return None


def extract_checkpoint_tag(weights_path: str) -> str:
    """
    Extracts the metric tag (e.g., 'val_loss', 'f1_score', 'MULTI', 'final') 
    from a checkpoint filename.
    """
    basename = os.path.basename(weights_path).split('-')[0]
    if basename in ('final_model_weights', 'pesos_modelo_final'):
        return 'final'
    for prefix in ('model_weights_', 'pesos_modelo_'):
        if basename.startswith(prefix):
            return basename.replace(prefix, '')
    return basename


def normalize_final_subfolder(parent_results_dir: str):
    """
    Merges duplicate 'final_model_weights' or 'pesos_modelo_final' subfolders 
    into the standardized 'final' folder.
    """
    if not os.path.exists(parent_results_dir):
        return
    target_final = os.path.join(parent_results_dir, "final")
    for dup_name in ("final_model_weights", "pesos_modelo_final"):
        dup_path = os.path.join(parent_results_dir, dup_name)
        if os.path.exists(dup_path):
            if not os.path.exists(target_final):
                shutil.move(dup_path, target_final)
            else:
                shutil.rmtree(dup_path, ignore_errors=True)


# ==============================================================================
# CORE NATIVE VERIFICATION ENGINE
# ==============================================================================

def verify_native_model(folder_name: str, base_directory: str, tasks_to_run: list):
    """
    Audits a single experiment directory and runs missing evaluation suites across 
    all discovered checkpoints.
    """
    base_save_dir = os.path.join(base_directory, folder_name)
    
    native_hair_removed = detect_training_hair_condition(base_save_dir)
    if native_hair_removed is None:
        print(f"⚠️ SKIPPING: {folder_name} (Could not determine original hair domain condition)")
        return
    
    # 1. Flatten legacy manual verification folders
    for legacy_sub in ("Manual_Verification", "Verificacion_Manual"):
        manual_dir = os.path.join(base_save_dir, legacy_sub)
        if os.path.exists(manual_dir):
            print(f"🧹 Normalizing directory structure for {folder_name} ({legacy_sub})...")
            for root, _, files in os.walk(manual_dir):
                for file in files:
                    src_file = os.path.join(root, file)
                    rel_path = os.path.relpath(src_file, manual_dir)
                    dst_file = os.path.join(base_save_dir, rel_path)
                    
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    if not os.path.exists(dst_file):
                        shutil.move(src_file, dst_file)
                    else:
                        os.remove(src_file)
            shutil.rmtree(manual_dir, ignore_errors=True)

    # 2. Clean up any duplicate 'final_model_weights' folders inside Results_*
    for ds_folder, _ in tasks_to_run:
        normalize_final_subfolder(os.path.join(base_save_dir, ds_folder))

    # 3. Discover all checkpoints
    weight_files = discover_checkpoint_files(base_save_dir)
    weight_files.sort(
        key=lambda x: (
            0 if 'MULTI' in x else (2 if 'final' in os.path.basename(x).lower() else 1), 
            int(re.search(r'\[(\d+)\]', x).group(1)) if re.search(r'\[(\d+)\]', x) else 999
        )
    )

    if not weight_files:
        print(f"❌ No checkpoint weights (.pth) found in {folder_name}.")
        return

    pending_weights = []
    for weights_path in weight_files:
        weight_tag = extract_checkpoint_tag(weights_path)
        if not all(glob.glob(os.path.join(base_save_dir, ds, weight_tag, "score_*.csv")) for ds, _ in tasks_to_run):
            pending_weights.append(weights_path)
    
    if not pending_weights:
        print(f"⏭️ SKIPPING: {folder_name} (All requested native evaluations already exist on disk).")
        return

    data_param['hair_removed'] = native_hair_removed
    data_param['afeitado'] = native_hair_removed
    folder_lower = folder_name.lower()
    
    if 'resnet' in folder_lower:
        model = create_resnet().to(hardware_device)
        model_variant = 'resnet50'
        data_param['size'] = 224
    elif 'convnext' in folder_lower:
        model = create_convnext().to(hardware_device)
        model_variant = 'convnext'
        data_param['size'] = 256
    elif 'swin' in folder_lower:
        model = create_swin().to(hardware_device)
        model_variant = 'swin'
        data_param['size'] = 224
    elif 'vgg' in folder_lower:
        model = create_vgg().to(hardware_device)
        model_variant = 'vgg'
        data_param['size'] = 224
    elif 'alexnet' in folder_lower:
        model = create_alexnet().to(hardware_device)
        model_variant = 'alexnet'
        data_param['size'] = 224
    elif ('vim' in folder_lower or 'vision' in folder_lower) and create_vim is not None:
        model = create_vim().to(hardware_device)
        model_variant = 'visionmamba'
        data_param['size'] = 224
    else:
        print(f"❌ Unrecognized architecture in folder name: {folder_name}")
        return

    print(f"\n📥 Instantiating Native DataLoaders for: {folder_name} (Hair-Removed: {native_hair_removed})...")
    requested_modes = {mode_str for _, mode_str in tasks_to_run}
    loader_map = {
        mode_str: DataLoader(load_dataset(data_param, mode_str), batch_size=64, shuffle=False)
        for mode_str in requested_modes
    }
    criterion_test = nn.CrossEntropyLoss().to(hardware_device)

    for weights_path in pending_weights:
        weight_tag = extract_checkpoint_tag(weights_path)
        log_path = os.path.join(base_save_dir, f"log_native_verif_{weight_tag}.txt")
        
        sys.stdout = ManualVerificationLogger(log_path)
        try:
            state_dict = torch.load(weights_path, map_location=hardware_device, weights_only=True)
            cleaned_state_dict = {
                k[6:] if k.startswith('model.') else k: v 
                for k, v in state_dict.items()
            }
            model.load_state_dict(cleaned_state_dict, strict=True)

            for ds_folder, mode_str in tasks_to_run:
                save_root_path = os.path.join(base_save_dir, ds_folder, weight_tag)
                
                if glob.glob(os.path.join(save_root_path, "score_*.csv")):
                    print(f"⏭️ SKIPPING: {mode_str.upper()} for [{weight_tag}] (Report already exists)")
                    continue
                
                print(f"\n⚙️ Evaluating Native {mode_str.upper()} | Checkpoint: [{weight_tag}]")
                test_model(
                    start_time=time.strftime('%Y-%m-%d-%H-%M'),
                    model=model,
                    data_loader=loader_map[mode_str],
                    criterion_test=criterion_test,
                    class_counts=class_counts_groundtruth,
                    model_weights_path=None,
                    save_folder=save_root_path,
                    mode=mode_str,
                    use_gradcam=True,
                    model_variant=model_variant
                )
        finally:
            if isinstance(sys.stdout, ManualVerificationLogger):
                sys.stdout.log.close()
                sys.stdout = sys.stdout.terminal
                
    del model, loader_map
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    RUN_TEST = True
    RUN_VAL = True
    RUN_TRAIN = False

    tasks_to_run = []
    if RUN_TEST:  tasks_to_run.append(('Results_TEST', 'test'))
    if RUN_VAL:   tasks_to_run.append(('Results_VAL', 'val'))
    if RUN_TRAIN: tasks_to_run.append(('Results_TRAIN', 'train'))

    base_directories = [MODELS_DIR]
    
    print("🤖 Starting Automated NATIVE Checkpoint Batch Verification Runner...")
    discovered_models = []
    
    for base_dir in base_directories:
        if not os.path.exists(base_dir):
            print(f"❌ Base directory not found: {base_dir}")
            continue
            
        print(f"📂 Scanning directory recursively: {base_dir}")
        for root, dirs, _ in os.walk(base_dir):
            for folder in dirs:
                if re.match(r'\d{4}-\d{2}-\d{2}', folder):
                    full_path = os.path.join(root, folder)
                    if discover_checkpoint_files(full_path):
                        discovered_models.append((folder, root))
                    
    print(f"\n🎯 Detected {len(discovered_models)} experiment directories ready for native verification.\n")
    
    for folder_name, parent_dir in discovered_models:
        verify_native_model(folder_name, parent_dir, tasks_to_run)
        
    print("\n🏁 NATIVE VERIFICATION COMPLETE. All native evaluation reports are up to date.")