"""
=========================================================================================
MODULE: Automated Cross-Domain Evaluation Runner (verify_models_cross_domain.py)
=========================================================================================
DESCRIPTION:
    Automatically audits trained model directories, identifies their original training 
    domain (Raw With-Hair vs. DullRazor Hair-Removed), and executes full evaluation 
    suites (Test and Validation metrics + Grad-CAMs) on the OPPOSITE target domain.
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
from configure_environment import CLASS_COUNTS_GROUNDTRUTH as class_counts_groundtruth
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

class CrossDomainEvaluationLogger:
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
# CORE CROSS-DOMAIN EVALUATION ENGINE
# ==============================================================================

def evaluate_cross_domain_model(folder_name: str, base_directory: str, tasks_to_run: list, cross_subdir_name: str = "Cross_Domain_Evaluation"):
    """
    Identifies the training domain of an experiment directory and evaluates all 
    checkpoints on the opposite hair condition domain across Test and Validation sets.
    """
    base_save_dir = os.path.join(base_directory, folder_name)
    
    # 1. Identify original training domain
    original_hair_removed = detect_training_hair_condition(base_save_dir)
    if original_hair_removed is None:
        print(f"⚠️ SKIPPING: {folder_name} (Could not determine original training hair condition)")
        return

    # 2. Invert domain condition for cross-domain testing
    cross_hair_removed = not original_hair_removed
    
    # Reuse legacy 'Verificacion_Cruzada' folder if it already exists on disk, otherwise use standard English name
    legacy_cross_dir = os.path.join(base_save_dir, "Verificacion_Cruzada")
    cross_dir = legacy_cross_dir if os.path.exists(legacy_cross_dir) else os.path.join(base_save_dir, cross_subdir_name)

    # Clean up any duplicate 'final_model_weights' folders inside Cross_Domain_Evaluation/Results_*
    for ds_folder, _ in tasks_to_run:
        normalize_final_subfolder(os.path.join(cross_dir, ds_folder))

    # 3. Discover and sort checkpoint weights (.pth)
    weight_files = discover_checkpoint_files(base_save_dir)
    
    def extract_epoch(filepath: str) -> int:
        match = re.search(r'\[(\d+)\]', filepath)
        return int(match.group(1)) if match else 999

    weight_files.sort(
        key=lambda x: (
            0 if 'MULTI' in x else (2 if 'final' in os.path.basename(x).lower() else 1), 
            extract_epoch(x)
        )
    )

    if not weight_files:
        print(f"❌ No checkpoint weights (.pth) found in {folder_name}.")
        return

    # 4. Smart global filter: check if any cross-domain reports are missing
    pending_weights = []
    for weights_path in weight_files:
        weight_tag = extract_checkpoint_tag(weights_path)
        if not all(glob.glob(os.path.join(cross_dir, ds, weight_tag, "score_*.csv")) for ds, _ in tasks_to_run):
            pending_weights.append(weights_path)
            
    if not pending_weights:
        print(f"⏭️ UP TO DATE: {folder_name} already has all cross-domain evaluations completed. Skipping...")
        return

    os.makedirs(cross_dir, exist_ok=True)
    log_file = os.path.join(cross_dir, f"log_cross_domain_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = CrossDomainEvaluationLogger(log_file)

    model = None
    loader_map = None

    try:
        # 5. Configure dynamic dataset parameters for the target cross-domain
        data_param['hair_removed'] = cross_hair_removed
        data_param['afeitado'] = cross_hair_removed
        
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
            print(f"❌ SKIPPING: Unrecognized architecture in {folder_name}")
            return

        source_domain_str = "Hair-Removed (DullRazor)" if original_hair_removed else "Raw (With Hair)"
        target_domain_str = "Hair-Removed (DullRazor)" if cross_hair_removed else "Raw (With Hair)"

        print("\n" + "═" * 70)
        print("🚀 STARTING AUTOMATED CROSS-DOMAIN EVALUATION")
        print(f"📂 Model Directory: {folder_name}")
        print(f"🎯 Architecture:    {model_variant.upper()} | Resolution: {data_param['size']}x{data_param['size']}")
        print(f"🔄 Domain Shift:    Trained on [{source_domain_str}] ---> Evaluating on [{target_domain_str}]")
        print("═" * 70)

        # 6. Instantiate Cross-Domain DataLoaders (Test and Val)
        print(f"📥 Instantiating cross-domain datasets (hair_removed={cross_hair_removed})...")
        requested_modes = {mode_str for _, mode_str in tasks_to_run}
        loader_map = {
            mode_str: DataLoader(load_dataset(data_param, mode_str), batch_size=64, shuffle=False)
            for mode_str in requested_modes
        }
        criterion_test = nn.CrossEntropyLoss().to(hardware_device)

        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}', folder_name)
        start_time_str = timestamp_match.group(0) if timestamp_match else time.strftime('%Y-%m-%d-%H-%M')

        # 7. Run evaluations for pending checkpoints
        for weights_path in pending_weights:
            filename = os.path.basename(weights_path)
            weight_tag = extract_checkpoint_tag(weights_path)
            
            print(f"\n🔍 Checkpoint: [{weight_tag}] ({filename})")
            
            try:
                state_dict = torch.load(weights_path, map_location=hardware_device, weights_only=True)
                cleaned_state_dict = {
                    k[6:] if k.startswith('model.') else k: v 
                    for k, v in state_dict.items()
                }
                model.load_state_dict(cleaned_state_dict, strict=True)
            except Exception as e:
                print(f"   ❌ Error mapping checkpoint weights: {e}")
                continue

            for ds_folder, mode_str in tasks_to_run:
                save_path = os.path.join(cross_dir, ds_folder, weight_tag)
                
                if glob.glob(os.path.join(save_path, "score_*.csv")):
                    print(f"   ⏭️ SKIPPING: Cross-Domain {mode_str.upper()} (Report already exists)")
                    continue
                
                print(f"   ⚙️ Evaluating Cross-Domain {mode_str.upper()}...")
                test_model(
                    start_time=start_time_str,
                    model=model,
                    data_loader=loader_map[mode_str],
                    criterion_test=criterion_test,
                    class_counts=class_counts_groundtruth,
                    model_weights_path=None,
                    save_folder=save_path,
                    mode=mode_str,
                    use_gradcam=True,
                    model_variant=model_variant
                )

    finally:
        if isinstance(sys.stdout, CrossDomainEvaluationLogger):
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal
            
        if model is not None:
            del model
        if loader_map is not None:
            del loader_map
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# BATCH EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    TASKS_TO_RUN = [
        ('Results_TEST', 'test'),
        ('Results_VAL', 'val')
    ]

    base_directories = [MODELS_DIR]
    
    print("🔬 Starting Automated CROSS-DOMAIN Evaluation Runner...")
    models_to_evaluate = []
    
    for base_dir in base_directories:
        if not os.path.exists(base_dir):
            print(f"❌ Directory not found: {base_dir}")
            continue
            
        print(f"📂 Scanning base directory recursively: {base_dir}")
        for root, dirs, _ in os.walk(base_dir):
            for folder in dirs:
                if re.match(r'\d{4}-\d{2}-\d{2}', folder):
                    full_path = os.path.join(root, folder)
                    if discover_checkpoint_files(full_path):
                        models_to_evaluate.append((folder, root))
                    
    print(f"\n🎯 Detected {len(models_to_evaluate)} valid model directories for cross-domain testing.\n")
    
    for model_folder, parent_dir in models_to_evaluate:
        evaluate_cross_domain_model(model_folder, parent_dir, TASKS_TO_RUN)
        
    print("\n🏁 CROSS-DOMAIN EVALUATION COMPLETE. All Test and Validation reports are up to date.")