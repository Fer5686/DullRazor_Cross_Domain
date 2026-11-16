"""
=========================================================================================
MODULE: ConvNeXt End-to-End Fine-Tuning Pipeline (finetuning_training_convnext.py)
=========================================================================================
DESCRIPTION:
    Orchestrates the three-phase progressive fine-tuning and evaluation protocol for 
    the modernized convolutional architecture (ConvNeXt, 'convnext_small.fb_in1k' via 
    timm, ~50.2M parameters).
    
    Methodological Specifications (see Manuscript Section 'Selection of Neural 
    Architectures' & Table 3):
    1. Spatial Resolution & Microstructural Design: Operates on 256x256 input tensors, 
       integrating 7x7 depthwise separable convolutions, LayerNorm2d, GELU activations, 
       and linearly distributed stochastic depth regularization (DropPath max = 0.4).
    2. Three-Phase Progressive Unfreezing:
       - Phase 1 (Warmup Head): Frozen convolutional backbone; trains only the classifier 
         head for up to 150 epochs (LR = 3e-3).
       - Phase 2 (Last 3 Blocks of Final Stage): Unfreezes the last 3 blocks of Stage 3 
         for up to 100 epochs (LR = 1e-4) to refine high-level semantic representations.
       - Phase 3 (Full Model): Unfreezes the entire ConvNeXt topology for global 
         fine-tuning for up to 150 epochs using a conservative learning rate (LR = 5e-6).
    3. Training Engine: Utilizes AdamW (weight decay = 0.05), Exponential LR decay 
       (gamma = 0.99, reset per phase), Automatic Mixed Precision (AMP float16), and 
       Curriculum Learning (dynamic RandAugment and CutMix).
    4. Early Stopping & Phase Synchronization: Guided by validation F2-Score with a 
       patience of 20 epochs. Reloads the best phase checkpoint prior to advancing to 
       the next unfreezing stage.
=========================================================================================
"""

import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

# Custom module imports
from device_config import hardware_device
from saving_functions import (
    save_model, save_data, process_plots, 
    save_initial_info, create_training_directory, log_speed_txt
)
from train import train_model_dynamic_cutmix
from val import validate_model, compute_model_metrics
from test import test_model
from instantiate_datasets import load_dataset, label_mapping, data_param as dp_import

# ⭐ IMPORTANT: Points to the custom ConvNeXt architecture module
from models.convnext import create_model, count_parameters 

from optimizer_scheduler import configure_optimizer_and_scheduler
from configure_environment import setup_training_environment
from early_stopping import EarlyStoppingManager


def train_convnext_end_to_end(main_param):
    """
    Executes the complete 3-phase fine-tuning lifecycle and multi-partition evaluation 
    for the ConvNeXt topology under the specified random seed and domain condition.
    """
    # ---------- 1. CONFIGURATION, SEED AND PHASES ----------
    seed = main_param.get("seed", random.randint(1, 1000000))
    setup_reproducibility(seed)
    
    # Extract debug mode flag
    debug_mode = main_param.get("debug_mode", False)
    
    dropout_rate = 0.3
    batch_size = 64
    
    # Define progressive unfreezing phases
    phases = [
        {"name": "P1_WARMUP_HEAD", "lr": 3e-3, "unfreeze": "none",    "epochs": 150},
        {"name": "P2_LAST_BLOCKS", "lr": 1e-4, "unfreeze": "partial", "epochs": 100},
        {"name": "P3_FULL_MODEL",  "lr": 5e-6, "unfreeze": "full",    "epochs": 150}
    ]
    
    # Scale down epochs and patience if debug_mode is active
    if debug_mode:
        for phase in phases:
            phase["epochs"] = max(1, phase["epochs"] // 10)  # Divide by 10 (e.g., 15, 10, 15)
        es_patience = 3
        print("🐛 [DEBUG MODE ACTIVE] Max epochs divided by 10 | Early Stopping patience = 3")
    else:
        es_patience = 20

    # Total estimated epochs is dynamically computed as the exact sum of all phase epochs
    total_estimated_epochs = sum(phase["epochs"] for phase in phases)
    
    # ---------- 2. INITIAL MODEL INSTANTIATION ----------
    model = create_model().to(hardware_device)
    print(f"✅ ConvNeXt model instantiated. Trainable parameters: {count_parameters(model)}")

    # ---------- 3. DATASETS AND ENVIRONMENT ----------
    train_ds = load_dataset(dp_import, dataset_type='train')
    val_ds = load_dataset(dp_import, dataset_type='val')
    test_ds = load_dataset(dp_import, dataset_type='test')

    env = setup_training_environment(
        main_param=main_param, 
        data_param=dp_import,
        hardware_device=hardware_device,
        dataset_train=train_ds, 
        dataset_val=val_ds, 
        dataset_test=test_ds,
        batch_size=batch_size, 
        epochs=total_estimated_epochs 
    )
    
    train_loader, val_loader, test_loader = env['train_loader'], env['val_loader'], env['test_loader']
    criterion_train, criterion_val, class_counts = env['criterion_train'], env['criterion_val'], env['class_counts']
    images_per_epoch = env['images_per_epoch']

    # ---------- 4. LOGGING AND DIRECTORY PREPARATION ----------
    start_time = time.time()
    output_dir = create_training_directory(start_time, model)
    
    # Move the active console log inside the newly created training directory
    if hasattr(sys.stdout, 'relocate'):
        sys.stdout.relocate(output_dir)
        
    print(f"🎬 End-to-End ConvNeXt | Strategy: {main_param['loss_strategy']} | Total Max Epochs: {total_estimated_epochs} | Output Dir: {output_dir}")
    
    early_stopping = EarlyStoppingManager(
        patience=es_patience, 
        start_time=start_time, 
        output_dir=output_dir, 
        save_func=save_model
    )

    train_losses, val_losses, conf_matrix_list = [], [], []
    history = {
        'epochs': [], 'train_loss': [], 'val_loss': [], 'class_f2': [], 
        'precision': [], 'recall': [], 'f1_score': [], 'f2_score': [], 
        'mel_f2': [], 'learning_rates': []
    }

    # ---------- 5. PROGRESSIVE UNFREEZING PHASES ----------
    # Phase 1: Trains ONLY the classifier head (LR = 3e-3).
    # Phase 2: Unfreezes the last 3 blocks of the final stage (LR = 1e-4).
    # Phase 3: Unfreezes the entire ConvNeXt backbone for global fine-tuning (LR = 5e-6).
    global_epoch = 0 

    # ========== PHASE TRAINING LOOP ==========
    for i, phase in enumerate(phases):
        print(f"\n{'='*50}\n🚀 PHASE: {phase['name']} | Unfreeze: {phase['unfreeze']} | Max Epochs: {phase['epochs']}\n{'='*50}")
        
        # Reset Early Stopping (patience and stop flag) for the new phase
        early_stopping.reset()
        
        # 2. Configure backbone unfreezing
        if phase["unfreeze"] == "none": 
            model.freeze_backbone()
            print("🛡️ Backbone FROZEN. Training classifier head only.")
        elif phase["unfreeze"] == "partial": 
            model.unfreeze_backbone(last_n_blocks=3) 
            print("🔓 Backbone PARTIALLY unfrozen (Last 3 blocks of Stage 3).")
        else: 
            model.unfreeze_backbone(last_n_blocks=None)
            print("🔥 Backbone FULLY unfrozen. Global fine-tuning active.")

        # 3. Re-instantiate Optimizer and Scheduler
        optimizer, scheduler, step_type = configure_optimizer_and_scheduler(
            model=model, 
            optim_name=main_param["optim"], 
            lr=phase["lr"],
            scheduler_name=main_param["scheduler"], 
            epochs=phase["epochs"],
            steps_per_epoch=len(train_loader)
        )

        # 4. Save initial architectural and hyperparameter metadata on the very first epoch
        if i == 0 and global_epoch == 0:
            save_initial_info(
                start_time, optimizer, scheduler, dropout_rate, None, model, 
                False, seed, model_fn=create_model
            )

        for phase_epoch in range(phase["epochs"]):
            # Dynamic training step (with Curriculum CutMix/RandAugment & AMP)
            res, t_loss, t_preds, t_labels, _, cur_lr = train_model_dynamic_cutmix(
                model, train_loader, optimizer, criterion_train, None, 
                False, use_amp=True, epoch=global_epoch, 
                epochs=total_estimated_epochs, scheduler=scheduler, use_curriculum=True
            )
            
            if res == 'nan': 
                print("💥 Error: Loss is NaN. Aborting training run.")
                return 'nan'

            # Validation step
            v_loss, v_preds, v_labels = validate_model(model, val_loader, criterion_val)
            prec, cl_f2, f1, rec, cm, f2, mel_f2 = compute_model_metrics(
                v_preds, v_labels, label_mapping, class_counts, 'train_val'
            )

            # Update training history
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            conf_matrix_list.append(cm)
            history['epochs'].append(global_epoch)
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['f1_score'].append(f1)
            history['f2_score'].append(f2)
            history['mel_f2'].append(mel_f2)
            history['class_f2'].append(cl_f2)
            history['learning_rates'].append(cur_lr)

            # 5. Update Early Stopping state
            stop_now = early_stopping.update(
                {'val_loss': v_loss, 'f1_score': f1, 'f2_score': f2, 'mel_f2': mel_f2}, 
                model, 
                global_epoch
            )
            
            global_epoch += 1
            if stop_now:
                print(f"🛑 Early Stopping triggered in phase {phase['name']}. Advancing to next phase...")
                break
        
        # 6. Phase Synchronization: reload best weights from current phase before advancing
        load_best_phase_by_strategy(model, output_dir, main_param['loss_strategy'])
        
    # ---------- 6. TRAINING FINALIZATION ----------
    print("\n🎬 Finalizing End-to-End training process.")
    finalize_training(
        start_time, global_epoch - 1, train_losses, val_losses, conf_matrix_list, 
        history, images_per_epoch, model, main_param, 
        train_loader, val_loader, test_loader, criterion_val, criterion_train, 
        class_counts, output_dir, early_stopping
    )
    return 'finish'


# --- AUXILIARY FUNCTIONS ---

def setup_reproducibility(seed):
    """
    Fixes pseudorandom generator states across Python, NumPy, and PyTorch (CPU/GPU) 
    to minimize stochastic initialization variance across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_best_phase_by_strategy(model, output_dir, strategy):
    """
    Identifies and loads the highest-performing model checkpoint from the completed phase 
    according to the active optimization metric (generalist f2_score vs. mel_f2).
    """
    tag = 'mel_f2' if strategy == 'mel_boost' else 'f2_score'
    files = [f for f in os.listdir(output_dir) if tag in f and f.endswith('.pth')]
    
    if files:
        files.sort()
        best_path = os.path.join(output_dir, files[-1])
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=hardware_device))
        print(f"🔄 Phase Synchronization: Loaded best {tag} checkpoint -> {os.path.basename(best_path)}")


def finalize_training(start_time, epoch, train_losses, val_losses, conf_matrix, 
                      history, images_per_epoch, model, main_param, 
                      train_loader, val_loader, test_loader, criterion_val, criterion_train, 
                      class_counts, output_dir, early_stopping_manager):
    """
    Saves final training artifacts, prunes intermediate checkpoints, and executes 
    post-training evaluation across TRAIN, VAL, and TEST partitions (including Grad-CAM).
    """
    print("\n" + "="*60 + "\n🏁 STARTING MULTI-MODEL POST-TRAINING EVALUATION\n" + "="*60)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_processed_images = images_per_epoch * (epoch + 1)

    save_model(model, start_time, epoch, delete_previous_weights=False, base_name='final_model_weights')
    save_data(start_time, elapsed_time, epoch, train_losses, val_losses, conf_matrix, history)
    log_speed_txt(start_time, epoch, elapsed_time, total_processed_images)
    process_plots(start_time)
    
    if early_stopping_manager is not None:
        early_stopping_manager.prune_models()
    
    eval_tasks = []
    if main_param.get("final_test", True):
        eval_tasks.append(('TRAIN', train_loader, criterion_train))
        if val_loader is not None: 
            eval_tasks.append(('VAL', val_loader, criterion_val))
        if test_loader is not None: 
            eval_tasks.append(('TEST', test_loader, criterion_val))

    weight_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]

    for ds_name, loader, crit in eval_tasks:
        print(f"\n📂 EVALUATING PARTITION: {ds_name}")
        for weight_file in weight_files:
            # Strip both English and legacy Spanish prefixes to extract the clean metric tag
            weight_tag = (
                weight_file.split('-')[0]
                .replace('final_model_weights', 'final')
                .replace('model_weights_', '')
                .replace('pesos_modelo_', '')
            )
            specific_save_path = os.path.join(output_dir, f"Results_{ds_name}", weight_tag)
            pth_path = os.path.join(output_dir, weight_file)
            
            test_model(
                start_time=start_time,
                model=model,
                data_loader=loader,
                criterion_test=crit,
                class_counts=class_counts,
                model_weights_path=pth_path,
                save_folder=specific_save_path,
                mode=ds_name.lower()
            )