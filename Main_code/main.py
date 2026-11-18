"""
=========================================================================================
MODULE: Main Training Orchestrator (main.py)
=========================================================================================
DESCRIPTION:
    Master execution script designed to run deep learning training experiments under 
    strictly controlled conditions for population-level statistical comparisons (N=5).
    
    Key functionalities:
    1. Dynamic Logging System: Redirects standard console output (sys.stdout) to 
       timestamped text files for isolated experiment auditing.
    2. Immutable Base Configuration: Centralizes shared hyperparameters (AdamW, 
       Class-Balanced loss, CutMix regularization) across all evaluated topologies.
    3. Deterministic Task Queue: Sequences independent runs across multiple pseudorandom 
       seeds, neural architectures (AlexNet, VGG-16, ResNet-50, Swin Transformer, ConvNeXt), 
       and domain conditions (With Hair / Without Hair), supporting optional checkpoint 
       resuming.
    4. Fault-Isolated Execution Engine: Wraps individual training runs in structured 
       try/except/finally blocks to prevent mathematical divergences (NaNs) or memory 
       exceptions from interrupting the batch queue.
=========================================================================================
"""

import sys
import os
import datetime
import traceback
import shutil

# Import centralized output directory
from config_paths import REPO_DIR

# Import the master dataset configuration dictionary.
from instantiate_datasets import data_param

# Import the main end-to-end training functions for each neural architecture (ordered chronologically).
from finetuning_training_alexnet import train_alexnet_end_to_end
from finetuning_training_vgg16 import train_vgg16_end_to_end
from finetuning_training_resnet import train_resnet50_end_to_end
from finetuning_training_swin import train_swin_end_to_end
from finetuning_training_convnext import train_convnext_end_to_end

def init_experiment_logger(exp_name, log_dir=None):
    """
    Redirects the standard console output (stdout) to a text file.
    Maintains an isolated and traceable log for each experiment in the batch queue.
    """
    if log_dir is None:
        log_dir = REPO_DIR

    # Ensure the target log directory exists.
    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamp to avoid overwriting previous logs.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"log_{exp_name}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    # Define a wrapper class to bifurcate the output stream.
    class Logger(object):
        def __init__(self, filepath):
            self.terminal = sys.stdout  # Save reference to the original console.
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.log = open(self.filepath, "w", encoding="utf-8")  # Open log file in write mode.

        def write(self, message):
            self.terminal.write(message)  # Print to the visual console.
            self.log.write(message)       # Simultaneously write to the physical file.

        def flush(self):
            # Flush both streams to ensure real-time writing before moving or closing.
            self.terminal.flush()
            if not self.log.closed:
                self.log.flush()

        def relocate(self, new_dir):
            """
            Moves the active log file into the newly created model training directory
            and seamlessly resumes writing in the new location.
            """
            os.makedirs(new_dir, exist_ok=True)
            new_filepath = os.path.join(new_dir, self.filename)
            
            if os.path.abspath(self.filepath) != os.path.abspath(new_filepath):
                self.flush()
                self.log.close()
                shutil.move(self.filepath, new_filepath)
                self.filepath = new_filepath
                self.log = open(self.filepath, "a", encoding="utf-8")  # Reopen in append mode
                print(f"📁 [LOGGER] Log file relocated inside training folder: {self.filepath}")

    # Replace the system's standard output with the bifurcated Logger class.
    sys.stdout = Logger(log_path)
    print(f"\n{'='*60}")
    print(f"[LOGGER] Starting log record: {log_path}")
    print(f"{'='*60}\n")
    return sys.stdout


def restore_logger(logger_obj):
    """
    Closes the current log file stream and restores sys.stdout control back to the console.
    Prevents memory leaks and file locks between experiment iterations.
    """
    if hasattr(logger_obj, 'log'):
        logger_obj.log.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    # Deferred imports to avoid unnecessary loading if the script is imported as a module.
    from device_config import hardware_device
    from saving_functions import save_main_info
    
    # =========================================================
    # 1. BASE PARAMETERS
    # =========================================================
    # Master dictionary of training hyperparameters shared across all models.
    main_param_base = {
        "initial_lr": 1e-5,             # Standard initial learning rate.
        "optim": 'adamw',               # Optimizer with decoupled weight decay.
        "init_weights": 'kaiming',      # Weight initialization strategy.
        "scheduler": 'exponential',     # Learning rate scheduler.
        "debug_mode": False,            # Logical flag for debug/test mode. It runs a short training.
        "loss_strategy": 'cb',          # Loss strategy: Class Balanced.
        "validation": 'train_val',      # Validation scheme to use.
        "final_test": True,             # Execute post-training test evaluation.
        "number_of_workers": 0,         # Threads for DataLoaders (0 = main thread).
        "penalty_factor": 1,            # Multiplier for penalty matrix.
        "use_penalty_matrix": False,    # Flag for cost matrix (disabled in favor of CB).
        "cutmix": True,                 # Active spatial regularization.
        "n_versions_per_epoch": 1,      # Iteration control over static augmented versions.
        "total_versions": 1,            # Total preprocessed versions available.
        "use_attention": False,         # Specific flag for models with attention mapping.
    }

    # =========================================================
    # 2. SEED DEFINITIONS
    # =========================================================
    # Deterministic seed blocks to guarantee statistical reproducibility 
    # across multiple independent training runs (N=5), ordered chronologically by model.

    # AlexNet Seeds
    SEED_A1 = 135790
    SEED_A2 = 246801
    SEED_A3 = 112233
    SEED_A4 = 303030 
    SEED_A5 = 404040

    # VGG-16 Seeds
    SEED_V1 = 123456
    SEED_V2 = 654321
    SEED_V3 = 987654
    SEED_V4 = 101010 
    SEED_V5 = 202020 

    # ResNet-50 Seeds
    SEED_R1 = 407650  
    SEED_R2 = 942741  
    SEED_R3 = 333333  
    SEED_R4 = 112211 
    SEED_R5 = 556655 

    # Swin Transformer Seeds
    SEED_S1 = 214631
    SEED_S2 = 831824  
    SEED_S3 = 999999  
    SEED_S4 = 776655 
    SEED_S5 = 445566 

    # ConvNeXt Seeds
    SEED_C1 = 966788 
    SEED_C2 = 825040 
    SEED_C3 = 666666 
    SEED_C4 = 443322 
    SEED_C5 = 998877 

    # =========================================================
    # 3. TASK QUEUE (Control Panel)
    # =========================================================
    # List of structured tuples. Each tuple defines a unique experiment.
    # Positional format: (Target_Function, Model_Name, Seed, Hair_Removed_Flag, [Optional_Resume_Path])
    experiment_queue = [
        
        # --- 1. ALEXNET ---
        (train_alexnet_end_to_end, "AlexNet", SEED_A1, False), 
        (train_alexnet_end_to_end, "AlexNet", SEED_A3, False), 
        (train_alexnet_end_to_end, "AlexNet", SEED_A3, True),  
        (train_alexnet_end_to_end, "AlexNet", SEED_A4, False),
        (train_alexnet_end_to_end, "AlexNet", SEED_A4, True),
        (train_alexnet_end_to_end, "AlexNet", SEED_A5, False),
        (train_alexnet_end_to_end, "AlexNet", SEED_A5, True),
        (train_alexnet_end_to_end, "AlexNet", SEED_A1, True),  
        (train_alexnet_end_to_end, "AlexNet", SEED_A2, False), 
        (train_alexnet_end_to_end, "AlexNet", SEED_A2, True),  

        # --- 2. VGG-16 ---
        (train_vgg16_end_to_end, "VGG16", SEED_V1, False), 
        (train_vgg16_end_to_end, "VGG16", SEED_V1, True),  
        (train_vgg16_end_to_end, "VGG16", SEED_V2, False), 
        (train_vgg16_end_to_end, "VGG16", SEED_V2, True),  
        (train_vgg16_end_to_end, "VGG16", SEED_V3, False), 
        (train_vgg16_end_to_end, "VGG16", SEED_V3, True),  
        (train_vgg16_end_to_end, "VGG16", SEED_V4, False),
        (train_vgg16_end_to_end, "VGG16", SEED_V4, True),
        (train_vgg16_end_to_end, "VGG16", SEED_V5, False),
        (train_vgg16_end_to_end, "VGG16", SEED_V5, True),

        # --- 3. RESNET-50 ---
        (train_resnet50_end_to_end, "ResNet50", SEED_R1, False), 
        (train_resnet50_end_to_end, "ResNet50", SEED_R1, True),  
        (train_resnet50_end_to_end, "ResNet50", SEED_R5, True),
        (train_resnet50_end_to_end, "ResNet50", SEED_R2, False), 
        (train_resnet50_end_to_end, "ResNet50", SEED_R2, True),  
        (train_resnet50_end_to_end, "ResNet50", SEED_R3, False), 
        (train_resnet50_end_to_end, "ResNet50", SEED_R3, True),  
        (train_resnet50_end_to_end, "ResNet50", SEED_R4, False),
        (train_resnet50_end_to_end, "ResNet50", SEED_R4, True),
        (train_resnet50_end_to_end, "ResNet50", SEED_R5, False),

        # --- 4. SWIN TRANSFORMER ---
        (train_swin_end_to_end, "Swin", SEED_S1, False), 
        (train_swin_end_to_end, "Swin", SEED_S1, True),  
        (train_swin_end_to_end, "Swin", SEED_S2, False), 
        (train_swin_end_to_end, "Swin", SEED_S2, True),  
        (train_swin_end_to_end, "Swin", SEED_S3, False), 
        (train_swin_end_to_end, "Swin", SEED_S3, True),  
        (train_swin_end_to_end, "Swin", SEED_S4, False),
        (train_swin_end_to_end, "Swin", SEED_S4, True),
        (train_swin_end_to_end, "Swin", SEED_S5, False),
        (train_swin_end_to_end, "Swin", SEED_S5, True),
        
        # --- 5. CONVNEXT ---
        (train_convnext_end_to_end, "ConvNeXt", SEED_C1, False), 
        (train_convnext_end_to_end, "ConvNeXt", SEED_C1, True),  
        (train_convnext_end_to_end, "ConvNeXt", SEED_C2, False), 
        (train_convnext_end_to_end, "ConvNeXt", SEED_C2, True),  
        (train_convnext_end_to_end, "ConvNeXt", SEED_C3, False), 
        (train_convnext_end_to_end, "ConvNeXt", SEED_C3, True),  
        (train_convnext_end_to_end, "ConvNeXt", SEED_C4, False),
        (train_convnext_end_to_end, "ConvNeXt", SEED_C4, True),
        (train_convnext_end_to_end, "ConvNeXt", SEED_C5, False),
        (train_convnext_end_to_end, "ConvNeXt", SEED_C5, True),

        # --- EXAMPLE: RESUMING AN INTERRUPTED RUN FROM A CHECKPOINT ---
        # If a training run is interrupted (e.g., power outage or hardware timeout), 
        # pass a 5th tuple element with the path to the saved .pth checkpoint weights 
        # to resume training from that state instead of starting from scratch:
        # (train_convnext_end_to_end, "ConvNeXt", SEED_C4, False, 
        #  r"./trained_models/2026-05-11-22-07-57_ConvNext_native_fb/model_weights_f2_score-2026-05-11-22-07-[294].pth"),
    ]

    # =========================================================
    # 4. EXECUTION ENGINE (Iterative Pipeline)
    # =========================================================
    total_tasks = len(experiment_queue)
    print(f"🚀 BATCH MANAGER: {total_tasks} pending experiments.")

    # Sequentially iterate over the defined experiment queue.
    for i, task in enumerate(experiment_queue):
        # Positional tuple unpacking.
        train_func = task[0]
        net_name = task[1]
        seed = task[2]
        hair_removed = task[3]
        # Conditional checkpoint extraction (if a 5th element is provided in the tuple).
        resume_path = task[4] if len(task) > 4 else None

        # Format nomenclature for logging and data persistence.
        condition_text = "WithoutHair" if hair_removed else "WithHair"
        experiment_name = f"{net_name}_Seed{seed}_{condition_text}"
        
        # Start console output isolation for this specific experiment.
        current_logger = init_experiment_logger(experiment_name)
        print(f"\n[{i+1}/{total_tasks}] STARTING EXPERIMENT: {experiment_name}")
        
        # Shallow copy of the base dictionary to inject parameters without contaminating subsequent runs.
        current_main_param = main_param_base.copy()
        current_main_param["seed"] = seed
        current_main_param["resume_path"] = resume_path
        
        # Inject domain condition into the globally imported dataset dictionary.
        data_param["hair_removed"] = hair_removed 
        
        # Assign spatial resolution based on network topology requirements.
        if net_name == "ConvNeXt":
            data_param["size"] = 256
        else:
            data_param["size"] = 224
        
        # Pre-training configuration persistence.
        save_main_info(current_main_param, data_param)
        
        try:
            # Execute the architecture training, passing the injected parameters.
            result = train_func(current_main_param)
            
            # Evaluate the exit status returned by the training loop.
            if result == 'finish':
                print(f"✅ EXPERIMENT COMPLETED: {experiment_name}")
            elif result == 'nan':
                print(f"❌ FAILED DUE TO NaN: {experiment_name}")
                
        except Exception as e:
            # Handle catastrophic exceptions to prevent a single failed run from halting the entire queue.
            print(f"💥 CRITICAL ERROR IN {experiment_name}:")
            print(traceback.format_exc())
            
        finally:
            # Guaranteed execution block: always closes the log regardless of success or failure.
            restore_logger(current_logger)

    print("\n🎉 ALL SCHEDULED EXPERIMENTS HAVE FINISHED.")