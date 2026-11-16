"""
=========================================================================================
MODULE: Persistence, Logging, and Visualization Utilities (saving_functions.py)
=========================================================================================
DESCRIPTION:
    Provides standardized I/O routines for saving model checkpoints (.pth), training 
    histories (.h5), confusion matrices (.npz, .pdf, .png), learning curves (.csv, .png), 
    hyperparameter audit logs (.txt), and computational throughput reports.
    
    Key functionalities:
    1. Experiment Directory Management: Generates timestamped, architecture-tagged 
       directories for isolated artifact storage.
    2. Checkpoint Persistence: Saves and prunes PyTorch state dictionaries formatted 
       with metric tags and epoch brackets (e.g., model_weights_f2_score-YYYY-MM-DD-[E].pth).
    3. Incremental HDF5 Telemetry: Records per-epoch losses, learning rates, global 
       metrics (Precision, Recall, F1, F2), and class-specific F2 trajectories.
    4. Automated Plotting & Smoothing: Generates raw and 5-epoch rolling-average curves 
       alongside normalized confusion matrix heatmaps across TRAIN, VAL, and TEST sets.
=========================================================================================
"""

import os
import time
import traceback
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from instantiate_datasets import label_mapping
from config_paths import REPO_DIR

# Base directory configuration
base_dir = REPO_DIR
output_dir = os.path.join(REPO_DIR, "trained_models")
os.makedirs(output_dir, exist_ok=True)


def create_training_directory(start_time, model, base_dir=base_dir):
    """
    Creates a timestamped output directory named after the active neural architecture 
    and updates the module-level output_dir reference for subsequent saving calls.
    """
    global output_dir
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(start_time))
    model_module = model.__class__.__module__
    model_name = os.path.splitext(os.path.basename(model_module.replace(".", "/")))[0]
    folder_name = f"{timestamp}_{model_name}"
    
    # Store inside the 'trained_models' subdirectory
    trained_models_root = os.path.join(base_dir, "trained_models")
    folder_dir = os.path.join(trained_models_root, folder_name)
    
    os.makedirs(folder_dir, exist_ok=True)
    output_dir = folder_dir
    return folder_dir


def calculate_xticks(num_epochs):
    """Determines optimal X-axis tick spacing based on total training epochs."""
    if num_epochs <= 20:
        return 1
    elif num_epochs <= 50:
        return 2
    elif num_epochs <= 100:
        return 5
    elif num_epochs <= 200:
        return 10
    elif num_epochs <= 300:
        return 25
    else:
        return 50


def plot_series(y_series, labels, ylabel, ylim, filename, ticks):
    """Generates and saves a standardized 2D line plot for training curves."""
    plt.figure(figsize=(10, 7.5))
    epochs = np.arange(1, len(y_series[0]) + 1)
    for y, label in zip(y_series, labels):
        plt.plot(epochs, y, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    if len(ticks) > 0:
        plt.xticks(ticks=ticks)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def save_model(model, start_time, epoch, target_dir=None, delete_previous_weights=False, base_name='model_weights', phase_info=None):
    """
    Saves the model state_dict to disk with robust error handling.
    """
    target_dir = target_dir or output_dir
    os.makedirs(target_dir, exist_ok=True)
    formatted_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(start_time))
    
    full_base_name = base_name
    if phase_info:
        full_base_name = f"{full_base_name}_{phase_info}"
    
    # Remove previous checkpoint files sharing the same timestamp and metric prefix
    for filename in os.listdir(target_dir):
        if formatted_start_time in filename and full_base_name in filename:
            file_path = os.path.join(target_dir, filename)
            try:
                os.remove(file_path)
            except OSError:
                pass
                    
    model_filename = os.path.join(target_dir, f'{full_base_name}-{formatted_start_time}-[{epoch+1}].pth')
    
    try:
        torch.save(model.state_dict(), model_filename)
    except Exception as e:
        print(f"⚠️ Warning: Failed to save checkpoint at epoch {epoch+1}: {e}")


def save_initial_info(start_time, optimizer, scheduler, dropout_rate, penalty_matrix, model, use_penalty_matrix, seed, model_fn=None, target_dir=None):
    """
    Writes an initial audit file detailing the model module, pseudorandom seed, 
    optimizer/scheduler hyperparameters, regularization rates, and activation layers.
    """
    target_dir = target_dir or output_dir
    formatted_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(start_time))
    
    log_filename = os.path.join(target_dir, f'initial_info-{formatted_start_time}.txt')
    with open(log_filename, 'w', encoding='utf-8') as f:
        # 1. Model Module
        if model_fn is not None:
            model_module_name = model_fn.__module__
        else:
            model_module_name = type(model).__module__
        f.write(f"Model Module: {model_module_name}\n\n")
        
        # 2. Random Seed
        f.write(f"Random Seed: {seed}\n\n")
        
        # 3. Optimizer Hyperparameters
        f.write("Optimizer Hyperparameters:\n")
        initial_lr = optimizer.param_groups[0]['lr']
        f.write(f"Initial Learning Rate: {initial_lr}\n")
        f.write(f"Momentum: {optimizer.param_groups[0].get('momentum', 'N/A')}\n")
        f.write(f"Weight Decay: {optimizer.param_groups[0]['weight_decay']}\n\n")
        
        # 4. Scheduler Hyperparameters
        f.write("Scheduler Hyperparameters:\n")
        f.write(f"Gamma: {getattr(scheduler, 'gamma', None)}\n\n")
        
        # 5. Dropout Rate
        f.write(f"Dropout rate = {dropout_rate}\n\n")
        
        # 6. Activation Functions
        f.write("Activation Functions:\n")
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                f.write(f"Layer Name: {name}\n")
                f.write(f"Type: {layer.__class__.__name__}\n\n")


def save_main_info(main_param, data_param, target_dir=None):
    """
    Persists the master training and dataset configuration dictionaries prior to 
    experiment execution.
    """
    target_dir = target_dir or output_dir
    os.makedirs(target_dir, exist_ok=True)
    formatted_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    log_filename = os.path.join(target_dir, f'param_info-{formatted_start_time}.txt')

    with open(log_filename, 'w', encoding='utf-8') as f:
        # 1. Ordered Training Parameters
        f.write("Training Parameters:\n")
        ordered_keys = [
            ('initial_lr', float),
            ('optim', str),
            ('init_weights', str),
            ('scheduler', str),
            ('debug_mode', bool),
            ('loss_strategy', str),
            ('validation', str),
            ('final_test', bool),
            ('number_of_workers', int),
            ('cutmix', bool),
            ('n_versions_per_epoch', int),
            ('total_versions', int),
        ]
        write_formatted_lines(f, main_param, ordered_keys)

        # 2. Remaining Unlisted Parameters (e.g., seed, resume_path)
        other_keys = set(main_param.keys()) - set(k for k, _ in ordered_keys)
        if other_keys:
            f.write("\n  Additional Training Parameters:\n")
            for k in sorted(other_keys):
                f.write(f"    {k}: {main_param[k]}\n")

        f.write("\n")

        # 3. Dataset Parameters
        f.write("Dataset Parameters:\n")
        write_formatted_lines(f, data_param, [
            ('ROI', bool),
            ('flipHV_rot', bool),
            ('affine', bool),
            ('jitter', bool),
            ('normalize', bool),
            ('masked', bool),
            ('masked_value', int),
            ('shift', int),
            ('max_cropped', bool),
            ('gaussian', bool),
            ('crop_type', str),
            ('channels', int),
            ('size', int),
            ('balanced', bool),
            ('multiplier', int),
            ('hair_removed', bool)
        ])


def write_formatted_lines(file, params, params_order):
    """
    Auxiliary function to format and write dictionary key-value pairs into a log file.
    """
    fallback_keys = {
        'debug_mode': 'prueba',
        'affine': 'afin',
        'shift': 'desplazamiento',
        'hair_removed': 'afeitado'
    }
    
    for i, (key, dtype) in enumerate(params_order):
        actual_key = key if key in params else fallback_keys.get(key, key)
        if actual_key not in params:
            continue
        value = params[actual_key]
        
        if dtype == bool:
            formatted_value = str(value).lower() if isinstance(value, bool) else str(value)
        elif dtype == str:
            formatted_value = f"'{value}'"
        else:
            formatted_value = str(value)
        
        line_end = ",\n" if i < len(params_order) - 1 else "\n"
        file.write(f'    "{key}": {formatted_value}{line_end}')


def save_data(start_time, elapsed_time, epoch, train_losses=None, val_losses=None, conf_matrix_list=None, history=None, target_dir=None):
    """
    Saves per-epoch confusion matrices (.npz) and appends training/validation telemetry 
    to an HDF5 (.h5) file.
    """
    target_dir = target_dir or output_dir
    formatted_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(start_time))
    
    # Clean up previous intermediate files from the same run (excluding checkpoints, logs, and .h5)
    for filename in os.listdir(target_dir):
        if formatted_start_time in filename:
            if not any(tag in filename for tag in ["model_weights", "pesos_modelo", "initial_info", "info_inicio", "param_info", "info_param"]):
                if not filename.endswith('.h5'):
                    try:
                        os.remove(os.path.join(target_dir, filename))
                    except OSError:
                        pass
    
    partition_tag = 'training' if (val_losses is None or len(val_losses) == 0) else 'validation'

    if conf_matrix_list is not None and len(conf_matrix_list) > 0:
        np.savez(os.path.join(target_dir, f'confusion_matrices_{partition_tag}-{formatted_start_time}.npz'), *conf_matrix_list)
        
    hdf5_filename = os.path.join(target_dir, f'Model_{partition_tag}_data-{formatted_start_time}.h5')
    
    with h5py.File(hdf5_filename, 'a') as f:
        # Prepare Class F2 Scores array
        class_f2_array = np.array([], dtype='float32')
        if history and 'class_f2' in history and len(history['class_f2']) > 0:
            try:
                class_f2_array = np.array([list(cp.values()) for cp in history['class_f2']], dtype='float32')
            except Exception:
                pass

        num_classes = len(label_mapping)
        if class_f2_array.size > 0 and len(class_f2_array.shape) > 1:
            num_classes = class_f2_array.shape[1]

        # Append or initialize datasets inside the HDF5 container
        if 'Epochs' in f:
            if len(history['epochs']) > 0:
                f['Epochs'].resize((f['Epochs'].shape[0] + 1), axis=0)
                f['Epochs'][-1] = history['epochs'][-1]
                
                f['Train Loss'].resize((f['Train Loss'].shape[0] + 1), axis=0)
                f['Train Loss'][-1] = history['train_loss'][-1]
                
                if partition_tag != 'training':
                    f['Validation Loss'].resize((f['Validation Loss'].shape[0] + 1), axis=0)
                    f['Validation Loss'][-1] = history['val_loss'][-1]
                
                ds_name = 'Class F2 Scores'
                if ds_name not in f:
                    if class_f2_array.size > 0:
                        f.create_dataset(ds_name, data=class_f2_array, dtype='float32', maxshape=(None, num_classes))
                else:
                    if class_f2_array.size > 0:
                        f[ds_name].resize((f[ds_name].shape[0] + 1), axis=0)
                        f[ds_name][-1] = class_f2_array[-1]

                metrics_map = {
                    'Precision': 'precision', 
                    'Recall': 'recall', 
                    'F1 Score': 'f1_score', 
                    'Learning Rate': 'learning_rates'
                }
                for ds_key, hist_key in metrics_map.items():
                    val = history.get(hist_key, [])[-1] if len(history.get(hist_key, [])) > 0 else 0
                    if ds_key not in f:
                        f.create_dataset(ds_key, data=[val], dtype='float32', maxshape=(None,))
                    else:
                        f[ds_key].resize((f[ds_key].shape[0] + 1), axis=0)
                        f[ds_key][-1] = val
                
                if 'F2 Score' not in f:
                    f.create_dataset('F2 Score', data=np.array(history.get('f2_score', [0])), dtype='float32', maxshape=(None,))
                else:
                    f['F2 Score'].resize((f['F2 Score'].shape[0] + 1), axis=0)
                    f['F2 Score'][-1] = history.get('f2_score', [0])[-1]
        else: 
            # Initial HDF5 dataset creation
            f.create_dataset('Epochs', data=np.array(history['epochs']), dtype='int32', maxshape=(None,))
            f.create_dataset('Train Loss', data=np.array(history['train_loss']), dtype='float32', maxshape=(None,))
            if partition_tag != 'training':
                f.create_dataset('Validation Loss', data=np.array(history['val_loss']), dtype='float32', maxshape=(None,))
            
            ds_name = 'Class F2 Scores'
            if class_f2_array.size == 0:
                f.create_dataset(ds_name, shape=(0, num_classes), dtype='float32', maxshape=(None, num_classes))
            else:
                f.create_dataset(ds_name, data=class_f2_array, dtype='float32', maxshape=(None, num_classes))
            try:
                f[ds_name].attrs['class_labels'] = np.array(list(label_mapping.keys()), dtype=np.bytes_)
            except Exception:
                pass

            f.create_dataset('Precision', data=np.array(history['precision']), dtype='float32', maxshape=(None,))
            f.create_dataset('Recall', data=np.array(history['recall']), dtype='float32', maxshape=(None,))
            f.create_dataset('F1 Score', data=np.array(history['f1_score']), dtype='float32', maxshape=(None,))
            f.create_dataset('F2 Score', data=np.array(history.get('f2_score', [])), dtype='float32', maxshape=(None,))
            f.create_dataset('Learning Rate', data=np.array(history['learning_rates']), dtype='float32', maxshape=(None,))


def process_plots(start_time, target_dir=None):
    """
    Reads the experiment HDF5 telemetry and generates CSV tables, learning curve plots 
    (Loss, F1/F2, raw and 5-epoch smoothed per-class F2), and confusion matrix PDFs.
    """
    target_dir = target_dir or output_dir
    if isinstance(start_time, (int, float)):
        formatted_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(start_time))
    else:
        formatted_start_time = str(start_time)

    # Locate the corresponding HDF5 file (supports both English and legacy Spanish names)
    hdf5_filename = None
    partition_tag = 'training'
    for filename in os.listdir(target_dir):
        if formatted_start_time in filename and ('Model_' in filename or 'Datos_' in filename) and filename.endswith('.h5'):
            hdf5_filename = os.path.join(target_dir, filename)
            if 'validation' in filename or 'de_validación' in filename:
                partition_tag = 'validation'
            else:
                partition_tag = 'training'
            break
    
    if not hdf5_filename:
        return

    try:
        with h5py.File(hdf5_filename, 'r') as f:
            if 'Epochs' not in f:
                return

            epochs_raw = f['Epochs'][:]
            train_loss_raw = f['Train Loss'][:]
            val_loss_raw = f['Validation Loss'][:] if 'Validation Loss' in f else []
            
            class_scores_raw = []
            if 'Class F2 Scores' in f:
                class_scores_raw = f['Class F2 Scores'][:] * 100
            elif 'Class Precisions' in f:
                class_scores_raw = f['Class Precisions'][:] * 100
            
            class_names = []
            if 'Class F2 Scores' in f and 'class_labels' in f['Class F2 Scores'].attrs:
                class_names = f['Class F2 Scores'].attrs['class_labels'].astype(str)
            elif 'Class Precisions' in f and 'class_labels' in f['Class Precisions'].attrs:
                class_names = f['Class Precisions'].attrs['class_labels'].astype(str)
            if len(class_names) == 0 and len(class_scores_raw) > 0:
                class_names = [f'C{i}' for i in range(class_scores_raw.shape[1])]

            f1_raw = f['F1 Score'][:]
            f2_raw = f['F2 Score'][:] if 'F2 Score' in f else np.zeros_like(f1_raw)

        # Pad arrays to uniform maximum length
        max_len = max(len(epochs_raw), len(train_loss_raw))
        
        def pad(arr, length):
            if len(arr) >= length:
                return arr[:length]
            return np.concatenate([arr, np.zeros(length - len(arr))])

        epochs = np.arange(1, max_len + 1)
        train_losses = pad(train_loss_raw, max_len)
        val_losses = pad(val_loss_raw, max_len)
        f1_score_value = pad(f1_raw, max_len)
        f2_score_value = pad(f2_raw, max_len)
        
        if len(class_scores_raw) > 0:
            rows, cols = class_scores_raw.shape
            if rows < max_len:
                class_scores = np.vstack([class_scores_raw, np.zeros((max_len - rows, cols))])
            else:
                class_scores = class_scores_raw[:max_len]
        else:
            class_scores = np.array([])

        # 1. Per-Class F2-Score (CSV, Raw PNG, and 5-Epoch Moving Average PNG)
        if len(class_scores) > 0:
            df = pd.DataFrame(class_scores, columns=class_names)
            df.index = epochs
            df.index.name = 'Epochs'
            df.to_csv(os.path.join(target_dir, f'f2_score_{partition_tag}_per_class-{formatted_start_time}.csv'))
            
            series = [class_scores[:, i] for i in range(class_scores.shape[1])]
            plot_series(
                series, class_names, 'F2-Score (%)', [0, 100], 
                os.path.join(target_dir, f'f2_score_{partition_tag}_per_class-{formatted_start_time}.png'), []
            )

            df_smooth = df.rolling(window=5, min_periods=1).mean()
            series_smooth = [df_smooth[col].values for col in class_names]
            plot_series(
                series_smooth, class_names, 'F2-Score (%) [Smooth 5ep]', [0, 100], 
                os.path.join(target_dir, f'f2_score_{partition_tag}_per_class_smooth-{formatted_start_time}.png'), []
            )

        # 2. General F1 & F2 Scores
        df_score = pd.DataFrame({'F1-Score': f1_score_value, 'F2-Score': f2_score_value})
        df_score.index = epochs
        df_score.index.name = 'Epochs'
        df_score.to_csv(os.path.join(target_dir, f'score_{partition_tag}-{formatted_start_time}.csv'))
        
        plot_series(
            [f1_score_value, f2_score_value], ['F1', 'F2'], 'Score', [0, 1], 
            os.path.join(target_dir, f'score_{partition_tag}_full-{formatted_start_time}.png'), []
        )

        # 3. Training and Validation Losses
        ratio = [v / t if t != 0 else 0 for v, t in zip(train_losses, val_losses)]
        df_loss = pd.DataFrame({'Train Loss': train_losses, 'Val Loss': val_losses, 'Ratio': ratio})
        df_loss.index = epochs
        df_loss.index.name = 'Epochs'
        df_loss.to_csv(os.path.join(target_dir, f'model_losses-{formatted_start_time}.csv'))

        max_loss_val = max(max(train_losses), max(val_losses) if len(val_loss_raw) > 0 else 0)
        y_max = max(1.5, max_loss_val * 1.1) 

        if len(val_loss_raw) == 0:
            plot_series(
                [train_losses], ['Train Loss'], 'Loss', [0, y_max], 
                os.path.join(target_dir, f'training_loss-{formatted_start_time}.png'), []
            )
        else:
            plot_series(
                [train_losses, val_losses], ['Train', 'Val'], 'Loss', [0, y_max], 
                os.path.join(target_dir, f'losses-{formatted_start_time}.png'), []
            )
            
        # 4. Confusion Matrices (Multi-page PDF and Final Epoch PNG)
        npz_filename = os.path.join(target_dir, f'confusion_matrices_{partition_tag}-{formatted_start_time}.npz')
        if os.path.exists(npz_filename):
            with np.load(npz_filename) as data:
                files = sorted(data.files, key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
                if files:
                    conf_matrix_list = [data[key] for key in files]
                    with PdfPages(os.path.join(target_dir, f'confusion_matrices_{partition_tag}-{formatted_start_time}.pdf')) as pdf:
                        for idx, cm in enumerate(conf_matrix_list):
                            _plot_matrix_inline(cm, idx + 1, pdf)
                    
                    _plot_matrix_inline(
                        conf_matrix_list[-1], len(conf_matrix_list), None, 
                        os.path.join(target_dir, f'last_confusion_matrix_{partition_tag}-{formatted_start_time}.png')
                    )

    except Exception as e:
        print(f"❌ Error processing plots: {e}")
        traceback.print_exc()


def _plot_matrix_inline(cm, epoch, pdf=None, filename=None):
    """Renders a normalized confusion matrix heatmap for PDF or PNG export."""
    plt.figure(figsize=(10, 7.5))
    cm_p = cm * 100 if cm.max() <= 1.05 else cm
    plt.imshow(cm_p, cmap='viridis', vmin=0, vmax=100)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.colorbar(label='%')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    try:
        labels = list(label_mapping.keys())
        plt.xticks(np.arange(len(labels)), labels, rotation=45)
        plt.yticks(np.arange(len(labels)), labels)
    except Exception:
        pass
    for i in range(len(cm_p)):
        for j in range(len(cm_p)):
            v = cm_p[i, j]
            if not np.isnan(v):
                plt.text(j, i, f'{v:.1f}%', ha='center', va='center', color='white' if v < 50 else 'black')
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    if filename:
        plt.savefig(filename)
    plt.close()


def save_test_results(formatted_start_time, history, conf_matrix, train_val_found, mode, output_dir_path):
    """
    Saves post-training evaluation metrics (.csv) and confusion matrices (.npz, .png) 
    for a specific partition (TRAIN, VAL, or TEST).
    """
    os.makedirs(output_dir_path, exist_ok=True)

    def get_val(key, default=0.0):
        val = history.get(key, default)
        return val[0] if isinstance(val, list) else val

    test_loss = get_val('test_loss')
    precision = get_val('precision')
    recall = get_val('recall')
    f1_score_value = get_val('f1_score')
    f2_score_value = get_val('f2_score')
    mel_f2_value = get_val('mel_f2')
    class_precisions = get_val('class_precision', {})

    # 1. Build Main Evaluation DataFrame
    data = {
        'Test Loss': test_loss,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score_value,
        'F2-Score': f2_score_value,
        'Melanoma F2': mel_f2_value 
    }

    if train_val_found == 'train_val':
        train_l = get_val('train_loss')
        val_l = get_val('val_loss')
        data['Train Loss'] = train_l
        data['Val Loss'] = val_l
        data['Train/Val'] = train_l / val_l if val_l != 0 else 0
    elif train_val_found == 'train':
        data['Train Loss'] = get_val('train_loss')

    df = pd.DataFrame([data])
    df.to_csv(os.path.join(output_dir_path, f'score_{mode}.csv'), index=False)

    # 2. Save Per-Class Scores
    if class_precisions:
        df_class = pd.DataFrame([class_precisions])
        df_class.to_csv(os.path.join(output_dir_path, f'class_scores_{mode}.csv'), index=False)

    # 3. Save Confusion Matrix (.npz and .png)
    np.savez(os.path.join(output_dir_path, f'confusion_matrix_{mode}.npz'), matrix=conf_matrix)

    plt.figure(figsize=(12, 9))
    matrix_to_plot = np.array(conf_matrix)
    if matrix_to_plot.max() <= 1.05:
        matrix_to_plot = matrix_to_plot * 100
        
    plt.imshow(matrix_to_plot, cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(label='Percentage (%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix {mode.upper()} (%)')
    
    try:
        labels = list(label_mapping.keys())
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
        plt.yticks(ticks=range(len(labels)), labels=labels)
    except Exception:
        pass

    for i in range(len(matrix_to_plot)):
        for j in range(len(matrix_to_plot)):
            value = matrix_to_plot[i, j]
            if not np.isnan(value):
                color = 'white' if value < 50 else 'black'
                plt.text(j, i, f'{value:.1f}%', ha='center', va='center', color=color)
            else:
                plt.text(j, i, 'N/A', ha='center', va='center', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_path, f'confusion_matrix_{mode}.png'))
    plt.close()

    # 4. Save Compact Summary CSV
    df_simple = pd.DataFrame({
        'Loss': [test_loss], 'Precision': [precision], 
        'Recall': [recall], 'F1': [f1_score_value], 'F2': [f2_score_value], 'Mel_F2': [mel_f2_value]
    }).round(4)
    df_simple.to_csv(os.path.join(output_dir_path, f'final_summary_{mode}.csv'), index=False)

    print(f"📄 Evaluation results saved in: {output_dir_path}")


def log_speed_txt(start_time, epoch, elapsed_time, total_processed_images, target_dir=None, custom_dir=None):
    """
    Generates a computational throughput report detailing total elapsed time and 
    average processing speed (images/second).
    """
    if isinstance(epoch, float) and isinstance(elapsed_time, int):
        epoch, elapsed_time = elapsed_time, epoch

    save_path = custom_dir or target_dir or output_dir
    formatted_start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(start_time))
    
    speed = total_processed_images / elapsed_time if elapsed_time > 0 else 0
    
    filename = f'speed_report-{formatted_start_time}-[{epoch+1}].txt'
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("--- PROCESSING SPEED REPORT ---\n")
        f.write(f"Start Timestamp: {formatted_start_time}\n")
        f.write(f"Completed Epochs: {epoch + 1}\n")
        f.write(f"Total Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Total Processed Images: {total_processed_images}\n")
        f.write(f"Average Throughput: {speed:.2f} images/second\n")
        
    print(f"📄 Speed report saved: {filename}")