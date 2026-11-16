"""
=========================================================================================
MODULE: Multi-Metric Early Stopping and Checkpoint Pruning Manager (early_stopping.py)
=========================================================================================
DESCRIPTION:
    Manages multi-metric early stopping criteria across progressive unfreezing phases 
    and executes intelligent checkpoint pruning and consolidation upon training completion.
    
    Methodological Specifications (see Manuscript Section 'Training Pipeline and Early 
    Stopping Protocol'):
    1. Multi-Objective Telemetry: Simultaneously monitors validation loss (minimization) 
       and global F1, global F2, and Melanoma-specific F2 scores (maximization).
    2. Phase Isolation: Provides explicit phase resetting (`reset()`) to ensure patience 
       counters do not carry over across unfreezing transitions (Phases 1, 2, and 3).
    3. Checkpoint Pruning & Merging (`prune_models`): Automatically identifies checkpoints
       saved at the same epoch across different metrics, unifies them into consolidated 
       multi-metric files, and purges checkpoints that fail to meet survival performance 
       thresholds.
=========================================================================================
"""

import os
import time


class EarlyStoppingManager:
    """
    Tracks validation metrics, manages patience and early stopping flags, saves optimal 
    model state dictionaries, and prunes substandard intermediate checkpoints.
    """
    def __init__(
        self,
        patience=10,
        min_delta=0.0001,
        save_func=None,
        start_time=None,
        output_dir=None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_func = save_func
        self.start_time = start_time
        self.output_dir = output_dir
        
        self.counter = 0
        self.early_stop = False
        
        # Survival thresholds and optimization directions (-1 = minimize, 1 = maximize)
        self.best_metrics = {
            'val_loss': {'value': float('inf'), 'direction': -1, 'threshold': 0.65},
            'f1_score': {'value': 0.0,         'direction': 1,  'threshold': 0.75},
            'f2_score': {'value': 0.0,         'direction': 1,  'threshold': 0.75},
            'mel_f2':   {'value': 0.0,         'direction': 1,  'threshold': 0.55} 
        }

    def reset(self):
        """Resets patience counters and stop flags prior to a new unfreezing phase."""
        self.counter = 0
        self.early_stop = False
        print("🔄 Early stopping patience reset for the new phase.")

    def update(self, current_metrics, model, epoch):
        """
        Evaluates current epoch metrics against best recorded values, triggers checkpoint 
        persistence on improvement, and updates the patience counter.
        """
        improved_any = False
        for metric_name, current_val in current_metrics.items():
            if metric_name not in self.best_metrics or current_val is None:
                continue
                
            best_info = self.best_metrics[metric_name]
            direction = best_info['direction']
            
            # Check for improvement beyond min_delta
            if (direction == 1 and current_val > best_info['value'] + self.min_delta) or \
               (direction == -1 and current_val < best_info['value'] - self.min_delta):
                
                best_info['value'] = current_val
                improved_any = True
                print(f"⭐ [Improvement {metric_name}]: {current_val:.4f}")
                
                if self.save_func:
                    self.save_func(
                        model, 
                        self.start_time, 
                        epoch, 
                        target_dir=self.output_dir,
                        delete_previous_weights=True, 
                        base_name=f"model_weights_{metric_name}"
                    )

        if improved_any:
            self.counter = 0
        else:
            self.counter += 1
            print(f"⏳ Patience Counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

    def prune_models(self):
        """
        Unifies multi-metric checkpoints saved during the same epoch and purges 
        checkpoints that fail to meet minimum performance survival thresholds.
        """
        if not self.output_dir:
            return
            
        print("\n🧹 Initiating checkpoint cleanup and consolidation...")
        
        # 1. Unify multi-metric checkpoints saved at the same epoch
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.pth') and "final" not in f]
        epochs = {} 

        for f in files:
            try:
                num_epoch = f.split('[')[-1].split(']')[0]
                if num_epoch not in epochs:
                    epochs[num_epoch] = []
                epochs[num_epoch].append(f)
            except Exception:
                continue

        for num, model_files in epochs.items():
            if len(model_files) > 1:
                metrics_found = []
                for m in model_files:
                    for real_metric in self.best_metrics.keys():
                        if real_metric in m:
                            metrics_found.append(real_metric)
                
                tag_multi = "_".join(list(set(metrics_found)))
                base_file = model_files[0]
                
                ext = base_file.split('-')[-1]
                new_name = f"model_weights_MULTI_{tag_multi}-{base_file.split('model_weights_')[-1].split('-')[0]}-{ext}"
                
                os.rename(os.path.join(self.output_dir, base_file), os.path.join(self.output_dir, new_name))
                print(f"🧬 Epoch {num}: Consolidated {len(model_files)} checkpoints -> {new_name}")
                
                for i in range(1, len(model_files)):
                    os.remove(os.path.join(self.output_dir, model_files[i]))

        # 2. Prune checkpoints failing to meet survival performance thresholds
        remaining_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pth') and "final" not in f]
        
        for f in remaining_files:
            file_metrics = [m for m in self.best_metrics.keys() if m in f]
            
            survives = False
            reason = ""
            for m in file_metrics:
                info = self.best_metrics[m]
                if (info['direction'] == 1 and info['value'] >= info['threshold']) or \
                   (info['direction'] == -1 and info['value'] <= info['threshold']):
                    survives = True
                    reason = f"{m} ({info['value']:.4f})"
                    break
            
            if not survives:
                print(f"🗑️ Discarding {f}: Did not reach survival threshold in any metric.")
                os.remove(os.path.join(self.output_dir, f))
            else:
                print(f"✅ Retained {f}: Survival validated by {reason}")