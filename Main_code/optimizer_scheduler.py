"""
=========================================================================================
MODULE: Optimizer and Learning Rate Scheduler Factory (optimizer_scheduler.py)
=========================================================================================
DESCRIPTION:
    Provides factory routines to instantiate PyTorch optimizers and learning rate 
    schedulers for each progressive unfreezing phase.
    
    Methodological Specifications (see Manuscript Section 'Common Training Parameters' 
    & Table 3):
    1. Baseline Optimizer (AdamW): Configured with decoupled weight decay (0.05) and 
       numerical stability epsilon (1e-8) across all evaluated neural topologies.
    2. Phase Re-instantiation: Optimizers and schedulers are re-instantiated at the 
       beginning of each progressive unfreezing phase (Phases 1, 2, and 3) using the 
       phase-specific initial learning rates defined in Table 3.
    3. Multi-Schedule Support: Supports exponential decay (gamma=0.99), cosine annealing, 
       warm restarts, cyclic learning rates, OneCycleLR, and ReduceLROnPlateau, returning 
       the corresponding stepping mode ('epoch', 'batch', 'plateau', or 'none').
=========================================================================================
"""

import torch
from torch.optim.lr_scheduler import OneCycleLR


class AdaptiveOneCycleLR:
    """
    Wrapper around PyTorch's OneCycleLR that monitors the generalization gap 
    (val_loss - train_loss) and applies a multiplicative learning rate reduction (0.7x) 
    if the gap exceeds a threshold for a consecutive number of patience epochs.
    """
    def __init__(
        self, 
        optimizer, 
        max_lr, 
        epochs, 
        steps_per_epoch, 
        gap_threshold=0.3, 
        patience=3
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.gap_threshold = gap_threshold
        self.patience = patience
        
        # Counter for consecutive epochs exhibiting a large generalization gap
        self.high_gap_count = 0
        
        # Base OneCycleLR instance
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=10000,
            anneal_strategy='cos'
        )
    
    def step(self, train_loss=None, val_loss=None):
        """
        Advances the scheduler by one step and checks for overfitting gap penalties.
        """
        if train_loss is not None and val_loss is not None:
            gap = val_loss - train_loss
            
            if gap > self.gap_threshold:
                self.high_gap_count += 1
                
                if self.high_gap_count >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.7  # 30% LR reduction
                    self.high_gap_count = 0
                    print(f"🔻 Learning rate reduced due to large generalization gap: {gap:.3f}")
            else:
                self.high_gap_count = 0
        
        self.scheduler.step()
    
    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def configure_optimizer_and_scheduler(
    model, 
    optim_name="adam", 
    lr=1e-3, 
    scheduler_name="exponential", 
    epochs=None, 
    steps_per_epoch=None
):
    """
    Instantiates the requested optimizer and learning rate scheduler for the active 
    training phase.

    Args:
        model (nn.Module): Target neural architecture.
        optim_name (str): Optimizer identifier ('adamw', 'adam', 'sgd', 'rmsprop').
        lr (float): Initial learning rate for the current phase.
        scheduler_name (str): Scheduler identifier ('exponential', 'cosine', 'onecycle', 
            'cyclic', 'cosine_restart', 'plateau', or 'none').
        epochs (int, optional): Maximum epochs in the current phase.
        steps_per_epoch (int, optional): Number of batches per epoch (required for OneCycleLR).

    Returns:
        tuple: (optimizer, scheduler, step_type)
            where step_type is one of 'batch', 'epoch', 'plateau', or 'none'.
    """
    # -------- 1. OPTIMIZERS --------
    optim_key = optim_name.lower()
    if optim_key == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_key == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, eps=1e-8)
    elif optim_key == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_key == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer specified: '{optim_name}'.")

    # -------- 2. LEARNING RATE SCHEDULERS --------
    scheduler = None

    if scheduler_name is None or scheduler_name.lower() == "none":
        step_type = "none"
    else:
        name = scheduler_name.lower()

        if name == "onecycle":
            if epochs is None or steps_per_epoch is None:
                raise ValueError("OneCycleLR requires both 'epochs' and 'steps_per_epoch'.")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 10,       # Peak LR set to 10x base LR
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.01,       # 1% warmup phase
                div_factor=25,        # Initial LR = max_lr / 25
                final_div_factor=1e4,
                anneal_strategy='cos'
            )
            step_type = "batch"

        elif name == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=lr / 10, max_lr=lr, mode='triangular2'
            )
            step_type = "batch"

        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=epochs, 
                eta_min=1e-6
            )
            step_type = "epoch"

        elif name == "cosine_restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2
            )
            step_type = "batch"

        elif name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.99
            )
            step_type = "epoch"

        elif name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',           # Maximizes validation metric (e.g., F1/F2)
                patience=5,           # Waits 5 epochs without improvement
                factor=0.5,           # Halves the learning rate
                min_lr=1e-7
            )
            step_type = "plateau"

        else:
            raise ValueError(f"Unknown scheduler specified: '{scheduler_name}'.")

    return optimizer, scheduler, step_type