"""
=========================================================================================
MODULE: Core Epoch Training Engine with Curriculum Learning & AMP (train.py)
=========================================================================================
DESCRIPTION:
    Implements the per-epoch forward and backward propagation loop for deep neural 
    network training. Designed to maximize GPU throughput and stabilize convergence.
    
    Methodological Specifications (see Manuscript Section 'Dynamic Augmentation and 
    Optimized Data Loading' & Figure 7, Stage 2):
    1. Automatic Mixed Precision (AMP): Executes forward passes and loss computations 
       in float16 precision via torch.autocast, paired with dynamic gradient scaling 
       (GradScaler) to prevent underflow during backpropagation.
    2. Curriculum Learning Schedule: Dynamically scales data augmentation severity as 
       training epochs progress:
       - RandAugment probability increases linearly (+0.01/epoch, up to 0.50).
       - RandAugment magnitude increases step-wise (2 + epoch // 5, up to 9).
       - CutMix spatial regularization probability increases linearly (+0.006/epoch, up to 0.50).
       This allows the network to assimilate fundamental lesion morphology in early 
       epochs before enforcing aggressive geometric and chromatic invariance.
    3. Flexible Learning Rate Stepping: Supports phase-constant learning rates 
       (step_type="none", default for baseline benchmarks) as well as intra-phase 
       per-epoch ("epoch") or per-batch ("batch") scheduler updates.
    4. Numerical Stability Guard: Actively monitors batch loss values for NaN or 
       exploding gradients (>= 1000), safely aborting the run if divergence occurs.
=========================================================================================
"""

import gc
import random
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from device_config import hardware_device


def train_model_dynamic_cutmix(
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    penalty_matrix=None, 
    use_penalty_matrix=False, 
    use_amp=True, 
    epoch=0, 
    epochs=500, 
    scheduler=None, 
    step_type="none",  # Options: "none" (constant per phase), "epoch" (decay per epoch), or "batch" (decay per batch)
    cutmix_alpha=1.0, 
    cutmix_prob=0.5,
    use_curriculum=True, 
    loss_threshold=0.5,   
    da_max_prob=0.5, 
    da_max_mag=9, 
    cutmix_max_prob=0.5, 
    phase_name="Training"   
):
    """
    Executes a single training epoch integrating Automatic Mixed Precision (AMP), 
    on-the-fly CutMix spatial regularization, and Curriculum Learning updates for 
    RandAugment intensities.
    
    Args:
        model (nn.Module): Target neural architecture.
        train_loader (DataLoader): PyTorch DataLoader yielding normalized image batches.
        optimizer (torch.optim.Optimizer): Active optimizer instance (e.g., AdamW).
        criterion (nn.Module): Loss function (e.g., Class-Balanced Cross-Entropy).
        penalty_matrix (torch.Tensor, optional): Cost matrix for misclassifications.
        use_penalty_matrix (bool): Whether to apply the penalty matrix.
        use_amp (bool): Enables float16 Automatic Mixed Precision.
        epoch (int): Current global or phase epoch index (0-based).
        epochs (int): Total estimated epochs for progress display.
        scheduler: Learning rate scheduler instance.
        step_type (str): Controls when scheduler.step() is triggered:
            - "none": Keeps LR constant within the current phase (default; LR changes across phases).
            - "epoch": Updates the scheduler once at the end of each epoch.
            - "batch": Updates the scheduler after every individual training batch.
        cutmix_alpha (float): Beta distribution alpha parameter for CutMix.
        cutmix_prob (float): Base probability for CutMix when curriculum is disabled.
        use_curriculum (bool): Linearly scales RandAugment and CutMix intensities by epoch.
        loss_threshold (float): Auxiliary threshold parameter.
        da_max_prob (float): Maximum RandAugment probability cap (default: 0.5).
        da_max_mag (int): Maximum RandAugment magnitude cap (default: 9).
        cutmix_max_prob (float): Maximum CutMix application probability cap (default: 0.5).
        phase_name (str): Descriptive label for console telemetry.

    Returns:
        tuple: (status_str, avg_loss, all_preds, all_labels, num_samples, avg_lr)
    """
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    all_preds, all_labels = [], []
    result = 'continue'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    learning_rates = []
    
    # 1. Configure Augmentation Intensities (Curriculum Learning vs. Static)
    if use_curriculum:
        base_randaug_prob = min(0.0 + epoch * 0.01, da_max_prob)
        base_randaug_mag = min(2 + epoch // 5, da_max_mag)
        base_cutmix_prob = min(0.0 + epoch * 0.006, cutmix_max_prob)
    else:
        base_randaug_prob, base_randaug_mag, base_cutmix_prob = da_max_prob, da_max_mag, cutmix_prob
    
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=7) if base_cutmix_prob > 0.01 else None
    total_batches = len(train_loader)
    cutmix_applied = 0
    
    # 2. Batch Iteration Loop
    for i, batch in enumerate(progress_bar):
        # Transfer normalized 3-channel RGB tensors to GPU VRAM
        images = batch['image'][:, :3].to(hardware_device, non_blocking=True)
        labels = batch['label'].to(hardware_device, non_blocking=True)
        
        # Single sanity check on Epoch 1, Batch 1 to verify normalization statistics
        if epoch == 0 and i == 0:
            print(
                f"\n🔍 [TRAIN TENSOR CHECK] Range: [{images.min():.2f}, {images.max():.2f}] | "
                f"Mean: {images.mean():.2f} | Std: {images.std():.2f}"
            )

        # Apply stochastic CutMix regularization across the active batch
        if cutmix is not None and random.random() < base_cutmix_prob:
            images, labels = cutmix(images, labels)
            cutmix_applied += 1
                
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass under float16 mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss_ce = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            if use_penalty_matrix and penalty_matrix is not None:
                idx_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
                loss = loss_ce + penalty_matrix[idx_labels, preds].mean()
            else:
                loss = loss_ce

        # Numerical divergence guard prior to backpropagation
        if torch.isnan(loss) or loss >= 1000:
            result = 'nan'
            break

        # Backpropagation with AMP gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Per-batch scheduler step (if step_type == "batch")
        if scheduler is not None and step_type == "batch":
            scheduler.step()

        # Collect batch telemetry
        actual_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(actual_labels.cpu().numpy())
        epoch_loss += loss_ce.item()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})

    # 3. Per-epoch scheduler step (if step_type == "epoch")
    if result != 'nan' and scheduler is not None and step_type == "epoch":
        scheduler.step()

    # 4. Update dataset RandAugment intensities for the upcoming epoch
    if hasattr(train_loader.dataset, 'set_intensities'):
        train_loader.dataset.set_intensities(base_randaug_prob, base_randaug_mag, 1)

    avg_loss = epoch_loss / len(progress_bar) if len(progress_bar) > 0 else 0.0
    avg_lr = sum(learning_rates) / len(learning_rates) if learning_rates else optimizer.param_groups[0]['lr']
    
    # 5. Console Epoch Summary
    print(f"\n" + "-" * 45)
    print(f"📊 Training Summary - Epoch {epoch+1} - Phase: {phase_name}")
    print(f"   🔸 Training Loss: {avg_loss:.4f}")
    print(f"   🔸 RandAugment Prob: {base_randaug_prob:.1%} | Mag: {base_randaug_mag}")
    print(f"   🔸 CutMix Applied: {cutmix_applied}/{total_batches} batches")
    print(f"   🔸 Learning Rate: {avg_lr:.2e}")
    print("-" * 45)

    torch.cuda.empty_cache()
    gc.collect()
    
    return result, float(avg_loss), all_preds, all_labels, len(train_loader.dataset), float(avg_lr)