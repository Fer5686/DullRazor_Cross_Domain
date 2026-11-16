"""
=========================================================================================
MODULE: Validation Inference and Clinical Metric Computation (val.py)
=========================================================================================
DESCRIPTION:
    Executes gradient-free evaluation passes on the validation partition and computes 
    both generalist multiclass and melanoma-specific diagnostic metrics.
    
    Methodological Specifications (see Manuscript Section 'Bifurcation of the Clinical 
    Approach'):
    1. F2-Score Formulation: Weights Recall (sensitivity) twice as heavily as Precision 
       (F2 = 5 * P * R / (4 * P + R)) to penalize missed malignancies (false negatives) 
       without causing diagnostic collapse from unconstrained false positives.
    2. Dual Clinical Analytical Lens:
       - Generalist Multiclass Performance: Evaluated via the support-weighted F2-Score 
         across all seven dermoscopic categories to account for natural class imbalance.
       - Oncological Sensitivity (Melanoma Detection): Isolates the class-specific 
         F2-Score for melanoma ('mel') to measure predictive robustness on the highest-risk 
         malignancy.
    3. Confusion Matrix Normalization: Computes row-normalized confusion matrices 
       (recall per true diagnostic category) for epoch-level telemetry and auditing.
=========================================================================================
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from device_config import hardware_device


def validate_model(model, val_loader, criterion, mode='validation'):
    """
    Performs a forward inference pass over the validation DataLoader without gradient 
    tracking, returning the mean validation loss, predicted class indices, and true labels.
    """
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader): 
            images = batch['image'].to(hardware_device)
            labels = batch['label'].to(hardware_device)
            
            # Optional sanity check on the first validation batch:
            # if i == 0:
            #     print(f"🔍 [VAL TENSOR CHECK] Range: [{images.min():.2f}, {images.max():.2f}] | Mean: {images.mean():.2f} | Std: {images.std():.2f}")

            # Ensure 3-channel RGB input if a 4th mask channel is present
            if images.shape[1] == 4:
                images = images[:, :3]
            
            # Forward pass
            outputs = model(images)
            
            # Extract logits if the model returns a dictionary wrapper
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Compute standard classification loss
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(logits, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    print(f"Loss ({mode}): {val_loss:.4f}")
    return val_loss, all_val_preds, all_val_labels


def compute_model_metrics(all_preds, all_labels, label_mapping, class_counts, mode):
    """
    Computes per-class and support-weighted global metrics (Precision, Recall, F1-Score, 
    F2-Score), row-normalized confusion matrices, and isolated Melanoma F2-Score.
    
    Returns:
        tuple: (
            weighted_precision, 
            class_f2_dict, 
            weighted_f1, 
            weighted_recall, 
            conf_matrix_normalized, 
            weighted_f2, 
            mel_f2
        )
    """
    # Configure console header suffix based on evaluation mode
    if mode in ['cross_validation', 'train_val', 'val', 'validation']:
        mode_tag = ' [Validation]'
    elif mode == 'test':
        mode_tag = ' [Test]'
    elif mode == 'train':
        mode_tag = ' [Train]'
    else:
        mode_tag = ''

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Normalize label mapping dictionaries
    if isinstance(next(iter(label_mapping.keys())), str):
        numeric_to_class = {v: k for k, v in label_mapping.items()}
    else:
        numeric_to_class = {k: v for k, v in label_mapping.items()}

    labels_numeric = sorted(label_mapping.values())
    conf_matrix = confusion_matrix(labels, preds, labels=labels_numeric)
    conf_matrix_normalized = confusion_matrix(labels, preds, labels=labels_numeric, normalize='true')

    real_support = conf_matrix.sum(axis=1)
    class_metrics = {}
    
    # Per-class metric computation
    for i, class_num in enumerate(labels_numeric):
        class_name = numeric_to_class[class_num]
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # F2-Score (weights Recall twice as heavily as Precision)
        numerator = 5 * (precision * recall)
        denominator = (4 * precision) + recall
        f2 = numerator / denominator if denominator > 0 else 0.0
        
        class_metrics[class_name] = {
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'f2': f2, 
            'support': real_support[i]
        }

    # Support-weighted global metrics
    total_samples = real_support.sum()
    global_accuracy = conf_matrix.trace() / total_samples if total_samples > 0 else 0.0
    weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics.values()) / total_samples if total_samples > 0 else 0.0
    weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics.values()) / total_samples if total_samples > 0 else 0.0
    weighted_f1 = sum(m['f1'] * m['support'] for m in class_metrics.values()) / total_samples if total_samples > 0 else 0.0
    weighted_f2 = sum(m['f2'] * m['support'] for m in class_metrics.values()) / total_samples if total_samples > 0 else 0.0

    # Isolate Melanoma ('mel') F2-Score for oncological sensitivity tracking
    mel_f2 = class_metrics.get('mel', {'f2': 0.0})['f2']

    # --- CONSOLE TELEMETRY ---
    print(f"\nNormalized Confusion Matrix (Recall per class %){mode_tag}:")
    print(np.array_str(conf_matrix_normalized * 100, precision=2, suppress_small=True))
    
    print(f"\nPer-Class Metrics{mode_tag}:")
    for name, m in class_metrics.items():
        print(
            f"- {name}: Prec={m['precision']*100:.2f}%, Rec={m['recall']*100:.2f}%, "
            f"F1={m['f1']*100:.2f}%, F2={m['f2']*100:.2f}%, Support={m['support']}"
        )
    
    print(f"\nGlobal Metrics{mode_tag}:")
    print(f"- Accuracy:           {global_accuracy*100:.2f}%")
    print(f"- Weighted Precision: {weighted_precision*100:.2f}%")
    print(f"- Weighted Recall:    {weighted_recall*100:.2f}%")
    print(f"- Weighted F1-Score:  {weighted_f1*100:.2f}%")
    print(f"- Weighted F2-Score:  {weighted_f2*100:.2f}%")
    print(f"- Melanoma F2-Score:  {mel_f2*100:.2f}%")

    return (
        weighted_precision, 
        {k: v['f2'] for k, v in class_metrics.items()}, 
        weighted_f1, 
        weighted_recall, 
        conf_matrix_normalized, 
        weighted_f2, 
        mel_f2
    )