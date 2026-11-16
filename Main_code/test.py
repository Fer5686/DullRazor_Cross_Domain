"""
=========================================================================================
MODULE: Post-Training Evaluation and Probability Export Engine (test.py)
=========================================================================================
DESCRIPTION:
    Executes post-training evaluation across data partitions (TRAIN, VAL, TEST), exports 
    raw probability distributions and true label arrays for statistical testing 
    (Welch's t-test, Paired Student's t-test, and Paired McNemar's test), and generates 
    standardized performance reports and confusion matrices.
    
    Key functionalities:
    1. Smart Checkpoint Synchronization: Dynamically resolves 'model.' wrapper prefixes 
       when loading saved .pth state dictionaries.
    2. Statistical Artifact Export: Saves 'probs.npy' and 'labels.npy' per evaluated 
       checkpoint to enable out-of-sample median selection and case-by-case McNemar 
       agreement auditing (N=1,511).
    3. Clinical Metric Reporting: Computes support-weighted F1/F2 scores, isolated 
       Melanoma F2-Score, and normalized confusion matrices via val.compute_model_metrics.
=========================================================================================
"""

import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from device_config import hardware_device
from instantiate_datasets import label_mapping
from saving_functions import save_test_results
from val import compute_model_metrics


def test_model(
    start_time, 
    model, 
    data_loader, 
    criterion_test, 
    class_counts, 
    model_weights_path, 
    save_folder, 
    mode='test', 
    **kwargs
):
    """
    Loads a saved model checkpoint, runs full partition inference, exports probability 
    arrays ('probs.npy', 'labels.npy'), and saves clinical evaluation metrics and 
    confusion matrix plots.
    """
    # Backward compatibility for legacy Spanish keyword argument ('modo')
    if 'modo' in kwargs:
        mode = kwargs['modo']

    os.makedirs(save_folder, exist_ok=True)

    # 1. Smart Checkpoint Loading
    torch.cuda.empty_cache()
    
    if model_weights_path is not None:
        if os.path.exists(model_weights_path):
            print(f"\n⚖️ LOADING CHECKPOINT WEIGHTS: {os.path.basename(model_weights_path)}")
            try:
                state_dict = torch.load(model_weights_path, map_location=hardware_device, weights_only=True)
                
                model_keys = model.state_dict().keys()
                has_prefix_model = any(k.startswith('model.') for k in model_keys)
                has_prefix_in_file = any(k.startswith('model.') for k in state_dict.keys())

                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if has_prefix_model and not has_prefix_in_file:
                        name = 'model.' + k
                    elif not has_prefix_model and has_prefix_in_file:
                        name = k[6:]
                    new_state_dict[name] = v

                model.load_state_dict(new_state_dict, strict=True)
                model.to(hardware_device)
                print("✅ Checkpoint weights synchronized successfully.")
            except Exception as e:
                print(f"❌ Critical error loading weights in test_model: {e}")
                return
        else:
            print(f"❌ Error: Checkpoint file not found at {model_weights_path}")
            return
    else:
        print("\n🔄 Evaluating model currently loaded in memory (Verification Mode).")

    # 2. Partition Inference Loop
    model.eval()
    running_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Inference {mode.upper()}", leave=False):
            if isinstance(batch, dict):
                imgs, lbls = batch['image'], batch['label']
            else:
                imgs, lbls = batch

            imgs = imgs[:, :3].to(hardware_device) 
            lbls = lbls.to(hardware_device)
            
            if imgs.dtype == torch.uint8:
                imgs = imgs.float().div(255.0)

            outputs = model(imgs)
            if isinstance(outputs, dict):
                outputs = outputs.get('logits', outputs.get('output', outputs))

            loss = criterion_test(outputs, lbls)
            running_loss += loss.item() * imgs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbls.cpu().numpy())

    final_probs = np.concatenate(all_probs)
    final_preds = np.concatenate(all_preds)
    final_labels = np.concatenate(all_labels)
    test_loss = running_loss / len(data_loader.dataset)

    # 3. Export raw probabilities and ground-truth labels for statistical testing
    np.save(os.path.join(save_folder, "probs.npy"), final_probs)
    np.save(os.path.join(save_folder, "labels.npy"), final_labels)

    # 4. Compute Clinical Performance Metrics
    (
        weighted_precision, 
        class_precision, 
        weighted_f1, 
        weighted_recall, 
        conf_matrix_normalized, 
        weighted_f2, 
        mel_f2
    ) = compute_model_metrics(
        final_preds, final_labels, label_mapping, class_counts, mode=mode
    )
    
    print(f"\n📊 SUMMARY ({mode.upper()}):")
    print(f"Loss: {test_loss:.4f} | F1: {weighted_f1*100:.2f}% | F2: {weighted_f2*100:.2f}% | Mel F2: {mel_f2*100:.2f}%")

    # 5. Export Final Evaluation Report and Confusion Matrix Plots
    history_lite = {
        'epoch': [0], 
        'test_loss': [test_loss],
        'class_precision': [class_precision],
        'precision': [weighted_precision],
        'recall': [weighted_recall],
        'f1_score': [weighted_f1],
        'f2_score': [weighted_f2],
        'mel_f2': [mel_f2],
        'conf_matrix': [confusion_matrix(final_labels, final_preds, labels=list(label_mapping.values()))]
    }

    save_test_results(
        start_time, history_lite, conf_matrix_normalized, 
        train_val_found='no', mode=mode, output_dir_path=save_folder
    )
    
    print(f"✅ Evaluation completed and saved in: {save_folder}")