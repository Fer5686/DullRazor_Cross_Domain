"""
=========================================================================================
MODULE: Lesion-Aware Stratified Train/Validation Splitter (split_train_val.py)
=========================================================================================
DESCRIPTION:
    Partitions the official HAM10000 metadata file into Training (85%) and Validation (15%) 
    subsets at the unique patient/lesion level (`lesion_id`) rather than the image level.
    
    Why Lesion-Level Splitting is Critical:
    HAM10000 contains 10,015 dermoscopic images derived from 7,470 unique skin lesions 
    (multiple views, magnifications, or follow-up captures per lesion). Splitting at the 
    lesion level strictly prevents cross-partition data leakage, ensuring that no views 
    of a training lesion ever appear in the validation set.
=========================================================================================
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- PATH CONFIGURATION IMPORT ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_paths import HAM10000_DIR

# Resolve CSV paths dynamically relative to HAM10000_DIR from config_paths
ORIGINAL_CSV_PATH = os.path.join(HAM10000_DIR, "HAM10000_metadata.csv")
TRAIN_CSV_PATH = os.path.join(HAM10000_DIR, "HAM10000_metadata_train.csv")
VAL_CSV_PATH = os.path.join(HAM10000_DIR, "HAM10000_metadata_val.csv")


def generate_csv_splits(test_size: float = 0.15, random_state: int = 42):
    """
    Executes a deterministic, lesion-stratified split of the HAM10000 dataset 
    and runs a self-audit to verify zero data leakage.
    """
    if not os.path.exists(ORIGINAL_CSV_PATH):
        raise FileNotFoundError(f"❌ Original metadata file not found at: {ORIGINAL_CSV_PATH}")

    # 1. Load original metadata
    df = pd.read_csv(ORIGINAL_CSV_PATH)
    
    # 2. Group by `lesion_id` to obtain unique lesions and their diagnosis (`dx`).
    # Crucial to prevent different photographs of the same lesion from splitting across train and val.
    df_lesions = df.groupby('lesion_id').agg({'dx': 'first'}).reset_index()
    
    # 3. Stratified split (85% train, 15% val) strictly at the LESION level
    train_lesions, val_lesions = train_test_split(
        df_lesions['lesion_id'],
        test_size=test_size,
        stratify=df_lesions['dx'],
        random_state=random_state
    )
    
    # 4. Map selected lesion IDs back to all corresponding images in the original DataFrame
    df_train = df[df['lesion_id'].isin(train_lesions)]
    df_val = df[df['lesion_id'].isin(val_lesions)]
    
    # 5. Export partitioned CSV files
    df_train.to_csv(TRAIN_CSV_PATH, index=False)
    df_val.to_csv(VAL_CSV_PATH, index=False)
    
    # 6. Integrity & Leakage Self-Audit
    leaked_lesions = df_train['lesion_id'].isin(df_val['lesion_id']).sum()
    total_preserved = len(df_train) + len(df_val)
    
    print("=" * 65)
    print("✅ LESION-STRATIFIED SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"📄 Total Original Images: {len(df)} ({len(df_lesions)} unique lesions)")
    print(f"📁 TRAIN CSV saved at:    {TRAIN_CSV_PATH}")
    print(f"   └── Images: {len(df_train)} ({len(train_lesions)} unique lesions)")
    print(f"📁 VAL CSV saved at:      {VAL_CSV_PATH}")
    print(f"   └── Images: {len(df_val)} ({len(val_lesions)} unique lesions)")
    print("-" * 65)
    print(f"🔒 Data Leakage Check (Shared `lesion_id` count, must be 0): {leaked_lesions}")
    print(f"🔒 Sample Conservation Check ({total_preserved}/{len(df)}): {'PASSED' if total_preserved == len(df) else 'FAILED'}")
    print("=" * 65)
    
    assert leaked_lesions == 0, "CRITICAL ERROR: Data leakage detected between Train and Val!"
    assert total_preserved == len(df), "CRITICAL ERROR: Image count mismatch after splitting!"


if __name__ == "__main__":
    generate_csv_splits()