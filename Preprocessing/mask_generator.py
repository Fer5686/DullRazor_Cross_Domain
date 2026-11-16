"""
=========================================================================================
MODULE: Automated K-Means Skin Lesion Mask Generator (mask_generator.py)
=========================================================================================
DESCRIPTION:
    Computes precise binary segmentation masks for dermoscopic images (ISIC/HAM10000 dataset) 
    using unsupervised K-Means color clustering paired with a central region validation 
    alarm and morphological security margins.
    
    Methodology:
    1. Downsampled K-Means Clustering: Resizes images to 150x150 for rapid color quantization 
       into 2 clusters (lesion vs. skin background).
    2. Central Alarm Filter: Eliminates false-positive peripheral artifacts by enforcing 
       that valid lesion contours intersect a central elliptic region of interest.
    3. Morphological Post-Processing & Security Margin: Applies opening/closing operations 
       followed by controlled dialation to ensure the mask fully covers boundary features 
       without under-segmentation.
    4. Multithreaded Execution: Parallelizes batch execution across CPU cores using 
       `ThreadPoolExecutor` with a live progress bar (`tqdm`).
=========================================================================================
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures

# --- PATH CONFIGURATION IMPORT ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_paths import HAM10000_DIR, DOCS_DIR

# Dinamically resolve relative directories based on config_paths
INPUT_DIR = os.path.join(HAM10000_DIR, "ISIC2018_Task3_Test_Images")
OUTPUT_DIR = os.path.join(DOCS_DIR, "HAM10000", "ISIC2018_Task3_Test_Masks_KMeans")


def apply_central_alarm(thresh: np.ndarray, img_shape: tuple) -> np.ndarray:
    """
    Filters contours: retains only the largest valid contour that intersects 
    the central elliptic region of the image.
    """
    h, w = img_shape[:2]
    central_region = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.20), int(h * 0.20))
    cv2.ellipse(central_region, center, axes, 0, 0, 360, 255, -1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    mask_final = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < 300: 
            continue
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if cv2.countNonZero(cv2.bitwise_and(temp_mask, central_region)) > 0:
            mask_final = temp_mask
            break
            
    return mask_final


def segment_kmeans(img: np.ndarray) -> np.ndarray:
    """
    Applies K-Means clustering to separate skin lesions from background based on color, 
    incorporating morphological opening/closing and an expanded security margin.
    """
    h, w = img.shape[:2]
    small_img = cv2.resize(img, (150, 150))
    pixel_values = small_img.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dark_cluster = np.argmin(np.sum(centers, axis=1))
    mask_small = np.zeros(labels.shape, dtype=np.uint8)
    mask_small[labels == dark_cluster] = 255
    mask_small = mask_small.reshape((150, 150))
    
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    
    # 1. Filter via central region alarm
    mask_validated = apply_central_alarm(mask_full, img.shape)
    
    # 2. Security Margin: Expand mask via morphological dilation
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_expanded = cv2.dilate(mask_validated, kernel_dilate, iterations=2)
    
    return mask_expanded


def worker(img_id: str) -> bool:
    """Worker function for parallelized image processing."""
    img_path = os.path.join(INPUT_DIR, img_id + '.jpg')
    out_path = os.path.join(OUTPUT_DIR, img_id + '_segmentation.png')
    
    if os.path.exists(out_path): 
        return True
    if not os.path.exists(img_path): 
        return False
    
    img = cv2.imread(img_path)
    if img is None: 
        return False
    
    mask = segment_kmeans(img)
    cv2.imwrite(out_path, mask)
    return True


def execute_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: Input directory does not exist: {INPUT_DIR}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_ids = [os.path.splitext(f)[0] for f in files]

    if not image_ids:
        print(f"❌ ERROR: No images found in {INPUT_DIR}")
        return

    print(f"🚀 Initializing K-Means segmentation pipeline with security margins: {len(image_ids)} images found.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(worker, image_ids), total=len(image_ids), unit="img"))


if __name__ == '__main__':
    execute_pipeline()