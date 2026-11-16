"""
=========================================================================================
MODULE: Multi-Channel Morphological Hair & Artifact Removal (hair_removal_dullrazor.py)
=========================================================================================
DESCRIPTION:
    Implements an advanced multi-channel hair and ruler-mark removal pipeline (V19) 
    for dermoscopic images prior to deep learning classification.
    
    Methodology:
    1. Multi-Channel Decomposition & Filtering: Extracts directional gradients (Sobel), 
       high-frequency residuals, and morphological Top-Hat / Black-Hat responses across 
       Grayscale and CIE L*a*b* (Luma) channels.
    2. Heuristic Geometric Judge (V19): Evaluates connected components using minimum-area 
       bounding boxes (aspect ratio, filling factor, perimeter tortuosity) and lesion 
       overlap to distinguish true filiform hairs and peripheral ruler marks from 
       vascular structures or pigment networks.
    3. Morphological Scalpel: Resolves hair-lesion collisions ("ambiguous" candidates) 
       via deep erosion and binary geodesic propagation, isolating the hair shell 
       without eroding dense biological tissue.
    4. Navier-Stokes Inpainting: Reconstructs occluded tissue over the dilated artifact 
       mask and exports lossless LZW-compressed TIFF images using multithreaded I/O.
=========================================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_propagation
import concurrent.futures

# ==========================================================
# --- PATH CONFIGURATION IMPORT ---
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_paths import HAM10000_DIR

DATASET_ROOT = HAM10000_DIR

# Set to True only if you want to generate tripartite visual debug grids (Original | Mask | Clean)
SAVE_DEBUG_COMPARISONS = False


# ==========================================================
# --- 1. MORPHOLOGICAL & AUXILIARY FUNCTIONS ---
# ==========================================================

def validate_objects(base_mask: np.ndarray, reference_mask: np.ndarray, min_pixels: int = 1) -> np.ndarray:
    """
    Extracts connected components from the candidate mask and retains only complete 
    structures that spatially intersect with the high-frequency reference mask.
    """
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
    validated_mask = np.zeros_like(base_mask)
    
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(component_mask, reference_mask)
        
        if cv2.countNonZero(overlap) >= min_pixels:
            validated_mask = cv2.bitwise_or(validated_mask, component_mask)
            
    return validated_mask


def morphological_scalpel(ambiguous_mask: np.ndarray, scalpel_radius: int = 3) -> np.ndarray:
    """
    Resolves hair-crust collisions ('ambiguous' structures) using severe morphological 
    erosion followed by geodesic binary propagation to separate filiform hairs 
    from dense lesion cores.
    """
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(ambiguous_mask, connectivity=8)
    recovered_hairs_mask = np.zeros_like(ambiguous_mask)
    
    scalpel_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (scalpel_radius * 2 + 1, scalpel_radius * 2 + 1)
    )
    
    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8)
        # Deep erosion: destroys thin filiform structures while preserving dense tissue cores
        lesion_core = cv2.morphologyEx(comp_mask, cv2.MORPH_ERODE, scalpel_kernel)
        
        # If the core vanishes completely, the structure was purely a thick hair
        if cv2.countNonZero(lesion_core) == 0:
            recovered_hairs_mask = cv2.bitwise_or(recovered_hairs_mask, comp_mask * 255)
            continue
            
        # If the core survives, reconstruct the dense biological tissue via binary propagation
        lesion_mask_bool = comp_mask.astype(bool)
        seed_bool = lesion_core.astype(bool)
        
        reconstructed_bool = binary_propagation(seed_bool, mask=lesion_mask_bool)
        reconstructed_lesion = reconstructed_bool.astype(np.uint8) * 255
        
        # Subtract reconstructed tissue from the total component to isolate the outer hair shell
        clean_hairs = cv2.subtract(comp_mask * 255, reconstructed_lesion)
        recovered_hairs_mask = cv2.bitwise_or(recovered_hairs_mask, clean_hairs)
        
    return recovered_hairs_mask


# ==========================================================
# --- 2. V19 GEOMETRIC JUDGE LOGIC ---
# ==========================================================

def geometric_judge_v19(comp_mask: np.ndarray, stats_i: np.ndarray, dims: tuple, mode: str, lesion_mask: np.ndarray) -> str:
    """
    Evaluates a connected component's geometric and topological properties to classify it 
    into: 'hair', 'ambiguous', 'vascular', or 'discard'.
    """
    # 1. Spatial area extraction and mode-dependent volumetric thresholds
    area = stats_i[cv2.CC_STAT_AREA]
    area_max = 600 if mode == "bright" else 1000
    if mode == "sobel":
        area_max = 999999  # Directional gradient does not penalize initial area
    
    # 2. Topological extraction and Minimum-Area Rotated Bounding Box
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "discard"
    
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    _, (w_r, h_r), _ = rect
    
    # 3. Primary geometric metrics
    length, width = max(w_r, h_r), min(w_r, h_r) + 1e-5
    aspect_ratio = length / width
    filling_factor = area / (length * width + 1e-5)
    
    # 4. Topological Risk Assessment via lesion protection mask
    if lesion_mask is not None:
        lesion_overlap = cv2.bitwise_and(comp_mask, lesion_mask)
        in_risk_zone = cv2.countNonZero(lesion_overlap) >= (area * 0.90)
    else:
        in_risk_zone = False
    
    # 5. Perimeter Tortuosity calculation
    perimeter = cv2.arcLength(cnt, True)
    tortuosity = (perimeter / 2.0) / (length + 1e-5)

    # 6. Strict Heuristic Rules
    # Vascular protection: strong gradient inside lesion with high curvature
    if mode == "sobel" and in_risk_zone and tortuosity > 1.35:
        return "vascular"
    
    # Extreme morphology filter: highly elongated or minimal bounding-box filling
    if filling_factor < 0.20 or aspect_ratio > 7.0:
        if area > area_max:
            if filling_factor < 0.12:
                return "hair"       # Exception for extraordinarily long and thin hairs
            else:
                return "ambiguous"  # Route to morphological scalpel reconstruction
        else:
            return "hair"
            
    # Volumetric discard for large, non-filiform blobs/shadows
    if area > area_max:
        return "discard"
        
    # 7. Moderate Tolerance Rules
    is_geometric_hair = False
    if mode == "sobel":
        if (filling_factor > 0.85 and aspect_ratio > 2.5 and length > 15) or \
           (aspect_ratio > 4.5 and filling_factor < 0.55) or \
           (length > 60 and aspect_ratio > 2.2):
            is_geometric_hair = True
    elif aspect_ratio > 3.0 and filling_factor < 0.35:
        is_geometric_hair = True
        
    return "hair" if is_geometric_hair else "discard"


def process_channel_v19(binary_mask: np.ndarray, lesion_mask: np.ndarray, mode: str = "dark", dims: tuple = None) -> np.ndarray:
    """
    Routes connected components of a given feature channel through the V19 geometric judge 
    and applies context-aware filtering (including peripheral ruler mark detection).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    h, w = dims if dims else binary_mask.shape[:2]
    
    direct_hairs_mask = np.zeros((h, w), dtype=np.uint8)
    ambiguous_for_scalpel_mask = np.zeros((h, w), dtype=np.uint8)
    ruler_candidates = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:
            continue  # High-frequency spatial noise suppression
        
        comp_mask = (labels == i).astype(np.uint8)
        status = geometric_judge_v19(comp_mask, stats[i], (h, w), mode, lesion_mask)
        
        if mode == "rulers":
            x_o, y_o = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w_c, h_c = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            touches_border = (x_o <= 5) or (y_o <= 5) or (x_o + w_c >= w - 5) or (y_o + h_c >= h - 5)
            ruler_candidates.append({'mask': comp_mask, 'border': touches_border, 'status': status})
        else:
            if status == "hair":
                direct_hairs_mask = cv2.bitwise_or(direct_hairs_mask, comp_mask * 255)
            elif status == "ambiguous":
                ambiguous_for_scalpel_mask = cv2.bitwise_or(ambiguous_for_scalpel_mask, comp_mask * 255)

    # Spatial context evaluation for peripheral fiducial/ruler marks
    if mode == "rulers":
        valid_border_ticks = sum(1 for c in ruler_candidates if c['border'] and c['status'] in ["hair", "ambiguous"])
        total_border_ticks = sum(1 for c in ruler_candidates if c['border'])
        
        for cand in ruler_candidates:
            if cand['border']:
                if valid_border_ticks >= 1 and total_border_ticks >= 2:
                    direct_hairs_mask = cv2.bitwise_or(direct_hairs_mask, cand['mask'] * 255)
            else:
                if cand['status'] == "hair":
                    direct_hairs_mask = cv2.bitwise_or(direct_hairs_mask, cand['mask'] * 255)
                elif cand['status'] == "ambiguous":
                    ambiguous_for_scalpel_mask = cv2.bitwise_or(ambiguous_for_scalpel_mask, cand['mask'] * 255)
                elif cand['status'] in ["discard", "vascular"]:
                    pass
            
    # Collision resolution and tissue subtraction
    scalpel_hairs = morphological_scalpel(ambiguous_for_scalpel_mask, scalpel_radius=3)
    
    return cv2.bitwise_or(direct_hairs_mask, scalpel_hairs)


# ==========================================================
# --- 3. MAIN V19 CLEANING PIPELINE ---
# ==========================================================

def advanced_hair_removal_v19(img: np.ndarray, lesion_mask: np.ndarray = None) -> tuple:
    """
    Executes the complete V19 multi-channel hair detection and Navier-Stokes inpainting pipeline.
    Returns (clean_image, dilated_inpainting_mask).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_lab = lab[:, :, 0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral Filter: preserves sharp edges while suppressing superficial skin texture noise
    gray_smooth = cv2.bilateralFilter(gray, 5, 75, 75)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    # Dynamic fallback lesion mask if no ground-truth/K-Means segmentation is provided
    if lesion_mask is None:
        _, base_lesion_mask = cv2.threshold(gray_smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        base_lesion_mask = cv2.morphologyEx(base_lesion_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
        base_lesion_mask = cv2.morphologyEx(base_lesion_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    else:
        _, base_lesion_mask = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)

    # 1. Multi-channel feature extraction (Sobel X/Y directional gradients)
    sx = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.uint8(np.absolute(cv2.magnitude(sx, sy)))
    sobel_closed = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    th_sobel = cv2.morphologyEx(sobel_closed, cv2.MORPH_TOPHAT, kernel_bh)
    
    # 2. High-frequency isolation via absolute difference
    noise_closed = cv2.morphologyEx(cv2.absdiff(gray, gray_smooth), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    th_noise = cv2.morphologyEx(noise_closed, cv2.MORPH_TOPHAT, kernel_bh)
    
    # 3. Standard Morphological Filtering (Black-Hat / Top-Hat)
    bh_l = cv2.morphologyEx(l_lab, cv2.MORPH_BLACKHAT, kernel_bh)
    bh_gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
    th_gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_bh)

    def normalized_subtraction(a: np.ndarray, b: np.ndarray, weight_b: float = 0.5) -> np.ndarray:
        a_n = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        b_n = cv2.normalize(b.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        res = cv2.subtract(a_n, cv2.multiply(b_n, weight_b))
        return cv2.normalize(np.maximum(res, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4. Weighted background suppression and directional cross-subtractions
    sub_sobel = normalized_subtraction(th_sobel, sobel_closed)
    sub_noise = normalized_subtraction(th_noise, noise_closed)
    sub_l = normalized_subtraction(bh_l, l_lab)
    
    bhg_thg = cv2.subtract(bh_gray.astype(np.float32), th_gray.astype(np.float32))
    bhg_thg = cv2.normalize(np.maximum(bhg_thg, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    thg_bhg = cv2.subtract(th_gray.astype(np.float32), bh_gray.astype(np.float32))
    thg_bhg = cv2.normalize(np.maximum(thg_bhg, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5. Binary Segmentation (Adaptive local & Otsu global thresholding)
    adapt_sobel = cv2.adaptiveThreshold(sub_sobel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -3)
    adapt_sobel = cv2.morphologyEx(adapt_sobel, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    _, otsu_noise = cv2.threshold(sub_noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_l = cv2.threshold(sub_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_bhg_thg = cv2.threshold(bhg_thg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_thg_bhg = cv2.threshold(thg_bhg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Topological cross-validation and spatial consolidation
    val_px = 12
    sobel_val = validate_objects(adapt_sobel, otsu_noise, val_px)
    thin_dark_val = validate_objects(otsu_bhg_thg, otsu_noise, val_px)
    thin_bright_val = validate_objects(otsu_thg_bhg, otsu_noise, val_px)
    
    # Strict spatial restriction for ruler marks: operate only on healthy peripheral skin
    raw_rulers = cv2.bitwise_and(otsu_l, cv2.bitwise_not(base_lesion_mask))

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_k5 = cv2.morphologyEx(thin_dark_val, cv2.MORPH_CLOSE, k5)
    bright_k5 = cv2.morphologyEx(thin_bright_val, cv2.MORPH_CLOSE, k5)

    # 7. Final evaluation across V19 channel router
    modes = ["sobel", "dark", "bright", "rulers"]
    inputs = [sobel_val, dark_k5, bright_k5, raw_rulers]
    
    final_mask = np.zeros_like(gray)
    for ch_input, mode in zip(inputs, modes):
        final_mask = cv2.bitwise_or(final_mask, process_channel_v19(ch_input, lesion_mask, mode=mode))

    # 8. Tissue Reconstruction (Navier-Stokes Inpainting)
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(final_mask, dilation_kernel, iterations=1)
    clean_img = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_NS)
    
    return clean_img, dilated_mask


# ==========================================================
# --- 4. MULTITHREADED BATCH PROCESSING & I/O ---
# ==========================================================

def process_image_worker(args: tuple) -> bool:
    """
    Atomic worker function for asynchronous parallel execution.
    Processes, inpaints, and exports a single dataset image.
    """
    img_id, in_dir, mask_dir, out_dir, comp_dir, save_comparisons = args
    
    out_tiff_path = os.path.join(out_dir, img_id + '.tiff')
    out_comp_path = os.path.join(comp_dir, img_id + '_comp.jpg') if save_comparisons else None
    
    # O(1) Resume check: skip if target output already exists on disk
    if os.path.exists(out_tiff_path) and (not save_comparisons or os.path.exists(out_comp_path)):
        return True
    
    img_path = os.path.join(in_dir, img_id + '.jpg')
    if not os.path.exists(img_path):
        return False
        
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    lesion_mask = None
    if mask_dir:
        mask_path = os.path.join(mask_dir, img_id + '_segmentation.png')
        if os.path.exists(mask_path):
            lesion_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
    clean_img, final_mask = advanced_hair_removal_v19(img, lesion_mask)
    
    # Export lossless LZW-compressed 2D TIFF image
    cv2.imwrite(out_tiff_path, clean_img, (cv2.IMWRITE_TIFF_COMPRESSION, 5))
    
    # Optional visual tripartite comparison grid (Original | Mask | Clean)
    if save_comparisons and out_comp_path:
        m_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        comp = np.hstack((img, m_bgr, clean_img))
        cv2.imwrite(out_comp_path, comp)
    
    return True


def process_directories(config_list: list, save_comparisons: bool = False):
    """Iterates over dataset partition configurations and launches multithreaded cleaning."""
    for config in config_list:
        in_dir = config['input']
        mask_dir = config.get('mask')
        out_dir = os.path.join(config['output'], 'tiff_images')
        comp_dir = os.path.join(config['output'], 'debug_comparisons')
        
        if not os.path.exists(in_dir):
            print(f"⚠️ Warning: Input directory not found, skipping: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        if save_comparisons:
            os.makedirs(comp_dir, exist_ok=True)
        
        if config.get('csv') and os.path.isfile(config['csv']):
            df = pd.read_csv(config['csv'])
            image_ids = df.iloc[:, 1].astype(str).tolist()
        else:
            image_ids = [
                os.path.splitext(f)[0] for f in os.listdir(in_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

        print(f"\n🚀 Starting V19 Hair Removal Pipeline (Multithreaded with Resume): {in_dir}")
        print(f"📁 Output Destination: {out_dir}")
        
        worker_args = [
            (img_id, in_dir, mask_dir, out_dir, comp_dir, save_comparisons) 
            for img_id in image_ids
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(process_image_worker, worker_args),
                total=len(worker_args),
                desc="Cleaning dermoscopic images",
                unit="img"
            ))


if __name__ == '__main__':
    PIPELINE_CONFIG = [
        {
            'input': os.path.join(DATASET_ROOT, "HAM10000_images"),
            'output': os.path.join(DATASET_ROOT, "HAM10000_images_dullrazor"),
            'csv': os.path.join(DATASET_ROOT, "HAM10000_metadata.csv"),
            'mask': os.path.join(DATASET_ROOT, "HAM10000_segmentations_lesion_tschandl")
        },
        {
            'input': os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_Images"),
            'output': os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_Images_dullrazor"),
            'csv': os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_GroundTruth.csv"),
            'mask': os.path.join(DATASET_ROOT, "ISIC2018_Task3_Test_Masks_KMeans")
        }
    ]

    process_directories(PIPELINE_CONFIG, save_comparisons=SAVE_DEBUG_COMPARISONS)