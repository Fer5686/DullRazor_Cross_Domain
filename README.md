# A Statistical Validation Framework for Evaluating Artifact Preprocessing Impact in Skin Lesion Classification

[![Paper](https://img.shields.io/badge/Paper-MethodsX-blue.svg)](#) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24384128.svg)](https://doi.org/10.5281/zenodo.24384128) [![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](#)

This repository contains the official open-source implementation for the methodological article: **"A statistical validation framework for evaluating artifact preprocessing impact in skin lesion classification"** (Submitted to *MethodsX*).

It provides a comprehensive pipeline to isolate, quantify, and statistically validate the true impact of morphological preprocessing (such as the DullRazor algorithm) versus native domain shift across deep neural networks of varying capacities (AlexNet, VGG-16, ResNet-50, Swin Transformer, and ConvNeXt).

---

## ⚠️ Important Note on Reproducibility and Stochastic Variance

Deep learning models trained with GPU acceleration involve inherent mathematical non-determinism. Due to stochastic batch shuffling, parallel reduction race conditions, and low-level cuDNN atomic operations, **individual training runs will not yield exactly identical scalar metrics, even when pseudo-random seeds are fixed.**

If you run a single training instance from this repository, your specific F2-Score or accuracy will likely differ slightly from the exact numbers reported in the paper. 

**This is the exact problem this framework solves.** 
The methodology is designed to abandon the "cherry-picking" of isolated runs. Instead, the framework evaluates **population-level statistical distributions** ($N=5$ independent runs per architecture/condition). By aggregating multiple runs, selecting out-of-sample medians, and applying rigorous tests (McNemar's, Welch's t-test, Paired t-test), the framework extracts objective, statistically significant conclusions about architectural behavior that are immune to stochastic initialization noise.

---

## 📁 Repository Structure

To ensure a clean and scalable environment, the project is strictly modularized:

- **`Datasets/`**: Target directory for the raw datasets (HAM10000 and ISIC 2018 Task 3). *Note: Raw `.jpg` images are excluded from version control. Only the algorithmically generated K-Means protective test masks are versioned here.*
- **`Preprocessing/`**: Contains the deterministic lesion-grouped train/validation split (`split_train_val.py`), K-Means protective mask generation (`mask_generator.py`), the customized DullRazor pipeline (`hair_removal_dullrazor.py`), and `tensor_packager.py` to serialize datasets into optimized `.pt` (`uint8`) tensors for CPU bottleneck mitigation.
- **`Main_code/`**: The core training loop (`main.py`) and architectural training modules (AlexNet, VGG-16, ResNet-50, Swin Transformer, and ConvNeXt), dynamic data augmentation loaders, Automatic Mixed Precision (AMP) logic, and Early Stopping mechanisms.
- **`Statistics/`**: Analytical scripts that extract run metrics (`clean_data_extractor.py`), compute population distributions, perform paired and cross-domain statistical tests, and generate the manuscript tables and supplementary figures.
- **`Tools/`**: Auxiliary scripts for native and cross-domain checkpoint verification (`verify_models.py`, `verify_models_cross_domain.py`).

---

## 🚀 How to Reproduce the Study

### 1. Data Preparation
Raw dermatoscopic images and clinical metadata are not redistributed in this repository or in the Zenodo archive; they must be obtained from their official sources:
1. Download the HAM10000 images, metadata CSV, and lesion segmentation masks from the official [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), along with the ISIC 2018 Task 3 Test images and ground truth CSV.
2. Place the extracted folders and metadata files (`HAM10000_images`, `ISIC2018_Task3_Test_Images`, `HAM10000_metadata`, `ISIC2018_Task3_Test_GroundTruth`, and segmentation masks if used) inside `Datasets/HAM10000/`.

### 2. Preprocessing & Tensor Serialization
To generate the stratified lesion-grouped split (`random_state=42`), apply the customized DullRazor morphological filtering, and serialize the dataset into `.pt` tensors for GPU training:
```bash
python Preprocessing/split_train_val.py
# python Preprocessing/mask_generator.py  # Optional: K-Means test masks are already versioned in Datasets/
python Preprocessing/hair_removal_dullrazor.py
python Preprocessing/tensor_packager.py
```
This produces the lesion-grouped split, occlusion-free images, and serialized `.pt` tensors for high-bandwidth DataLoader ingestion.

### 3. Model Training
To execute the complete paired cohort of 50 training runs (5 architectures x 5 independent seeds x 2 conditions: With Hair vs. Without Hair):
```bash
python Main_code/main.py
```
Outputs, model weights, logs, and inference probabilities are saved locally. Checkpoints and prediction arrays (`.pth`, `.npy`) are excluded from Git to prevent repository bloat.

### 4. Statistical Validation & Paper Outputs
To compute the statistical comparisons (Welch's t-test, seed-paired t-test, and McNemar's test) and generate all manuscript tables and supplementary figures:
```bash
python Statistics/macro_architecture_comparison.py
python Statistics/paper_tables_and_matrices.py
python Statistics/supplementary_figures_generator.py
```

### 5. Reproducing Tables Without Retraining (Zenodo Archive)
All precomputed evaluation metrics (`score_*.csv`), prediction arrays (`.npy`), and run logs for the 50 official runs are permanently archived on Zenodo:

- **Zenodo Dataset (v1.0.0):** [https://doi.org/10.5281/zenodo.24384128](https://doi.org/10.5281/zenodo.24384128)

To reproduce all tables and figures from the manuscript directly without retraining, download `dullrazor_cross_domain_eval_results.zip`, extract its contents into the expected models directory, and run the scripts in `Statistics/`.

---

## 🛠️ Environment & Dependencies

- **OS:** Windows 11 / Linux (Tested on Windows with Visual Studio 2022 build tools)
- **Hardware:** CUDA-capable GPU highly recommended (Tested on NVIDIA RTX 3090, 24GB VRAM)
- **Python:** Tested with Python `3.11.13`
- **Dependencies:** Exact package versions are pinned in `requirements.txt` (`torch==2.7.1`, `torchvision==0.24.0`, `timm==1.0.28`, `opencv-python==4.12.0`, `scikit-learn==1.7.2`, `scipy==1.16.0`, `statsmodels==0.14.6`, `numpy==2.3.1`, `pandas==2.3.3`, `matplotlib==3.10.7`, `Pillow==11.3.0`, `reportlab==5.1.0`, `tqdm==4.67.1`, `openpyxl==3.1.5`, `XlsxWriter==3.2.9`).

Install dependencies with:
```bash
pip install -r requirements.txt
```

> **Note on PyTorch/CUDA installation:** The original environment was built using Conda (`conda-forge`) to handle GPU/CUDA dependencies automatically (`conda install -c conda-forge pytorch-gpu torchvision`). If installing purely via `pip`, install a matched `torch`/`torchvision` pair for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) (e.g., `torch==2.7.1` with `torchvision==0.22.1`) before installing the remaining packages.

---

## 📝 License & Acknowledgments

This project is provided open-source under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license to support reproducible research in computational dermatology.

If you utilize this framework, the customized preprocessing pipeline, or the archived evaluation results, please cite the associated *MethodsX* manuscript and the Zenodo archive (`10.5281/zenodo.24384128`).

- **Data Attribution:** This repository utilizes data from the HAM10000 dataset (Tschandl et al., 2018) and the ISIC 2018 Task 3 benchmark (Codella et al., 2019).