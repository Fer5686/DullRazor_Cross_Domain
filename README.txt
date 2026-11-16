# A Statistical Validation Framework for Evaluating Artifact Preprocessing Impact in Skin Lesion Classification

[![Paper](https://img.shields.io/badge/Paper-MethodsX-blue.svg)](#) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](#)

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

* **`Datasets/`**: Target directory for the raw datasets (HAM10000 & ISIC 2018). *Note: Raw `.jpg` images are ignored in version control. Only the algorithmically generated K-Means protective masks are versioned here.*
* **`Preprocessing/`**: Contains the customized DullRazor pipeline, the K-Means morphological mask generators, and the `tensor_packager.py` script to serialize the dataset into optimized `.pt` / `uint8` formats for CPU bottleneck mitigation.
* **`Main_code/`**: The core training loop. Contains architectural adaptations (AlexNet, VGG-16, ResNet-50, Swin, ConvNeXt), dynamic data augmentation loaders, Automatic Mixed Precision (AMP) logic, and Early Stopping mechanisms.
* **`Statistics/`**: Analytical scripts that process the output predictions. Calculates population distributions, performs Cross-Domain Shift Evaluation, and generates the final supplementary PDFs with granular confusion matrices.
* **`Tools/`**: Auxiliary scripts for environment verification and cross-domain debugging.

---

## 🚀 How to Reproduce the Study

### 1. Data Preparation
This framework utilizes the public HAM10000 dataset and the ISIC 2018 Task 3 Test ground truth.
1. Download the raw images and metadata from the official [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
2. Place the extracted folders (`HAM10000_images`, `ISIC2018_Task3_Test_Images`) inside the `Datasets/HAM10000/` directory.

### 2. Preprocessing & Tensor Serialization
To reproduce the customized DullRazor morphological filtering and optimize the data for GPU training:
```bash
python Preprocessing/hair_removal_dullrazor.py
python Preprocessing/tensor_packager.py

This will generate the required occlusion-free images and serialize all data into .pt tensors for high-bandwidth DataLoader ingestion.

3. Model Training

To execute the N=5 population training across all architectures and scenarios:

python Main_code/main.py

Outputs, model weights, and inference probabilities will be saved locally. Checkpoints are excluded from the repository to prevent git bloat.

4. Statistical Validation & Paper Outputs

To evaluate the models and generate the statistical tables (McNemar's tests, t-tests) and the supplementary MethodsX PDF:

python Statistics/macro_architecture_comparison.py
python Statistics/paper_tables_and_matrices.py
python Statistics/supplementary_figures_generator.py

🛠️ Environment & Dependencies

    OS: Windows 11 / Linux (Tested on Windows with Visual Studio 2022 build tools)

    Hardware: CUDA-capable GPU highly recommended (Tested on Nvidia RTX 3090, 24GB VRAM)

    Core Libraries: torch, torchvision, timm, opencv-python, scikit-learn, scipy, pandas, reportlab

(Ensure reportlab is installed in your environment to generate the supplementary PDF containing the confusion matrices).

📝 License & Acknowledgments

The code is provided open-source to support reproducible research in computational dermatology. If you utilize this framework or the customized preprocessing pipeline, please cite the associated MethodsX paper.

Data used in this repository relies on the HAM10000 benchmark by Tschandl et al. (2018).