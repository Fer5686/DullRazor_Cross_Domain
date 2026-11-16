"""
=========================================================================================
MODULE: Cross-Platform Relative Path Configuration (config_paths.py)
=========================================================================================
DESCRIPTION:
    Centralizes base directory resolution and path definitions for datasets, model 
    checkpoints, and experimental telemetry artifacts, ensuring seamless execution 
    across operating systems without hardcoded user dependencies.
    
    Key functionalities:
    1. Relative Base Resolution: Automatically anchors paths relative to the project root.
    2. Environment Variable Overrides: Allows custom storage directories via `.env` or 
       system environment variables (`MODEL_OUTPUT_DIR`, `DATASET_DIR`, `HAM10000_PATH`).
    3. Automatic Directory Provisioning: Ensures target output and dataset folders exist.
=========================================================================================
"""

import os

# Define the project root directory (parent folder of 'Main_code')
DEFAULT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Resolve core directories with secure environment fallbacks
REPO_DIR = os.getenv("REPO_ROOT_DIR", DEFAULT_BASE)
MODELS_DIR = os.getenv("MODEL_OUTPUT_DIR", os.path.join(REPO_DIR, "trained_models"))
DOCS_DIR = os.getenv("DATASET_DIR", os.path.join(DEFAULT_BASE, "Datasets"))
HAM10000_DIR = os.getenv("HAM10000_PATH", os.path.join(DOCS_DIR, "HAM10000"))

# Ensure essential directories are automatically created if they don't exist
os.makedirs(REPO_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(HAM10000_DIR, exist_ok=True)