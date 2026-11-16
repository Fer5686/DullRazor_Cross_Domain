"""
=========================================================================================
MODULE: Hardware Device Configuration Manager (device_config.py)
=========================================================================================
DESCRIPTION:
    Centralizes hardware target allocation (CPU vs. CUDA GPU) using a Singleton-like 
    static manager pattern. Ensures that all neural network models, data tensors, and 
    loss criteria across the framework are instantiated and allocated on a unified 
    compute device.
    
    Hardware Reference (see Manuscript Section 'Hardware Configuration'):
    Experiments were conducted on an NVIDIA RTX 3090 GPU (24 GB VRAM) paired with an 
    AMD Ryzen 5 5600X processor and 48 GB RAM.
=========================================================================================
"""

import os
import torch

# Optional: Uncomment for synchronous CUDA error debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class DeviceManager:
    """
    Static manager class to initialize and retrieve the active PyTorch compute device.
    """
    _device_initialized = None

    @staticmethod
    def set_device(device_type="auto"):
        """
        Configures the active compute device according to the specified type.
        
        Args:
            device_type (str): 
                - "cpu": Forces execution on the host CPU.
                - "cuda": Forces execution on the NVIDIA GPU (requires CUDA availability).
                - "auto" (default): Automatically selects CUDA if available, otherwise CPU.
        """
        if device_type == "cpu":
            DeviceManager._device_initialized = torch.device("cpu")
        elif device_type == "cuda":
            if torch.cuda.is_available():
                DeviceManager._device_initialized = torch.device("cuda")
            else:
                raise RuntimeError("CUDA was explicitly requested, but no compatible NVIDIA GPU is available.")
        elif device_type == "auto":
            DeviceManager._device_initialized = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(f"Unknown device type specified: '{device_type}'. Expected 'cpu', 'cuda', or 'auto'.")

    @staticmethod
    def get_device():
        """
        Returns the currently initialized torch.device object. 
        Initializes to 'auto' if not previously configured.
        """
        if DeviceManager._device_initialized is None:
            DeviceManager.set_device("auto")
        return DeviceManager._device_initialized


# Explicitly set the compute device to GPU (CUDA) for the training pipeline
DeviceManager.set_device("cuda")
hardware_device = DeviceManager.get_device()

print(f"Configured compute device: {hardware_device}")