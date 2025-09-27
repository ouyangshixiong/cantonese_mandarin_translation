"""
Dataset implementation for Cantonese-Mandarin translation
Supports both PyTorch and PaddlePaddle with OmegaConf configuration
"""
import logging
from typing import Dict, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Try to import both frameworks
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import paddle
    from paddle.io import Dataset as PaddleDataset
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Import framework-specific implementations
try:
    from .pytorch_dataset import CantoneseTranslationDataset as PyTorchDataset
    from .pytorch_dataset import CantoneseTranslationDataModule as PyTorchDataModule
except ImportError:
    PyTorchDataset = None
    PyTorchDataModule = None

try:
    from .paddle_dataset import CantoneseTranslationDatasetPaddle as PaddleDataset
    from .paddle_dataset import CantoneseTranslationDataModulePaddle as PaddleDataModule
except ImportError:
    PaddleDataset = None
    PaddleDataModule = None

# Import utilities
from .utils import create_mini_dataset, create_sample_data_splits, validate_dataset
from .base import BaseDataset


def create_dataset(data_path: str, tokenizer, max_length: int = 128, 
                  framework: str = "pytorch", mode: str = "train"):
    """Factory function to create appropriate dataset based on framework"""
    
    if framework == "pytorch":
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        if PyTorchDataset is None:
            raise RuntimeError("PyTorch dataset implementation not available")
        return PyTorchDataset(data_path, tokenizer, max_length, mode)
    
    elif framework == "paddle":
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddlePaddle not available")
        if PaddleDataset is None:
            raise RuntimeError("PaddlePaddle dataset implementation not available")
        return PaddleDataset(data_path, tokenizer, max_length, mode)
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def create_data_module(tokenizer, data_config: Dict, batch_size: int = 8, 
                      max_length: int = 128, num_workers: int = 4, framework: str = "pytorch"):
    """Factory function to create appropriate data module based on framework"""
    
    if framework == "pytorch":
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        if PyTorchDataModule is None:
            raise RuntimeError("PyTorch data module implementation not available")
        return PyTorchDataModule(tokenizer, data_config, batch_size, max_length, num_workers)
    
    elif framework == "paddle":
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddlePaddle not available")
        if PaddleDataModule is None:
            raise RuntimeError("PaddlePaddle data module implementation not available")
        return PaddleDataModule(tokenizer, data_config, batch_size, max_length, num_workers)
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# Export main classes and functions
__all__ = [
    'create_dataset',
    'create_data_module', 
    'create_mini_dataset',
    'create_sample_data_splits',
    'validate_dataset',
    'BaseDataset'
]

# Add framework-specific classes if available
if PyTorchDataset:
    __all__.append('CantoneseTranslationDataset')
if PyTorchDataModule:
    __all__.append('CantoneseTranslationDataModule')
if PaddleDataset:
    __all__.append('CantoneseTranslationDatasetPaddle')
if PaddleDataModule:
    __all__.append('CantoneseTranslationDataModulePaddle')