"""
Training utilities and components for Cantonese-Mandarin translation
"""
from .utils import parse_training_args, setup_output_directory, log_model_info
from .pytorch_trainer import PyTorchTrainer

# Try to import PaddlePaddle components
try:
    from .paddle_trainer import PaddleTrainer
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    PaddleTrainer = None

from .callbacks import TrainingCallback, ModelCheckpoint, EarlyStopping

__all__ = [
    'parse_training_args',
    'setup_output_directory', 
    'log_model_info',
    'PyTorchTrainer',
    'TrainingCallback',
    'ModelCheckpoint',
    'EarlyStopping'
]

if PADDLE_AVAILABLE:
    __all__.append('PaddleTrainer')