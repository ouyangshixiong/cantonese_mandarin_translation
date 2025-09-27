"""
Training utility functions
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_training_args():
    """Parse command line arguments for training"""
    parser = argparse.ArgumentParser(description="Train Cantonese-Mandarin translation model")
    
    # Framework selection
    parser.add_argument("--framework", type=str, choices=["pytorch", "paddle", "both"], 
                       default="pytorch", help="Deep learning framework to use")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Tencent-Hunyuan/Hunyuan-MT-7B",
                       help="Hunyuan model name or path")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Hardware configuration
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--precision", type=int, choices=[16, 32], default=16,
                       help="Model precision (16 or 32)")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for checkpoints")
    parser.add_argument("--save_top_k", type=int, default=1,
                       help="Save top k checkpoints")
    
    # Validation configuration
    parser.add_argument("--val_check_interval", type=float, default=0.25,
                       help="Validation check interval")
    
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    return output_path


def log_model_info(model, framework: str = "pytorch") -> Dict[str, Any]:
    """Log and return model information"""
    if framework == "pytorch":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif framework == "paddle":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    model_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "framework": framework
    }
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    return model_info


def log_training_config(args: argparse.Namespace) -> None:
    """Log training configuration"""
    logger.info("Training configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")


def validate_training_args(args: argparse.Namespace) -> None:
    """Validate training arguments"""
    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1")
    
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if args.max_length < 1:
        raise ValueError("Max length must be at least 1")
    
    logger.info("Training arguments validated successfully")


def create_output_subdirs(output_dir: Path) -> Dict[str, Path]:
    """Create output subdirectories"""
    subdirs = {
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs", 
        "models": output_dir / "models",
        "results": output_dir / "results"
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def save_training_metadata(output_dir: Path, args: argparse.Namespace, 
                          model_info: Dict[str, Any]) -> None:
    """Save training metadata to file"""
    import json
    import datetime
    
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "arguments": vars(args),
        "model_info": model_info,
        "system_info": {
            "python_version": get_python_version(),
            "framework": args.framework
        }
    }
    
    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training metadata saved to: {metadata_file}")


def get_python_version() -> str:
    """Get Python version"""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"