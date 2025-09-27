#!/usr/bin/env python3
"""
Training script for Cantonese-Mandarin translation using Hunyuan-MT-7B
Supports both PyTorch Lightning and PaddlePaddle frameworks
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.utils import parse_training_args, setup_output_directory, log_training_config, validate_training_args
from src.training.pytorch_trainer import PyTorchTrainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping, MetricsLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PaddlePaddle availability flag
PADDLE_AVAILABLE = False

# Try to import frameworks
try:
    from src.models.pytorch.cantonese_translation import CantoneseTranslationModule
    PYTORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch model import failed: {e}")
    PYTORCH_AVAILABLE = False

# Note: PaddlePaddle is not available on macOS, so we skip it
# if PADDLE_AVAILABLE:
#     try:
#         from src.models.paddle.cantonese_translation import CantoneseTranslationModel
#         from src.training import PaddleTrainer
#     except ImportError as e:
#         logger.warning(f"PaddlePaddle model import failed: {e}")
#         PADDLE_AVAILABLE = False


def create_pytorch_trainer(args):
    """Create PyTorch trainer with callbacks"""
    # Initialize model
    logger.info(f"Loading PyTorch model: {args.model_name}")
    model = CantoneseTranslationModule(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        weight_decay=args.weight_decay
    )
    
    # Log model info
    model_info = log_model_info(model, "pytorch")
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=args.output_dir + "/checkpoints/best_model",
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5
        ),
        MetricsLogger(
            log_file=args.output_dir + "/logs/metrics.json"
        )
    ]
    
    # Create trainer
    trainer = PyTorchTrainer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=callbacks
    )
    
    return trainer, model_info


def create_paddle_trainer(args):
    """Create PaddlePaddle trainer with callbacks"""
    if not PADDLE_AVAILABLE:
        logger.error("PaddlePaddle not available")
        return None, None
    
    # Initialize model
    logger.info(f"Loading PaddlePaddle model: {args.model_name}")
    model = CantoneseTranslationModel(
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    # Log model info
    model_info = log_model_info(model, "paddle")
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=args.output_dir + "/checkpoints/best_model_paddle",
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        ),
        MetricsLogger(
            log_file=args.output_dir + "/logs/metrics_paddle.json"
        )
    ]
    
    # Create trainer
    trainer = PaddleTrainer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=callbacks
    )
    
    return trainer, model_info


def train_with_framework(args, framework: str):
    """Train with specified framework"""
    logger.info(f"Starting {framework} training...")
    
    if framework == "pytorch":
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        trainer, model_info = create_pytorch_trainer(args)
    elif framework == "paddle":
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddlePaddle not available")
        trainer, model_info = create_paddle_trainer(args)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    if trainer is None:
        raise RuntimeError(f"Failed to create {framework} trainer")
    
    # Save training metadata
    from src.training.utils import save_training_metadata
    save_training_metadata(Path(args.output_dir), args, model_info)
    
    return trainer


def main():
    """Main training function"""
    # Parse arguments
    args = parse_training_args()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Validate arguments
    validate_training_args(args)
    
    # Log configuration
    log_training_config(args)
    
    try:
        # Train based on framework selection
        if args.framework == "pytorch":
            trainer = train_with_framework(args, "pytorch")
            logger.info("PyTorch training completed successfully!")
            
        elif args.framework == "paddle":
            trainer = train_with_framework(args, "paddle")
            logger.info("PaddlePaddle training completed successfully!")
            
        elif args.framework == "both":
            logger.info("Training with both frameworks...")
            pytorch_trainer = train_with_framework(args, "pytorch")
            paddle_trainer = train_with_framework(args, "paddle")
            logger.info("Both framework training completed!")
            
        else:
            logger.error(f"Unsupported framework: {args.framework}")
            sys.exit(1)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()