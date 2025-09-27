"""
Base model components and utilities for Cantonese-Mandarin translation
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

# Configure logging
logger = logging.getLogger(__name__)


class BaseTranslationModel(ABC):
    """Abstract base class for translation models"""
    
    def __init__(self, model_name: str, max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
    
    @abstractmethod
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        pass
    
    @abstractmethod
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and tokenizer"""
        if self.model:
            self.model.save_pretrained(save_directory)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")
    
    def from_pretrained(self, model_name_or_path: str):
        """Load model from pretrained"""
        self.model_name = model_name_or_path
        self._load_model()
        return self
    
    @abstractmethod
    def _load_model(self):
        """Load model and tokenizer"""
        pass


class BaseTrainer(ABC):
    """Abstract base class for trainers"""
    
    def __init__(self, model: BaseTranslationModel, learning_rate: float = 2e-5,
                 warmup_steps: int = 500, weight_decay: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer = None
        self.scheduler = None
    
    @abstractmethod
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        pass
    
    @abstractmethod
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model"""
        pass
    
    @abstractmethod
    def fit(self, train_dataloader, val_dataloader=None, epochs: int = 1) -> Dict[str, List[float]]:
        """Train the model"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save model"""
        self.model.save_pretrained(filepath)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model.get_model_info()


def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score (simplified implementation)"""
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    
    try:
        # Simple BLEU calculation - in practice use proper BLEU implementation
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if not pred_tokens or not ref_tokens:
                continue
            
            # Calculate precision
            matches = sum(1 for token in pred_tokens if token in ref_tokens)
            precision = matches / len(pred_tokens) if pred_tokens else 0.0
            
            total_score += precision
        
        return total_score / len(predictions) if predictions else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {str(e)}")
        return 0.0


def validate_model_config(model_name: str, max_length: int) -> bool:
    """Validate model configuration"""
    if not model_name or len(model_name.strip()) == 0:
        logger.error("Model name cannot be empty")
        return False
    
    if max_length < 1 or max_length > 2048:
        logger.error("Max length must be between 1 and 2048")
        return False
    
    return True


def log_model_info(model: BaseTranslationModel) -> None:
    """Log model information"""
    try:
        info = model.get_model_info()
        logger.info("Model Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")


def create_model_summary(model: BaseTranslationModel) -> str:
    """Create model summary"""
    try:
        info = model.get_model_info()
        return f"""
Model Summary:
- Name: {info.get('model_name', 'Unknown')}
- Parameters: {info.get('total_parameters', 0):,} total, {info.get('trainable_parameters', 0):,} trainable
- Framework: {info.get('framework', 'Unknown')}
- Device: {info.get('device', 'Unknown')}
"""
    except Exception as e:
        return f"Failed to create model summary: {str(e)}"