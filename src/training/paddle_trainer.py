"""
PaddlePaddle-specific training implementation
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import paddle
import paddle.nn as nn
from paddle.optimizer import AdamW
from paddle.io import DataLoader

# Configure logging
logger = logging.getLogger(__name__)


class PaddleTrainer:
    """PaddlePaddle trainer for Cantonese-Mandarin translation"""
    
    def __init__(self, model: nn.Layer, learning_rate: float = 2e-5,
                 warmup_steps: int = 500, weight_decay: float = 0.01,
                 device: str = "auto", callbacks: Optional[List] = None):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.device = self._setup_device(device)
        self.callbacks = callbacks or []
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        
        # Move model to device
        if self.device == "gpu":
            self.model = self.model.cuda()
        
        logger.info(f"PaddlePaddle trainer initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                device = "gpu"
            else:
                device = "cpu"
        
        return device
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer"""
        return AdamW(
            learning_rate=self.learning_rate,
            parameters=self.model.parameters(),
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        # Simple linear warmup scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0
        
        return lr_lambda
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if self.should_stop:
                break
            
            # Move batch to device
            if self.device == "gpu":
                batch = {k: v.cuda() if isinstance(v, paddle.Tensor) else v 
                        for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            paddle.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.clear_grad()
            
            # Update metrics
            total_loss += float(loss)
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {float(loss):.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with paddle.no_grad():
            for batch in val_loader:
                # Move batch to device
                if self.device == "gpu":
                    batch = {k: v.cuda() if isinstance(v, paddle.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                total_loss += float(loss)
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate BLEU score (placeholder)
        bleu_score = self._calculate_bleu(val_loader)
        
        return {
            "val_loss": avg_loss,
            "val_bleu": bleu_score
        }
    
    def _calculate_bleu(self, data_loader: DataLoader) -> float:
        """Calculate BLEU score (placeholder implementation)"""
        # This is a simplified BLEU calculation
        # In practice, you would use a proper BLEU implementation
        return 0.5  # Placeholder
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 1) -> Dict[str, List[float]]:
        """Train the model"""
        logger.info(f"Starting training for {epochs} epochs")
        
        history = {"train_loss": [], "val_loss": [], "val_bleu": []}
        
        # Training start callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start(self, self.model)
        
        for epoch in range(epochs):
            if self.should_stop:
                break
            
            self.current_epoch = epoch
            
            # Epoch start callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_start'):
                    callback.on_epoch_start(self, self.model, epoch)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_bleu"].append(val_metrics["val_bleu"])
            
            # Combine metrics
            all_metrics = {"train_loss": train_loss}
            all_metrics.update(val_metrics)
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{epochs} - " + 
                       ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()]))
            
            # Epoch end callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self, self.model, epoch, all_metrics)
            
            # Early stopping check
            if self.should_stop:
                logger.info(f"Training stopped at epoch {epoch + 1}")
                break
        
        # Training end callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end(self, self.model)
        
        logger.info("Training completed")
        return history
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        try:
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(filepath)
            else:
                paddle.save(self.model.state_dict(), filepath)
            
            logger.info(f"Model saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        try:
            if hasattr(self.model, 'from_pretrained'):
                self.model = self.model.from_pretrained(filepath)
            else:
                self.model.set_state_dict(paddle.load(filepath))
            
            if self.device == "gpu":
                self.model = self.model.cuda()
            
            logger.info(f"Model loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if not p.stop_gradient)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "learning_rate": self.learning_rate,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch
        }