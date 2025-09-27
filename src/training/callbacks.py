"""
Training callbacks for monitoring and control
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base training callback class"""
    
    def __init__(self, name: str = "TrainingCallback"):
        self.name = name
        self.training_start_time = None
        self.epoch_start_time = None
    
    def on_training_start(self, trainer, model):
        """Called when training starts"""
        self.training_start_time = time.time()
        logger.info(f"[{self.name}] Training started")
    
    def on_training_end(self, trainer, model):
        """Called when training ends"""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            logger.info(f"[{self.name}] Training completed in {total_time:.2f}s")
    
    def on_epoch_start(self, trainer, model, epoch: int):
        """Called when epoch starts"""
        self.epoch_start_time = time.time()
        logger.info(f"[{self.name}] Epoch {epoch} started")
    
    def on_epoch_end(self, trainer, model, epoch: int, metrics: Dict[str, float]):
        """Called when epoch ends"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            logger.info(f"[{self.name}] Epoch {epoch} completed in {epoch_time:.2f}s")
            if metrics:
                logger.info(f"[{self.name}] Epoch {epoch} metrics: {metrics}")


class ModelCheckpoint(TrainingCallback):
    """Model checkpoint callback"""
    
    def __init__(self, filepath: str, save_best_only: bool = True, 
                 monitor: str = "val_loss", mode: str = "min", save_freq: int = 1):
        super().__init__("ModelCheckpoint")
        self.filepath = Path(filepath)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self.best_metric = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer, model, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        super().on_epoch_end(trainer, model, epoch, metrics)
        
        if epoch % self.save_freq != 0:
            return
        
        if self.save_best_only:
            current_metric = metrics.get(self.monitor)
            if current_metric is None:
                logger.warning(f"[{self.name}] Metric {self.monitor} not found in metrics")
                return
            
            if self.best_metric is None:
                self.best_metric = current_metric
                should_save = True
            elif self.mode == "min" and current_metric < self.best_metric:
                self.best_metric = current_metric
                should_save = True
            elif self.mode == "max" and current_metric > self.best_metric:
                self.best_metric = current_metric
                should_save = True
            else:
                should_save = False
            
            if should_save:
                self._save_model(model, epoch, current_metric)
        else:
            self._save_model(model, epoch, metrics.get(self.monitor, 0.0))
    
    def _save_model(self, model, epoch: int, metric_value: float):
        """Save model to file"""
        try:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(self.filepath)
            elif hasattr(model, 'state_dict'):
                import torch
                torch.save(model.state_dict(), self.filepath)
            
            logger.info(f"[{self.name}] Model saved to {self.filepath} (epoch {epoch}, {self.monitor}={metric_value:.4f})")
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to save model: {str(e)}")


class EarlyStopping(TrainingCallback):
    """Early stopping callback"""
    
    def __init__(self, monitor: str = "val_loss", mode: str = "min", 
                 patience: int = 10, min_delta: float = 0.001):
        super().__init__("EarlyStopping")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer, model, epoch: int, metrics: Dict[str, float]):
        """Check if training should stop early"""
        super().on_epoch_end(trainer, model, epoch, metrics)
        
        current_metric = metrics.get(self.monitor)
        if current_metric is None:
            logger.warning(f"[{self.name}] Metric {self.monitor} not found in metrics")
            return
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return
        
        improved = False
        if self.mode == "min":
            improved = current_metric < (self.best_metric - self.min_delta)
        elif self.mode == "max":
            improved = current_metric > (self.best_metric + self.min_delta)
        
        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            logger.info(f"[{self.name}] Early stopping triggered at epoch {epoch}")


class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler_func: Callable, monitor: str = "val_loss", 
                 mode: str = "min", frequency: int = 1):
        super().__init__("LearningRateScheduler")
        self.scheduler_func = scheduler_func
        self.monitor = monitor
        self.mode = mode
        self.frequency = frequency
        self.step_count = 0
    
    def on_epoch_end(self, trainer, model, epoch: int, metrics: Dict[str, float]):
        """Update learning rate"""
        super().on_epoch_end(trainer, model, epoch, metrics)
        
        self.step_count += 1
        if self.step_count % self.frequency != 0:
            return
        
        current_metric = metrics.get(self.monitor)
        if current_metric is not None:
            try:
                self.scheduler_func(current_metric, epoch)
                logger.info(f"[{self.name}] Learning rate updated based on {self.monitor}={current_metric:.4f}")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to update learning rate: {str(e)}")


class MetricsLogger(TrainingCallback):
    """Metrics logging callback"""
    
    def __init__(self, log_file: str):
        super().__init__("MetricsLogger")
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
    
    def on_epoch_end(self, trainer, model, epoch: int, metrics: Dict[str, float]):
        """Log metrics to file"""
        super().on_epoch_end(trainer, model, epoch, metrics)
        
        epoch_metrics = {
            "epoch": epoch,
            "timestamp": time.time(),
            "metrics": metrics.copy()
        }
        
        self.metrics_history.append(epoch_metrics)
        
        try:
            import json
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{self.name}] Metrics logged to {self.log_file}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to log metrics: {str(e)}")
    
    def get_best_metric(self, metric_name: str, mode: str = "min") -> Optional[float]:
        """Get best value for a specific metric"""
        if not self.metrics_history:
            return None
        
        values = [
            epoch["metrics"].get(metric_name) 
            for epoch in self.metrics_history 
            if metric_name in epoch["metrics"]
        ]
        
        if not values:
            return None
        
        return min(values) if mode == "min" else max(values)