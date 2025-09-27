"""
PaddlePaddle base model components
"""
import paddle
import paddle.nn as nn
from typing import Dict, Any, List, Optional
from ..base import BaseTranslationModel, calculate_bleu_score


class PaddleBaseModel(BaseTranslationModel, nn.Layer):
    """Base PaddlePaddle model for translation"""
    
    def __init__(self, model_name: str, max_length: int = 128):
        BaseTranslationModel.__init__(self, model_name, max_length)
        nn.Layer.__init__(self)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if not p.stop_gradient)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "framework": "paddle",
            "device": "gpu" if paddle.is_compiled_with_cuda() else "cpu",
            "max_length": self.max_length
        }
    
    def training_step(self, batch: Dict[str, paddle.Tensor]) -> paddle.Tensor:
        """Training step"""
        outputs = self(**batch)
        return outputs.loss
    
    def validation_step(self, batch: Dict[str, paddle.Tensor]) -> Dict[str, float]:
        """Validation step"""
        # Generate translations
        generated_ids = self.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Decode predictions and references
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        
        # Calculate BLEU score
        bleu_score = calculate_bleu_score(preds, targets)
        
        return {
            "val_bleu": bleu_score,
            "preds": preds[:3],  # Store first 3 for logging
            "targets": targets[:3]
        }
    
    def predict_step(self, batch: Dict[str, paddle.Tensor]) -> List[str]:
        """Prediction step"""
        generated_ids = self.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)