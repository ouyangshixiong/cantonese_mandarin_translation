"""
Core translator class for Cantonese-Mandarin translation
Supports both PyTorch and PaddlePaddle frameworks
"""
import logging
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logger = logging.getLogger(__name__)

# Try to import both frameworks
PYTORCH_AVAILABLE = False
PADDLE_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from models.pytorch.cantonese_translation import CantoneseTranslationModule
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available")

try:
    import paddle
    from models.paddle.cantonese_translation import CantoneseTranslationModel
    PADDLE_AVAILABLE = True
except ImportError:
    logger.warning("PaddlePaddle not available")


class CantoneseTranslator:
    """Unified translator class supporting both frameworks"""
    
    def __init__(self, model_name: str = "Tencent-Hunyuan/Hunyuan-MT-7B", 
                 framework: str = "pytorch", device: str = "auto"):
        self.model_name = model_name
        self.framework = framework
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.max_length = 128
        self.is_causal_lm = False  # Will be set during model loading
        
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            if self.framework == "pytorch" and torch.cuda.is_available():
                return "cuda"
            elif self.framework == "paddle" and paddle.is_compiled_with_cuda():
                return "gpu"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading {self.framework} model: {self.model_name}")
        
        if self.framework == "pytorch" and PYTORCH_AVAILABLE:
            self._load_pytorch_model()
        elif self.framework == "paddle" and PADDLE_AVAILABLE:
            self._load_paddle_model()
        else:
            raise RuntimeError(f"Framework {self.framework} not available")
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        # Check if this is an FP8 model
        is_fp8_model = "fp8" in self.model_name.lower()
        
        # Use local cache path for ModelScope models
        if is_fp8_model:
            local_model_path = f"/home/ouyang/.cache/modelscope/hub/models/{self.model_name.replace('/', '/')}/"
            if not os.path.exists(local_model_path):
                local_model_path = self.model_name  # Fallback to original name
        else:
            local_model_path = self.model_name
        
        if is_fp8_model:
            logger.info("Detected FP8 quantized model, using CausalLM architecture")
            logger.info(f"Loading from: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                device_map='auto'
            )
            self.is_causal_lm = True
        else:
            logger.info("Using standard Seq2Seq model")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
            self.is_causal_lm = False
            
            if self.device == "cuda":
                self.model = self.model.cuda()
        
        self.model.eval()
    
    def _load_paddle_model(self):
        """Load PaddlePaddle model"""
        self.model = CantoneseTranslationModel(model_name=self.model_name)
        self.tokenizer = self.model.tokenizer
        
        if self.device == "gpu":
            self.model = self.model.cuda()
        
        self.model.eval()
    
    def translate(self, text: str, source_lang: str = "cantonese", 
                  target_lang: str = "mandarin", beam_size: int = 4) -> str:
        """Translate text from source to target language"""
        start_time = time.time()
        
        try:
            if self.framework == "pytorch":
                result = self._translate_pytorch(text, source_lang, target_lang, beam_size)
            else:
                result = self._translate_paddle(text, source_lang, target_lang, beam_size)
            
            inference_time = time.time() - start_time
            logger.info(f"Translation completed in {inference_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise
    
    def _translate_pytorch(self, text: str, source_lang: str, 
                          target_lang: str, beam_size: int) -> str:
        """PyTorch translation implementation"""
        if self.is_causal_lm:
            return self._translate_causal_lm(text, source_lang, target_lang, beam_size)
        else:
            return self._translate_seq2seq(text, source_lang, target_lang, beam_size)
    
    def _translate_causal_lm(self, text: str, source_lang: str, 
                           target_lang: str, beam_size: int) -> str:
        """FP8 CausalLM translation implementation using chat template"""
        # Create translation prompt
        prompt = f"Translate from {source_lang} to {target_lang}: {text}"
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        # Apply chat template
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors='pt'
        )
        
        # Generate translation using recommended parameters
        with torch.no_grad():
            outputs = self.model.generate(
                tokenized_chat.to(self.model.device), 
                max_new_tokens=50,
                top_k=20,
                top_p=0.6,
                repetition_penalty=1.05,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode result
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the translation part (remove the prompt)
        if prompt in result:
            result = result.replace(prompt, "").strip()
        
        return result
    
    def _translate_seq2seq(self, text: str, source_lang: str, 
                          target_lang: str, beam_size: int) -> str:
        """Standard Seq2Seq translation implementation"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=beam_size,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
        
        # Decode result
        result = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return result
    
    def _translate_paddle(self, text: str, source_lang: str, 
                         target_lang: str, beam_size: int) -> str:
        """PaddlePaddle translation implementation"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pd"
        )
        
        if self.device == "gpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        with paddle.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=beam_size,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
        
        # Decode result
        result = self.tokenizer.decode(
            generated_ids[0][0],  # PaddlePaddle format difference
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return result
    
    def batch_translate(self, texts: List[str], source_lang: str = "cantonese",
                       target_lang: str = "mandarin", beam_size: int = 4) -> List[str]:
        """Batch translate multiple texts"""
        results = []
        
        for text in texts:
            try:
                translation = self.translate(text, source_lang, target_lang, beam_size)
                results.append(translation)
            except Exception as e:
                logger.error(f"Failed to translate '{text}': {str(e)}")
                results.append("")  # Return empty string for failed translations
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if 
                              (hasattr(p, 'requires_grad') and p.requires_grad) or
                              (hasattr(p, 'stop_gradient') and not p.stop_gradient))
        
        return {
            "model_name": self.model_name,
            "framework": self.framework,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "max_length": self.max_length
        }