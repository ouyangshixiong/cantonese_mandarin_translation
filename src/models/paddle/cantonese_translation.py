"""
PaddlePaddle implementation for Cantonese-Mandarin translation using Hunyuan-MT-7B
Optimized for ≤150 lines and high GPU utilization
"""
import paddle
import paddlenlp
from paddlenlp.transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from paddle.io import DataLoader, Dataset
from paddlenlp.metrics import BLEU
import logging

from .base import PaddleBaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CantoneseTranslationModel(PaddleBaseModel):
    """PaddlePaddle model for Cantonese-Mandarin translation with Hunyuan-MT-7B"""
    
    def __init__(self, model_name="hunyuan-mt-7b", max_length=128):
        super().__init__(model_name, max_length)
        self._load_model()
        
        # BLEU metric
        self.bleu_metric = BLEU()
    
    def _load_model(self):
        """Load model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        logger.info(f"Loaded PaddlePaddle model: {self.model_name}")


class CantoneseTranslationDataset(Dataset):
    """Dataset for Cantonese-Mandarin translation"""
    
    def __init__(self, tokenizer, max_length=128, mode="train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data = self._load_placeholder_data()
    
    def _load_placeholder_data(self):
        """Load placeholder data for testing"""
        return [
            {"source": "你好嗎？", "target": "你好吗？"},
            {"source": "我今日好開心", "target": "我今天很开心"},
            {"source": "食咗飯未？", "target": "吃饭了吗？"}
        ]
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = self.data[idx]
        
        # Tokenize source and target
        source_encoding = self.tokenizer(
            item["source"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pd"
        )
        
        target_encoding = self.tokenizer(
            item["target"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pd"
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data)


class CantoneseTrainer:
    """Trainer class for PaddlePaddle model"""
    
    def __init__(self, model, learning_rate=2e-5, warmup_steps=500, weight_decay=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
    
    def _create_optimizer(self):
        """Create AdamW optimizer"""
        return paddle.optimizer.AdamW(
            learning_rate=self.learning_rate,
            parameters=self.model.parameters(),
            weight_decay=self.weight_decay
        )
    
    def _create_lr_scheduler(self):
        """Create linear warmup scheduler"""
        return paddlenlp.transformers.LinearDecayWithWarmup(
            learning_rate=self.learning_rate,
            total_steps=1000,
            warmup=self.warmup_steps
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            loss = self.model.training_step(batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.clear_grad()
            
            total_loss += float(loss)
            
            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {float(loss):.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_bleu = 0
        
        with paddle.no_grad():
            for batch in dataloader:
                result = self.model.validation_step(batch)
                total_bleu += result["val_bleu"]
        
        avg_bleu = total_bleu / len(dataloader)
        logger.info(f"Validation BLEU: {avg_bleu:.4f}")
        
        return avg_bleu


def create_paddle_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    """Create PaddlePaddle DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        return_list=True
    )


if __name__ == "__main__":
    # Quick test and validation
    logger.info("Testing PaddlePaddle Cantonese Translation Model...")
    
    model = CantoneseTranslationModel()
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    dataset = CantoneseTranslationDataset(model.tokenizer)
    dataloader = create_paddle_dataloader(dataset, batch_size=2)
    
    trainer = CantoneseTrainer(model)
    
    logger.info("Setup complete. Ready for training/inference.")