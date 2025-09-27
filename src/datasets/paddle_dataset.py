"""
PaddlePaddle-specific dataset implementations
"""
import paddle
from paddle.io import Dataset, DataLoader
from typing import Dict, Optional, Callable
from .base import BaseDataset


class CantoneseTranslationDatasetPaddle(BaseDataset, Dataset):
    """PaddlePaddle dataset class for Cantonese-Mandarin translation"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128, mode: str = "train"):
        super().__init__(data_path, tokenizer, max_length, "paddle", mode)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = self.data[idx]
        
        # Preprocess Cantonese if needed
        source_text = self.preprocess_cantonese(item['cantonese'])
        target_text = item['mandarin']
        
        # Tokenize
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pd"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pd"
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0)
        }


class CantoneseTranslationDataModulePaddle:
    """Data module for PaddlePaddle"""
    
    def __init__(self, tokenizer, data_config: Dict, batch_size: int = 8, 
                 max_length: int = 128, num_workers: int = 4):
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = CantoneseTranslationDatasetPaddle(
                self.data_config['train_path'],
                self.tokenizer,
                self.max_length,
                "train"
            )
            
            self.val_dataset = CantoneseTranslationDatasetPaddle(
                self.data_config['val_path'],
                self.tokenizer,
                self.max_length,
                "val"
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = CantoneseTranslationDatasetPaddle(
                self.data_config['test_path'],
                self.tokenizer,
                self.max_length,
                "test"
            )
    
    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            return_list=True
        )
    
    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            return_list=True
        )
    
    def test_dataloader(self):
        """Test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            return_list=True
        )