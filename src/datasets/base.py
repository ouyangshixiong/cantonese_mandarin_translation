"""
Base dataset classes and utilities for Cantonese-Mandarin translation
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# Configure logging
logger = logging.getLogger(__name__)


class BaseDataset:
    """Base dataset class for Cantonese-Mandarin translation"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128, 
                 framework: str = "pytorch", mode: str = "train"):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.framework = framework
        self.mode = mode
        self.data = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from file"""
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            self.data = self._create_placeholder_data()
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.suffix == '.jsonl':
                    self.data = [json.loads(line) for line in f if line.strip()]
                elif self.data_path.suffix == '.json':
                    self.data = json.load(f)
                else:
                    # Assume TSV format
                    self.data = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            self.data.append({
                                'cantonese': parts[0],
                                'mandarin': parts[1]
                            })
            
            logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")
            
            # Apply data filtering
            self.data = self._filter_data(self.data)
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            self.data = self._create_placeholder_data()
    
    def _create_placeholder_data(self):
        """Create placeholder data for testing"""
        logger.info("Creating placeholder data for testing")
        return [
            {"cantonese": "你好嗎？", "mandarin": "你好吗？"},
            {"cantonese": "我今日好開心", "mandarin": "我今天很开心"},
            {"cantonese": "食咗飯未？", "mandarin": "吃饭了吗？"},
            {"cantonese": "呢度好靚", "mandarin": "这里很漂亮"},
            {"cantonese": "幾多錢？", "mandarin": "多少钱？"},
            {"cantonese": "我哋去邊度？", "mandarin": "我们去哪里？"},
            {"cantonese": "呢個好好食", "mandarin": "这个很好吃"},
            {"cantonese": "天氣好好", "mandarin": "天气很好"},
            {"cantonese": "多謝你", "mandarin": "谢谢你"},
            {"cantonese": "對唔住", "mandarin": "对不起"}
        ]
    
    def _filter_data(self, data: List[Dict]) -> List[Dict]:
        """Filter data based on quality criteria"""
        filtered = []
        
        for item in data:
            # Check if required fields exist
            if 'cantonese' not in item or 'mandarin' not in item:
                continue
            
            cantonese_text = item['cantonese'].strip()
            mandarin_text = item['mandarin'].strip()
            
            # Length filtering
            if len(cantonese_text) < 1 or len(mandarin_text) < 1:
                continue
            
            if len(cantonese_text) > self.max_length or len(mandarin_text) > self.max_length:
                continue
            
            # Length ratio filtering
            ratio = max(len(cantonese_text), len(mandarin_text)) / min(len(cantonese_text), len(mandarin_text))
            if ratio > 3.0:
                continue
            
            # Chinese character ratio check
            cantonese_chinese = sum(1 for c in cantonese_text if '\u4e00' <= c <= '\u9fff')
            mandarin_chinese = sum(1 for c in mandarin_text if '\u4e00' <= c <= '\u9fff')
            
            if cantonese_chinese / len(cantonese_text) < 0.5 or mandarin_chinese / len(mandarin_text) < 0.5:
                continue
            
            filtered.append({
                'cantonese': cantonese_text,
                'mandarin': mandarin_text
            })
        
        logger.info(f"Filtered to {len(filtered)} samples from {len(data)}")
        return filtered
    
    def preprocess_cantonese(self, text: str) -> str:
        """Preprocess Cantonese text"""
        # Cantonese to Mandarin mapping
        cantonese_map = {
            '嘅': '的', '咗': '了', '咁': '这么', '噉': '这样',
            '啲': '些', '哋': '们', '乜': '什么', '冇': '没有',
            '呢': '这', '嗰': '那', '喺': '在', '瞓': '睡',
            '飲': '喝', '食': '吃', '行': '走', '返': '回'
        }
        
        for cantonese, mandarin in cantonese_map.items():
            text = text.replace(cantonese, mandarin)
        
        return text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        raise NotImplementedError("Subclasses must implement __getitem__")