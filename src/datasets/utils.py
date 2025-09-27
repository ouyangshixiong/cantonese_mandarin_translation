"""
Dataset utilities and data generation functions
"""
import json
import logging
from pathlib import Path
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)


def create_mini_dataset(output_path: str, num_samples: int = 10000):
    """Create mini dataset for quick validation"""
    logger.info(f"Creating mini dataset with {num_samples} samples")
    
    # Sample Cantonese-Mandarin pairs
    sample_pairs = [
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
    
    # Generate synthetic data by varying the samples
    synthetic_data = []
    for i in range(num_samples):
        base_pair = sample_pairs[i % len(sample_pairs)]
        
        # Add some variation
        variation = f" (變化 {i % 100})"
        synthetic_data.append({
            "cantonese": base_pair["cantonese"] + variation,
            "mandarin": base_pair["mandarin"] + f" (变化 {i % 100})"
        })
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in synthetic_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Mini dataset saved to {output_path}")


def create_sample_data_splits(base_path: str = "data"):
    """Create sample data splits for training, validation, and testing"""
    base_path = Path(base_path)
    
    # Create directories
    (base_path / "train").mkdir(parents=True, exist_ok=True)
    (base_path / "val").mkdir(parents=True, exist_ok=True)
    (base_path / "test").mkdir(parents=True, exist_ok=True)
    
    # Create mini datasets
    create_mini_dataset(str(base_path / "train" / "mini_train.jsonl"), 10000)
    create_mini_dataset(str(base_path / "val" / "mini_val.jsonl"), 1000)
    create_mini_dataset(str(base_path / "test" / "mini_test.jsonl"), 1000)
    
    logger.info("Sample data splits created successfully")


def validate_dataset(file_path: str, max_samples: int = 10) -> Dict:
    """Validate dataset file and return statistics"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        samples = []
        total_samples = 0
        valid_samples = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Check required fields
                    if 'cantonese' not in data or 'mandarin' not in data:
                        continue
                    
                    cantonese_text = data['cantonese'].strip()
                    mandarin_text = data['mandarin'].strip()
                    
                    if not cantonese_text or not mandarin_text:
                        continue
                    
                    valid_samples += 1
                    
                    # Collect sample data
                    if len(samples) < max_samples:
                        samples.append({
                            "cantonese": cantonese_text,
                            "mandarin": mandarin_text,
                            "length_ratio": len(mandarin_text) / len(cantonese_text) if len(cantonese_text) > 0 else 0
                        })
                    
                    total_samples += 1
                    
                except json.JSONDecodeError:
                    continue
        
        return {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "validity_rate": valid_samples / total_samples if total_samples > 0 else 0,
            "sample_data": samples,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        return {"error": f"Failed to validate dataset: {str(e)}"}