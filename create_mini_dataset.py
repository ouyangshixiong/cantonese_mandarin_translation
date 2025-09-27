#!/usr/bin/env python3
"""
Create mini dataset for testing and validation
"""
import json
import os
from pathlib import Path

# Sample Cantonese-Mandarin translation pairs
SAMPLE_PAIRS = [
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

def create_synthetic_data(base_pairs, num_samples):
    """Create synthetic data by adding variations"""
    synthetic_data = []
    for i in range(num_samples):
        base_pair = base_pairs[i % len(base_pairs)]
        
        # Add some variation
        variation = f" (變化 {i % 100})"
        synthetic_data.append({
            "cantonese": base_pair["cantonese"] + variation,
            "mandarin": base_pair["mandarin"] + f" (变化 {i % 100})"
        })
    return synthetic_data

def create_mini_dataset():
    """Create mini dataset for testing"""
    # Create data directories
    data_dirs = ["data/train", "data/val", "data/test", "data/mini"]
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    datasets = {
        "data/train/mini_train.jsonl": 10000,
        "data/val/mini_val.jsonl": 1000,
        "data/test/mini_test.jsonl": 1000,
        "data/mini/mini_sample.jsonl": 100
    }
    
    for filepath, num_samples in datasets.items():
        print(f"Creating {filepath} with {num_samples} samples...")
        
        synthetic_data = create_synthetic_data(SAMPLE_PAIRS, num_samples)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in synthetic_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Created {filepath}")
    
    # Create a small sample for quick testing
    sample_data = create_synthetic_data(SAMPLE_PAIRS, 50)
    with open('data/mini/sample_50.jsonl', 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("✅ Created data/mini/sample_50.jsonl")
    
    print("\n📊 Dataset Statistics:")
    for filepath, num_samples in datasets.items():
        print(f"  {filepath}: {num_samples} samples")
    print(f"  data/mini/sample_50.jsonl: 50 samples")
    
    print("\n🎉 Mini dataset creation completed!")

if __name__ == "__main__":
    create_mini_dataset()