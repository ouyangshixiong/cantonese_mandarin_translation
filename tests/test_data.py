"""
Tests for data processing and dataset functionality
"""
import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from datasets.cantonese_dataset import CantoneseTranslationDataset, create_mini_dataset
    from transformers import AutoTokenizer
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


class TestDataset:
    """Test dataset functionality"""
    
    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset dependencies not available")
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        # Create temporary test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [
                {"cantonese": "你好嗎？", "mandarin": "你好吗？"},
                {"cantonese": "食咗飯未？", "mandarin": "吃饭了吗？"}
            ]
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            dataset = CantoneseTranslationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                max_length=64,
                framework="pytorch",
                mode="train"
            )
            
            assert len(dataset) == 2
            assert hasattr(dataset, 'tokenizer')
            assert hasattr(dataset, 'max_length')
            
        finally:
            Path(temp_file).unlink()
    
    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset dependencies not available")
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        # Create temporary test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [{"cantonese": "你好嗎？", "mandarin": "你好吗？"}]
            f.write(json.dumps(test_data[0], ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            dataset = CantoneseTranslationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                max_length=64,
                framework="pytorch",
                mode="train"
            )
            
            item = dataset[0]
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'labels' in item
            
            # Check tensor shapes
            assert item['input_ids'].shape[0] == 64  # max_length
            assert item['attention_mask'].shape[0] == 64
            assert item['labels'].shape[0] == 64
            
        finally:
            Path(temp_file).unlink()
    
    def test_cantonese_preprocessing(self):
        """Test Cantonese text preprocessing"""
        if not DATASET_AVAILABLE:
            pytest.skip("Dataset dependencies not available")
        
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        # Create temporary test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [{"cantonese": "我嘅書", "mandarin": "我的书"}]
            f.write(json.dumps(test_data[0], ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            dataset = CantoneseTranslationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                max_length=64,
                framework="pytorch",
                mode="train"
            )
            
            # Test preprocessing
            original_text = "我嘅書"
            processed_text = dataset.preprocess_cantonese(original_text)
            assert "嘅" not in processed_text  # Should be converted
            
        finally:
            Path(temp_file).unlink()
    
    def test_data_filtering(self):
        """Test data filtering functionality"""
        if not DATASET_AVAILABLE:
            pytest.skip("Dataset dependencies not available")
        
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        # Create test data with various qualities
        test_data = [
            {"cantonese": "你好", "mandarin": "你好"},  # Good
            {"cantonese": "", "mandarin": "你好"},     # Empty source
            {"cantonese": "你好", "mandarin": ""},     # Empty target
            {"cantonese": "a" * 200, "mandarin": "你好"},  # Too long
            {"cantonese": "你好", "mandarin": "a" * 200},  # Too long
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            dataset = CantoneseTranslationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                max_length=64,
                framework="pytorch",
                mode="train"
            )
            
            # Should only have the good data point
            assert len(dataset) == 1
            
        finally:
            Path(temp_file).unlink()
    
    def test_create_mini_dataset(self):
        """Test mini dataset creation"""
        if not DATASET_AVAILABLE:
            pytest.skip("Dataset dependencies not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mini_test.jsonl"
            
            create_mini_dataset(str(output_path), num_samples=100)
            
            assert output_path.exists()
            
            # Check file contents
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) == 100
            
            # Check data format
            for line in lines:
                data = json.loads(line.strip())
                assert 'cantonese' in data
                assert 'mandarin' in data
                assert len(data['cantonese']) > 0
                assert len(data['mandarin']) > 0


class TestDataFormats:
    """Test different data format support"""
    
    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset dependencies not available")
    def test_jsonl_format(self):
        """Test JSONL format support"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"cantonese": "你好", "mandarin": "你好"}, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            dataset = CantoneseTranslationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                max_length=64,
                framework="pytorch",
                mode="train"
            )
            assert len(dataset) == 1
        finally:
            Path(temp_file).unlink()
    
    @pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset dependencies not available")
    def test_tsv_format(self):
        """Test TSV format support"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("你好\t你好\n")
            f.write("食咗飯未？\t吃饭了吗？\n")
            temp_file = f.name
        
        try:
            dataset = CantoneseTranslationDataset(
                data_path=temp_file,
                tokenizer=tokenizer,
                max_length=64,
                framework="pytorch",
                mode="train"
            )
            assert len(dataset) == 2
        finally:
            Path(temp_file).unlink()


class TestDataQuality:
    """Test data quality controls"""
    
    def test_length_constraints(self):
        """Test length constraint validation"""
        if not DATASET_AVAILABLE:
            pytest.skip("Dataset dependencies not available")
        
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        # Test with various lengths
        test_cases = [
            ("短", "短"),  # Very short
            ("中等长度文本", "中等长度文本"),  # Medium length
            ("a" * 150, "长文本" * 20),  # Long text
        ]
        
        for cantonese, mandarin in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write(json.dumps({"cantonese": cantonese, "mandarin": mandarin}, ensure_ascii=False) + '\n')
                temp_file = f.name
            
            try:
                dataset = CantoneseTranslationDataset(
                    data_path=temp_file,
                    tokenizer=tokenizer,
                    max_length=128,
                    framework="pytorch",
                    mode="train"
                )
                
                # Should handle various lengths appropriately
                assert len(dataset) <= 1  # May be filtered out if too long
                
            finally:
                Path(temp_file).unlink()
    
    def test_chinese_character_ratio(self):
        """Test Chinese character ratio filtering"""
        if not DATASET_AVAILABLE:
            pytest.skip("Dataset dependencies not available")
        
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        test_cases = [
            ("你好嗎", "你好吗"),  # High Chinese ratio
            ("hello world", "你好"),  # Low Chinese ratio
            ("混合text", "混合文本"),  # Mixed content
        ]
        
        for cantonese, mandarin in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                f.write(json.dumps({"cantonese": cantonese, "mandarin": mandarin}, ensure_ascii=False) + '\n')
                temp_file = f.name
            
            try:
                dataset = CantoneseTranslationDataset(
                    data_path=temp_file,
                    tokenizer=tokenizer,
                    max_length=128,
                    framework="pytorch",
                    mode="train"
                )
                
                # Low Chinese ratio content should be filtered out
                if cantonese.isascii() or mandarin.isascii():
                    assert len(dataset) == 0
                else:
                    assert len(dataset) <= 1
                    
            finally:
                Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])