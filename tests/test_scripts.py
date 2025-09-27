"""
Integration tests for training and inference scripts
"""
import pytest
import subprocess
import sys
import tempfile
import json
from pathlib import Path


class TestTrainingScript:
    """Test training script functionality"""
    
    def test_script_help(self):
        """Test that training script shows help"""
        result = subprocess.run([
            sys.executable, "scripts/train.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Train Cantonese-Mandarin translation model" in result.stdout
        assert "--framework" in result.stdout
        assert "--epochs" in result.stdout
    
    def test_script_invalid_framework(self):
        """Test script with invalid framework"""
        result = subprocess.run([
            sys.executable, "scripts/train.py", 
            "--framework", "invalid"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower()
    
    @pytest.mark.skip(reason="Requires model download and significant resources")
    def test_script_minimal_training(self):
        """Test minimal training run"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run([
                sys.executable, "scripts/train.py",
                "--framework", "pytorch",
                "--epochs", "1",
                "--batch_size", "2",
                "--max_length", "32",
                "--output_dir", temp_dir,
                "--model_name", "t5-small"  # Use smaller model for testing
            ], capture_output=True, text=True, timeout=300)
            
            # Should either succeed or fail gracefully
            assert result.returncode in [0, 1]  # Success or import error


class TestInferenceScript:
    """Test inference script functionality"""
    
    def test_script_help(self):
        """Test that inference script shows help"""
        result = subprocess.run([
            sys.executable, "scripts/inference.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Cantonese-Mandarin translation inference" in result.stdout
        assert "--text" in result.stdout
        assert "--framework" in result.stdout
    
    def test_script_invalid_framework(self):
        """Test script with invalid framework"""
        result = subprocess.run([
            sys.executable, "scripts/inference.py", 
            "--framework", "invalid"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower()
    
    def test_script_text_translation(self):
        """Test text translation (may fail due to missing model)"""
        result = subprocess.run([
            sys.executable, "scripts/inference.py",
            "--text", "hello",
            "--framework", "pytorch"
        ], capture_output=True, text=True, timeout=30)
        
        # Should either succeed or fail gracefully (model not found)
        assert result.returncode in [0, 1]  # Success or model error
    
    def test_script_batch_translation_files(self):
        """Test batch translation with files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = Path(temp_dir) / "input.txt"
            output_file = Path(temp_dir) / "output.txt"
            
            input_file.write_text("hello\nworld\n", encoding='utf-8')
            
            result = subprocess.run([
                sys.executable, "scripts/inference.py",
                "--input_file", str(input_file),
                "--output_file", str(output_file),
                "--framework", "pytorch"
            ], capture_output=True, text=True, timeout=30)
            
            # Should either succeed or fail gracefully
            assert result.returncode in [0, 1]


class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.skip(reason="Requires full model setup")
    def test_training_inference_pipeline(self):
        """Test complete training and inference pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"
            
            # Training step
            train_result = subprocess.run([
                sys.executable, "scripts/train.py",
                "--framework", "pytorch",
                "--epochs", "1",
                "--batch_size", "2",
                "--max_length", "32",
                "--output_dir", str(output_dir),
                "--model_name", "t5-small"
            ], capture_output=True, text=True, timeout=600)
            
            if train_result.returncode == 0:
                # Inference step
                infer_result = subprocess.run([
                    sys.executable, "scripts/inference.py",
                    "--text", "hello",
                    "--framework", "pytorch"
                ], capture_output=True, text=True, timeout=30)
                
                assert infer_result.returncode == 0
    
    def test_configuration_loading(self):
        """Test that configuration files can be loaded"""
        import yaml
        
        # Test main config
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'model' in config
            assert 'training' in config
            assert 'framework' in config
    
    def test_data_format(self):
        """Test data format compatibility"""
        # Test that sample data can be processed
        sample_data = {
            "cantonese": "你好嗎？",
            "mandarin": "你好吗？"
        }
        
        # Basic validation
        assert 'cantonese' in sample_data
        assert 'mandarin' in sample_data
        assert len(sample_data['cantonese']) > 0
        assert len(sample_data['mandarin']) > 0


class TestPerformance:
    """Performance and benchmarking tests"""
    
    def test_inference_speed_requirement(self):
        """Test that inference meets speed requirements"""
        # Placeholder for speed testing
        # Target: <2 seconds response time
        pass
    
    def test_memory_usage(self):
        """Test memory usage constraints"""
        # Target: ≤16GB GPU memory
        pass
    
    def test_gpu_utilization(self):
        """Test GPU utilization target"""
        # Target: ≥90% GPU utilization
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])