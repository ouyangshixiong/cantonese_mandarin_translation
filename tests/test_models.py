"""
Unit tests for model implementations
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from models.pytorch.cantonese_translation import CantoneseTranslationModule, CantoneseDataModule
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import paddle
    from models.paddle.cantonese_translation import CantoneseTranslationModel, CantoneseTranslationDataset
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class TestPyTorchModels:
    """Test PyTorch model implementations"""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_initialization(self):
        """Test model initialization"""
        model = CantoneseTranslationModule(
            model_name="Tencent-Hunyuan/Hunyuan-MT-7B",
            max_length=64  # Reduced for testing
        )
        
        # Check model properties
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'model')
        assert hasattr(model, 'bleu')
        assert model.hparams.max_length == 64
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_forward_pass(self):
        """Test forward pass"""
        model = CantoneseTranslationModule(max_length=64)
        
        # Create dummy input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Test forward pass
        outputs = model(input_ids, attention_mask, labels)
        assert hasattr(outputs, 'loss')
        assert outputs.loss.dim() == 0  # Scalar loss
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_parameters(self):
        """Test model parameter count"""
        model = CantoneseTranslationModule()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_data_module(self):
        """Test data module"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("Tencent-Hunyuan/Hunyuan-MT-7B")
        data_module = CantoneseDataModule(
            tokenizer=tokenizer,
            batch_size=2,
            max_length=64
        )
        
        assert hasattr(data_module, 'train_dataloader')
        assert hasattr(data_module, 'val_dataloader')
        assert data_module.batch_size == 2


class TestPaddleModels:
    """Test PaddlePaddle model implementations"""
    
    @pytest.mark.skipif(not PADDLE_AVAILABLE, reason="PaddlePaddle not available")
    def test_model_initialization(self):
        """Test model initialization"""
        model = CantoneseTranslationModel(
            model_name="hunyuan-mt-7b",
            max_length=64
        )
        
        # Check model properties
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'model')
        assert hasattr(model, 'bleu_metric')
        assert model.max_length == 64
    
    @pytest.mark.skipif(not PADDLE_AVAILABLE, reason="PaddlePaddle not available")
    def test_forward_pass(self):
        """Test forward pass"""
        model = CantoneseTranslationModel(max_length=64)
        
        # Create dummy input
        batch_size = 2
        seq_len = 32
        input_ids = paddle.randint(0, 1000, (batch_size, seq_len))
        attention_mask = paddle.ones((batch_size, seq_len))
        labels = paddle.randint(0, 1000, (batch_size, seq_len))
        
        # Test forward pass
        outputs = model(input_ids, attention_mask, labels)
        assert hasattr(outputs, 'loss')
    
    @pytest.mark.skipif(not PADDLE_AVAILABLE, reason="PaddlePaddle not available")
    def test_model_parameters(self):
        """Test model parameter count"""
        model = CantoneseTranslationModel()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    @pytest.mark.skipif(not PADDLE_AVAILABLE, reason="PaddlePaddle not available")
    def test_dataset(self):
        """Test dataset"""
        model = CantoneseTranslationModel(max_length=64)
        dataset = CantoneseTranslationDataset(
            tokenizer=model.tokenizer,
            max_length=64,
            mode="train"
        )
        
        assert len(dataset) > 0
        
        # Test getting an item
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item


class TestModelCompatibility:
    """Test compatibility between frameworks"""
    
    @pytest.mark.skipif(not (PYTORCH_AVAILABLE and PADDLE_AVAILABLE), 
                       reason="Both frameworks not available")
    def test_output_consistency(self):
        """Test that both models produce similar output shapes"""
        # This is a placeholder for more comprehensive consistency tests
        assert PYTORCH_AVAILABLE and PADDLE_AVAILABLE
    
    def test_api_consistency(self):
        """Test that both frameworks have consistent APIs"""
        if PYTORCH_AVAILABLE:
            from models.pytorch.cantonese_translation import CantoneseTranslationModule
            pytorch_model = CantoneseTranslationModule(max_length=64)
            assert hasattr(pytorch_model, 'forward')
            assert hasattr(pytorch_model, 'training_step')
        
        if PADDLE_AVAILABLE:
            from models.paddle.cantonese_translation import CantoneseTranslationModel
            paddle_model = CantoneseTranslationModel(max_length=64)
            assert hasattr(paddle_model, 'forward')
            assert hasattr(paddle_model, 'training_step')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])