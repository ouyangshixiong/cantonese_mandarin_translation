"""
Test suite for XTTS-v2 integration
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tts import XTTSV2Engine
from omegaconf import OmegaConf


class TestXTTSV2Engine:
    """Test cases for XTTS-v2 engine."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = OmegaConf.create({
            'tts': {
                'sample_rate': 22050,
                'hop_length': 256,
                'win_length': 1024,
                'n_mels': 80,
                'n_fft': 1024,
                'fmin': 80,
                'fmax': 8000,
                'model_paths': {
                    'yue': './models/tts/index_tts2_cantonese.pth',
                    'cmn': './models/tts/index_tts2_mandarin.pth'
                },
                'vocoder_paths': {
                    'yue': './models/tts/vocoder_cantonese.pth',
                    'cmn': './models/tts/vocoder_mandarin.pth'
                },
                'speaker_embedding_dim': 256,
                'default_speaker_yue': 'yue_female_001',
                'default_speaker_cmn': 'cmn_female_001'
            }
        })
        return config
    
    def test_engine_initialization(self, mock_config):
        """Test XTTS-v2 engine initialization."""
        # Mock TTS library availability
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                assert engine.sample_rate == 22050
                assert 'yue' in engine.languages
                assert 'cmn' in engine.languages
                assert 'en' in engine.languages  # English support
                assert engine.device.type == "cpu"
    
    def test_synthesize_basic(self, mock_config):
        """Test basic speech synthesis."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_instance = Mock()
                mock_instance.tts_to_file = Mock()
                mock_tts.return_value = mock_instance
                
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                # Mock audio loading
                with patch.object(engine, '_load_audio_file', return_value=(np.random.randn(1000), 22050)):
                    # Test synthesis
                    text = "你好嗎？"
                    audio_array, sample_rate = engine.synthesize(
                        text=text,
                        language="yue",
                        emotion="neutral",
                        speed=1.0
                    )
                    
                    assert isinstance(audio_array, np.ndarray)
                    assert sample_rate == 22050
                    assert len(audio_array) > 0
    
    def test_synthesize_different_languages(self, mock_config):
        """Test synthesis for different languages."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_instance = Mock()
                mock_instance.tts_to_file = Mock()
                mock_tts.return_value = mock_instance
                
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                # Mock audio loading
                with patch.object(engine, '_load_audio_file', return_value=(np.random.randn(1000), 22050)):
                    # Test Cantonese
                    text_yue = "你好嗎？"
                    audio_yue, sr_yue = engine.synthesize(text_yue, "yue")
                    assert isinstance(audio_yue, np.ndarray)
                    
                    # Test Mandarin
                    text_cmn = "你好吗？"
                    audio_cmn, sr_cmn = engine.synthesize(text_cmn, "cmn")
                    assert isinstance(audio_cmn, np.ndarray)
                    
                    # Test English
                    text_en = "Hello, how are you?"
                    audio_en, sr_en = engine.synthesize(text_en, "en")
                    assert isinstance(audio_en, np.ndarray)
                    
                    # Same sample rate for all
                    assert sr_yue == sr_cmn == sr_en == 22050
    
    def test_synthesize_with_emotions(self, mock_config):
        """Test synthesis with different emotions."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_instance = Mock()
                mock_instance.tts_to_file = Mock()
                mock_tts.return_value = mock_instance
                
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                # Mock audio loading
                with patch.object(engine, '_load_audio_file', return_value=(np.random.randn(1000), 22050)):
                    text = "今天天氣很好"
                    emotions = ["neutral", "happy", "sad", "angry"]
                    
                    for emotion in emotions:
                        audio_array, sample_rate = engine.synthesize(
                            text=text,
                            language="yue",
                            emotion=emotion
                        )
                        
                        assert isinstance(audio_array, np.ndarray)
                        assert sample_rate == 22050
    
    def test_synthesize_with_speed_modification(self, mock_config):
        """Test synthesis with speed modification."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_instance = Mock()
                mock_instance.tts_to_file = Mock()
                mock_tts.return_value = mock_instance
                
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                # Mock audio loading
                with patch.object(engine, '_load_audio_file', return_value=(np.random.randn(1000), 22050)):
                    text = "這是一段測試文字"
                    
                    # Test different speeds
                    speeds = [0.5, 1.0, 1.5, 2.0]
                    
                    for speed in speeds:
                        audio_array, sample_rate = engine.synthesize(
                            text=text,
                            language="yue",
                            speed=speed
                        )
                        
                        assert isinstance(audio_array, np.ndarray)
                        assert sample_rate == 22050
    
    def test_invalid_language(self, mock_config):
        """Test handling of invalid language codes."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                with pytest.raises(ValueError):
                    engine.synthesize("test", "invalid_lang")
    
    def test_get_available_speakers(self, mock_config):
        """Test speaker listing functionality."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                speakers = engine.get_available_speakers("yue")
                assert isinstance(speakers, list)
                assert len(speakers) > 0
    
    def test_english_text_preprocessing(self, mock_config):
        """Test English text preprocessing functionality."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                # Test English text preprocessing
                test_cases = [
                    ("don't worry", "do not worry"),  # Contraction expansion
                    ("I'm happy", "i am happy"),      # Contraction expansion
                    ("Hello 123", "Hello one two three"),  # Number conversion
                    ("It's a test", "it is a test"),  # Multiple contractions
                ]
                
                for input_text, expected_output in test_cases:
                    processed = engine._process_english_text(input_text, is_multilingual=False)
                    # Check that contractions are expanded
                    assert "don't" not in processed.lower()
                    assert "i'm" not in processed.lower()
                    assert "it's" not in processed.lower()
    
    def test_get_supported_emotions(self, mock_config):
        """Test emotion listing functionality."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                emotions = engine.get_supported_emotions()
                assert isinstance(emotions, list)
                assert "neutral" in emotions
                assert "happy" in emotions
    
    def test_estimate_synthesis_time(self, mock_config):
        """Test synthesis time estimation."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                # Test different text lengths
                text_lengths = [10, 50, 100, 200]
                
                for length in text_lengths:
                    estimated_time = engine.estimate_synthesis_time(length)
                    
                    assert isinstance(estimated_time, float)
                    assert estimated_time > 0
                    assert estimated_time < 10  # Should be reasonable
    
    def test_memory_usage_reporting(self, mock_config):
        """Test memory usage reporting."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_tts.return_value = Mock()
                engine = XTTSV2Engine(mock_config, device="cpu")
                
                memory_stats = engine.get_memory_usage()
                assert isinstance(memory_stats, dict)
                
                # On CPU, should have empty GPU stats
                assert 'gpu_allocated' not in memory_stats or memory_stats['gpu_allocated'] == 0








class TestIntegration:
    """Integration tests for TTS module."""
    
    @pytest.fixture
    def full_config(self):
        """Create full configuration for integration testing."""
        config = OmegaConf.create({
            'model': {'name': 'test_model'},
            'framework': {'name': 'pytorch'},
            'tts': {
                'enabled': True,
                'sample_rate': 22050,
                'n_mels': 80,
                'hop_length': 256
            }
        })
        return config
    
    def test_end_to_end_synthesis(self, full_config):
        """Test end-to-end synthesis pipeline."""
        with patch('tts.xtts_v2_engine.XTTS_AVAILABLE', True):
            with patch('TTS.api.TTS') as mock_tts:
                mock_instance = Mock()
                mock_instance.tts_to_file = Mock()
                mock_tts.return_value = mock_instance
                
                # Initialize TTS engine
                tts_engine = XTTSV2Engine(full_config, device="cpu")
                
                # Mock audio loading
                with patch.object(tts_engine, '_load_audio_file', return_value=(np.random.randn(1000), 22050)):
                    # Test synthesis
                    text = "這是一個測試"
                    audio_array, sample_rate = tts_engine.synthesize(
                        text=text,
                        language="yue",
                        emotion="neutral"
                    )
                    
                    # Save audio using torchaudio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                        torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
                        
                        assert Path(tmp_file.name).exists()
                        
                        # Clean up
                        os.unlink(tmp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])