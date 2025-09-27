"""
Test suite for Index-TTS2 integration
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

from tts import IndexTTS2Engine, CrossLingualVoiceConverter, SpeakerManager, AudioProcessor
from omegaconf import OmegaConf


class TestIndexTTS2Engine:
    """Test cases for Index-TTS2 engine."""
    
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
        """Test Index-TTS2 engine initialization."""
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            assert engine.sample_rate == 22050
            assert engine.hop_length == 256
            assert 'yue' in engine.languages
            assert 'cmn' in engine.languages
            assert engine.device.type == "cpu"
    
    def test_synthesize_basic(self, mock_config):
        """Test basic speech synthesis."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
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
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            # Test Cantonese
            text_yue = "你好嗎？"
            audio_yue, sr_yue = engine.synthesize(text_yue, "yue")
            assert isinstance(audio_yue, np.ndarray)
            
            # Test Mandarin
            text_cmn = "你好吗？"
            audio_cmn, sr_cmn = engine.synthesize(text_cmn, "cmn")
            assert isinstance(audio_cmn, np.ndarray)
            
            # Same sample rate for both
            assert sr_yue == sr_cmn == 22050
    
    def test_synthesize_with_emotions(self, mock_config):
        """Test synthesis with different emotions."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
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
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            text = "這是一段測試文字"
            
            # Test different speeds
            speeds = [0.5, 1.0, 1.5, 2.0]
            audio_lengths = []
            
            for speed in speeds:
                audio_array, sample_rate = engine.synthesize(
                    text=text,
                    language="yue",
                    speed=speed
                )
                
                assert isinstance(audio_array, np.ndarray)
                audio_lengths.append(len(audio_array))
            
            # Verify that faster speed produces shorter audio
            # (This is a basic check - actual lengths may vary due to placeholder implementation)
            assert audio_lengths[0] >= audio_lengths[-1]  # 0.5x should be longer than 2.0x
    
    def test_invalid_language(self, mock_config):
        """Test handling of invalid language codes."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            with pytest.raises(ValueError):
                engine.synthesize("test", "invalid_lang")
    
    def test_get_available_speakers(self, mock_config):
        """Test speaker listing functionality."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            speakers = engine.get_available_speakers("yue")
            assert isinstance(speakers, list)
            assert len(speakers) > 0
    
    def test_get_supported_emotions(self, mock_config):
        """Test emotion listing functionality."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            emotions = engine.get_supported_emotions()
            assert isinstance(emotions, list)
            assert "neutral" in emotions
            assert "happy" in emotions
    
    def test_estimate_synthesis_time(self, mock_config):
        """Test synthesis time estimation."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            # Test different text lengths
            text_lengths = [10, 50, 100, 200]
            
            for length in text_lengths:
                text = "測試" * (length // 2)  # Create text of approximate length
                estimated_time = engine.estimate_synthesis_time(len(text))
                
                assert isinstance(estimated_time, float)
                assert estimated_time > 0
                assert estimated_time < 10  # Should be reasonable
    
    def test_memory_usage_reporting(self, mock_config):
        """Test memory usage reporting."""
        with patch('pathlib.Path.exists', return_value=True):
            engine = IndexTTS2Engine(mock_config, device="cpu")
            
            memory_stats = engine.get_memory_usage()
            assert isinstance(memory_stats, dict)
            
            # On CPU, should have empty GPU stats
            assert 'gpu_allocated' not in memory_stats or memory_stats['gpu_allocated'] == 0


class TestCrossLingualVoiceConverter:
    """Test cases for cross-lingual voice converter."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = OmegaConf.create({
            'voice_conversion': {
                'enabled': True,
                'preserve_speaker': True,
                'adapt_prosody': True,
                'cross_lingual': True
            }
        })
        return config
    
    def test_converter_initialization(self, mock_config):
        """Test voice converter initialization."""
        converter = CrossLingualVoiceConverter(mock_config, device="cpu")
        
        assert converter.device.type == "cpu"
        assert converter.content_encoder is not None
        assert converter.speaker_encoder is not None
        assert converter.decoder is not None
    
    def test_voice_conversion(self, mock_config):
        """Test basic voice conversion."""
        converter = CrossLingualVoiceConverter(mock_config, device="cpu")
        
        # Create dummy audio
        sample_rate = 22050
        duration = 2.0  # seconds
        audio = np.random.randn(int(sample_rate * duration)) * 0.1
        
        # Perform conversion
        converted_audio = converter.convert_voice(
            source_audio=audio,
            source_language="yue",
            target_language="cmn"
        )
        
        assert isinstance(converted_audio, np.ndarray)
        assert len(converted_audio) > 0
    
    def test_invalid_language_conversion(self, mock_config):
        """Test handling of invalid language codes."""
        converter = CrossLingualVoiceConverter(mock_config, device="cpu")
        
        audio = np.random.randn(1000) * 0.1
        
        with pytest.raises(ValueError):
            converter.convert_voice(audio, "invalid_lang", "yue")
        
        with pytest.raises(ValueError):
            converter.convert_voice(audio, "yue", "invalid_lang")
    
    def test_conversion_info(self, mock_config):
        """Test conversion information retrieval."""
        converter = CrossLingualVoiceConverter(mock_config, device="cpu")
        
        info = converter.get_conversion_info()
        
        assert isinstance(info, dict)
        assert 'supported_languages' in info
        assert 'yue' in info['supported_languages']
        assert 'cmn' in info['supported_languages']
    
    def test_estimate_conversion_time(self, mock_config):
        """Test conversion time estimation."""
        converter = CrossLingualVoiceConverter(mock_config, device="cpu")
        
        audio_lengths = [1.0, 2.0, 5.0, 10.0]  # seconds
        
        for length in audio_lengths:
            estimated_time = converter.estimate_conversion_time(length)
            
            assert isinstance(estimated_time, float)
            assert estimated_time > 0
            assert estimated_time > length * 0.1  # Should be longer than audio


class TestSpeakerManager:
    """Test cases for speaker manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = OmegaConf.create({
            'speaker': {
                'embedding_dim': 256,
                'min_audio_duration': 5.0,
                'max_audio_duration': 300.0,
                'voice_cloning_threshold': 0.85,
                'similarity_threshold': 0.7
            }
        })
        return config
    
    def test_speaker_manager_initialization(self, mock_config):
        """Test speaker manager initialization."""
        manager = SpeakerManager(mock_config, device="cpu")
        
        assert manager.device.type == "cpu"
        assert manager.embedding_dim == 256
        assert manager.speaker_encoder is not None
        assert manager.voice_cloner is not None
    
    def test_predefined_speakers_loading(self, mock_config):
        """Test loading of predefined speakers."""
        manager = SpeakerManager(mock_config, device="cpu")
        
        # Should have predefined speakers
        assert len(manager.speakers) > 0
        assert len(manager.speaker_embeddings) > 0
        
        # Check for expected speakers
        speaker_ids = list(manager.speakers.keys())
        assert any("yue" in sid for sid in speaker_ids)
        assert any("cmn" in sid for sid in speaker_ids)
    
    def test_list_speakers(self, mock_config):
        """Test speaker listing functionality."""
        manager = SpeakerManager(mock_config, device="cpu")
        
        # List all speakers
        all_speakers = manager.list_speakers()
        assert len(all_speakers) > 0
        
        # List by language
        yue_speakers = manager.list_speakers(language="yue")
        assert len(yue_speakers) > 0
        assert all(s.language == "yue" for s in yue_speakers)
        
        cmn_speakers = manager.list_speakers(language="cmn")
        assert len(cmn_speakers) > 0
        assert all(s.language == "cmn" for s in cmn_speakers)
    
    def test_speaker_comparison(self, mock_config):
        """Test speaker similarity comparison."""
        manager = SpeakerManager(mock_config, device="cpu")
        
        # Get speaker IDs
        speaker_ids = list(manager.speakers.keys())
        if len(speaker_ids) >= 2:
            sid1, sid2 = speaker_ids[0], speaker_ids[1]
            
            similarity = manager.compare_speakers(sid1, sid2)
            
            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1
    
    def test_find_similar_speakers(self, mock_config):
        """Test finding similar speakers."""
        manager = SpeakerManager(mock_config, device="cpu")
        
        speaker_ids = list(manager.speakers.keys())
        if speaker_ids:
            reference_id = speaker_ids[0]
            
            similar_speakers = manager.find_similar_speakers(reference_id, threshold=0.5)
            
            assert isinstance(similar_speakers, list)
            # Should find at least the speaker itself (but it's filtered out)
            # So we might get an empty list or other speakers
    
    def test_database_stats(self, mock_config):
        """Test database statistics."""
        manager = SpeakerManager(mock_config, device="cpu")
        
        stats = manager.get_database_stats()
        
        assert isinstance(stats, dict)
        assert 'total_speakers' in stats
        assert 'languages' in stats
        assert 'genders' in stats
        assert 'age_groups' in stats
        
        assert stats['total_speakers'] > 0


class TestAudioProcessor:
    """Test cases for audio processor."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor for testing."""
        return AudioProcessor(sample_rate=22050, n_mels=80, hop_length=256, win_length=1024)
    
    def test_audio_processor_initialization(self, audio_processor):
        """Test audio processor initialization."""
        assert audio_processor.sample_rate == 22050
        assert audio_processor.n_mels == 80
        assert audio_processor.hop_length == 256
        assert audio_processor.mel_transform is not None
    
    def test_sample_rate_conversion(self, audio_processor):
        """Test sample rate conversion."""
        # Create test audio at different sample rates
        orig_sr = 44100
        target_sr = 22050
        duration = 1.0
        
        audio = torch.randn(1, int(orig_sr * duration))
        
        converted_audio = audio_processor.convert_sample_rate(audio, orig_sr, target_sr)
        
        expected_length = int(target_sr * duration)
        assert converted_audio.shape[1] == expected_length
    
    def test_mono_conversion(self, audio_processor):
        """Test stereo to mono conversion."""
        # Create stereo audio
        stereo_audio = torch.randn(2, 1000)
        
        mono_audio = audio_processor.convert_to_mono(stereo_audio)
        
        assert mono_audio.shape[0] == 1
        assert mono_audio.shape[1] == 1000
    
    def test_audio_normalization(self, audio_processor):
        """Test audio normalization."""
        # Create audio with different amplitudes
        audio = torch.randn(1, 1000) * 0.5
        
        normalized_audio = audio_processor.normalize_audio(audio, target_db=-20.0)
        
        assert normalized_audio.shape == audio.shape
        # Should be normalized to target level
        rms = torch.sqrt(torch.mean(normalized_audio ** 2))
        target_rms = 10 ** (-20.0 / 20)
        assert abs(rms.item() - target_rms) < 0.1
    
    def test_audio_to_mel_conversion(self, audio_processor):
        """Test audio to mel-spectrogram conversion."""
        audio = torch.randn(1, 1000)
        
        mel_spec = audio_processor.audio_to_mel(audio)
        
        assert isinstance(mel_spec, torch.Tensor)
        assert mel_spec.shape[0] == 1  # Batch dimension
        assert mel_spec.shape[1] == 80  # Mel bins
        assert mel_spec.shape[2] > 0  # Time dimension
    
    def test_mel_to_audio_conversion(self, audio_processor):
        """Test mel-spectrogram to audio conversion."""
        # Create dummy mel-spectrogram
        mel_spec = torch.randn(1, 80, 10)  # Batch, mel_bins, time
        
        audio = audio_processor.mel_to_audio(mel_spec)
        
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] == 1  # Batch dimension
        assert audio.shape[1] > 0  # Audio samples
    
    def test_audio_quality_assessment(self, audio_processor):
        """Test audio quality assessment."""
        # Create test audio
        audio = torch.randn(1, 1000) * 0.1
        
        quality_metrics = audio_processor.assess_quality(audio)
        
        assert isinstance(quality_metrics, audio_processor.AudioQualityMetrics)
        assert hasattr(quality_metrics, 'snr_db')
        assert hasattr(quality_metrics, 'dynamic_range')
        assert hasattr(quality_metrics, 'clipping_ratio')
        assert hasattr(quality_metrics, 'overall_score')
    
    def test_audio_effects(self, audio_processor):
        """Test audio effects application."""
        audio = torch.randn(1, 1000)
        
        # Test speed change
        speed_changed = audio_processor.change_speed(audio, 1.5)
        assert speed_changed.shape == audio.shape
        
        # Test pitch change
        pitch_changed = audio_processor.change_pitch(audio, 2.0)
        assert pitch_changed.shape == audio.shape
    
    def test_audio_splitting(self, audio_processor):
        """Test audio segmentation."""
        # Create longer audio
        audio = torch.randn(1, 22050 * 3)  # 3 seconds at 22kHz
        
        segments = audio_processor.split_audio(audio, segment_length=1.0)
        
        assert isinstance(segments, list)
        assert len(segments) >= 2  # Should split into multiple segments
        assert all(isinstance(seg, torch.Tensor) for seg in segments)


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
                'hop_length': 256,
                'model_paths': {'yue': './test.pth', 'cmn': './test.pth'},
                'vocoder_paths': {'yue': './test_vocoder.pth', 'cmn': './test_vocoder.pth'},
                'default_speaker_yue': 'yue_test',
                'default_speaker_cmn': 'cmn_test',
                'speaker_embedding_dim': 256
            },
            'speaker': {
                'embedding_dim': 256,
                'min_audio_duration': 1.0,
                'max_audio_duration': 10.0
            }
        })
        return config
    
    def test_end_to_end_synthesis(self, full_config):
        """Test end-to-end synthesis pipeline."""
        with patch('pathlib.Path.exists', return_value=True):
            # Initialize components
            tts_engine = IndexTTS2Engine(full_config, device="cpu")
            speaker_manager = SpeakerManager(full_config, device="cpu")
            audio_processor = AudioProcessor()
            
            # Test synthesis
            text = "這是一個測試"
            audio_array, sample_rate = tts_engine.synthesize(
                text=text,
                language="yue",
                emotion="neutral"
            )
            
            # Save audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                success = audio_processor.save_audio(
                    audio=torch.from_numpy(audio_array).unsqueeze(0),
                    filepath=tmp_file.name,
                    sample_rate=sample_rate
                )
                
                assert success
                assert Path(tmp_file.name).exists()
                
                # Clean up
                os.unlink(tmp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])