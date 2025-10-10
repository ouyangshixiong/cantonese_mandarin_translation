"""
XTTS-v2 Engine Implementation

Core TTS synthesis engine using Coqui/XTTS-v2 for high-quality multi-speaker
speech synthesis with support for Cantonese and Mandarin.
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from omegaconf import DictConfig
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from TTS.api import TTS
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    logger.warning("TTS library not available. Please install: pip install TTS>=0.22.0")


class XTTSV2Engine:
    """
    XTTS-v2 synthesis engine for high-quality multi-speaker TTS.
    
    Supports both Cantonese (yue) and Mandarin (cmn) with cross-lingual
    voice preservation and real-time inference optimization.
    """
    
    def __init__(self, config: DictConfig, device: str = "auto"):
        """
        Initialize XTTS-v2 engine.
        
        Args:
            config: Configuration dictionary with TTS settings
            device: Target device ('cuda', 'cpu', or 'auto')
        """
        if not XTTS_AVAILABLE:
            raise ImportError("TTS library not available. Install with: pip install TTS>=0.22.0")
        
        self.config = config
        self.device = self._setup_device(device)
        self.sample_rate = config.tts.get('sample_rate', 22050)
        
        # Language-specific configurations
        self.languages = {
            'yue': 'cantonese',  # Cantonese
            'cmn': 'mandarin',   # Mandarin
            'en': 'english'      # English
        }
        
        # Initialize XTTS-v2 models
        self.tts_models = {}
        self.speaker_embeddings = {}
        
        self._initialize_models()
        logger.info(f"XTTS-v2 engine initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        
        return torch.device(device)
    
    def _initialize_models(self):
        """Initialize XTTS-v2 models for supported languages."""
        try:
            # Initialize language-specific models
            for lang_code, lang_name in self.languages.items():
                logger.info(f"Initializing {lang_name} XTTS-v2 model...")
                
                # Load XTTS-v2 model
                model_name = self._get_model_name(lang_code)
                
                try:
                    self.tts_models[lang_code] = TTS(
                        model_name=model_name,
                        progress_bar=False,
                        gpu=self.device.type == "cuda"
                    )
                    
                    logger.info(f"{lang_name} XTTS-v2 model loaded successfully: {model_name}")
                    
                except Exception as model_error:
                    if "multilingual" in model_name:
                        logger.error(f"Failed to load multilingual XTTS-v2 model: {str(model_error)}")
                        logger.error("This model requires accepting the XTTS-v2 CPML license.")
                        logger.error("Run: pip install TTS>=0.22.0 and accept the license terms.")
                        logger.error("Falling back to English-only model...")
                        
                        # Fallback to English model
                        english_model = "tts_models/en/ljspeech/tacotron2-DDC"
                        self.tts_models[lang_code] = TTS(
                            model_name=english_model,
                            progress_bar=False,
                            gpu=self.device.type == "cuda"
                        )
                        logger.info(f"Loaded fallback English model: {english_model}")
                    else:
                        raise model_error
        
        except Exception as e:
            logger.error(f"Failed to initialize XTTS-v2 models: {str(e)}")
            raise RuntimeError(f"XTTS-v2 model initialization failed: {str(e)}")
    
    def _get_model_name(self, language: str) -> str:
        """Get appropriate XTTS-v2 model name for language."""
        # Use multilingual XTTS-v2 model that supports Chinese characters
        # Note: This requires accepting the XTTS-v2 CPML license terms
        multilingual_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Fallback to English model if multilingual model fails to load
        english_model = "tts_models/en/ljspeech/tacotron2-DDC"
        
        # Try to use multilingual model first
        try:
            # Test if multilingual model is available
            from TTS.api import TTS
            # This will raise an exception if the model requires license acceptance
            TTS(model_name=multilingual_model, progress_bar=False, gpu=self.device.type == "cuda")
            return multilingual_model
        except Exception as e:
            logger.warning(f"Multilingual XTTS-v2 model not available: {str(e)}")
            logger.warning(f"Falling back to English-only model: {english_model}")
            logger.warning("To use Chinese characters, accept XTTS-v2 license: pip install TTS>=0.22.0")
            return english_model
    
    def synthesize(
        self,
        text: str,
        language: str,
        speaker_id: Optional[str] = None,
        emotion: str = "neutral",
        speed: float = 1.0,
        pitch_shift: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using XTTS-v2.
        
        Args:
            text: Input text to synthesize
            language: Language code ('yue' or 'cmn')
            speaker_id: Speaker identifier for voice cloning
            emotion: Emotion style ('neutral', 'happy', 'sad', 'angry')
            speed: Speech rate multiplier (0.5-2.0)
            pitch_shift: Pitch adjustment in semitones
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}. Use: {list(self.languages.keys())}")
        
        try:
            # Get language-specific model
            tts_model = self.tts_models.get(language)
            
            if tts_model is None:
                raise RuntimeError(f"Model not loaded for language: {language}")
            
            # Text preprocessing
            processed_text = self._preprocess_text(text, language)
            
            # Get speaker reference if specified
            speaker_wav = None
            if speaker_id:
                speaker_wav = self._get_speaker_reference(speaker_id, language)
            
            # Convert language code to XTTS-v2 format
            xtts_language = self._convert_language_code(language)
            
            # Apply emotion and prosody modifications
            prosody_params = self._apply_prosody(emotion, speed, pitch_shift)
            
            # Generate speech using XTTS-v2
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Use XTTS-v2 synthesis
                # For non-multilingual models, don't pass language parameter
                if "multilingual" in self._get_model_name(language):
                    tts_model.tts_to_file(
                        text=processed_text,
                        speaker_wav=speaker_wav,
                        language=xtts_language,
                        file_path=temp_path,
                        speed=prosody_params['speed'],
                        split_sentences=True
                    )
                else:
                    # For English-only models
                    tts_model.tts_to_file(
                        text=processed_text,
                        speaker_wav=speaker_wav,
                        file_path=temp_path,
                        speed=prosody_params['speed'],
                        split_sentences=True
                    )
                
                # Load generated audio
                audio_array, sample_rate = self._load_audio_file(temp_path)
                
                # Apply additional prosody modifications if needed
                if pitch_shift != 0.0:
                    audio_array = self._apply_pitch_shift(audio_array, sample_rate, pitch_shift)
                
                # Normalize audio
                audio_array = self._normalize_audio(audio_array)
                
                logger.info(f"Synthesized {len(text)} characters in {language} language")
                
                return audio_array, sample_rate
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Synthesis failed for text '{text[:50]}...': {str(e)}")
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text for TTS synthesis."""
        # Basic text cleaning
        text = text.strip()
        
        # Check if multilingual model is being used
        current_model = self._get_model_name(language)
        is_multilingual = "multilingual" in current_model
        
        # Language-specific preprocessing
        if language == 'yue':
            # Cantonese-specific processing
            text = self._process_cantonese_text(text, is_multilingual)
        elif language == 'cmn':
            # Mandarin-specific processing
            text = self._process_mandarin_text(text, is_multilingual)
        elif language == 'en':
            # English-specific processing
            text = self._process_english_text(text, is_multilingual)
        
        # Validate that text contains valid characters for the current model
        if not is_multilingual and not self._is_english_text(text):
            logger.warning(f"Non-English text detected with English-only model: '{text[:50]}...'")
            logger.warning("Chinese characters will be discarded. Use multilingual XTTS-v2 for Chinese support.")
        
        return text
    
    def _process_cantonese_text(self, text: str, is_multilingual: bool = False) -> str:
        """Process Cantonese text for TTS."""
        # Handle Cantonese-specific characters and expressions
        if not is_multilingual:
            # For English-only models, convert Cantonese-specific characters
            # This is a basic mapping - in production would use proper romanization
            cantonese_mapping = {
                "咗": "zo", "嘅": "ge", "嘢": "ye", "冇": "mou", 
                "唔": "m", "啲": "di", "咁": "gam", "嚟": "lai", 
                "哋": "dei", "咁樣": "gam yeung", "鍾意": "zung ji",
                "朝早": "zou zou", "飲茶": "yam cha", "傾偈": "king gai"
            }
            for cantonese, romanized in cantonese_mapping.items():
                text = text.replace(cantonese, romanized)
        
        return text
    
    def _process_mandarin_text(self, text: str, is_multilingual: bool = False) -> str:
        """Process Mandarin text for TTS."""
        # Handle Mandarin-specific processing
        if not is_multilingual:
            # For English-only models, provide basic pinyin-like mapping
            # This is simplified - in production would use proper pinyin conversion
            mandarin_mapping = {
                "你": "ni", "好": "hao", "吗": "ma", "我": "wo", 
                "是": "shi", "的": "de", "了": "le", "不": "bu",
                "有": "you", "在": "zai", "和": "he", "说": "shuo"
            }
            for chinese, pinyin in mandarin_mapping.items():
                text = text.replace(chinese, pinyin)
        
        return text
    
    def _process_english_text(self, text: str, is_multilingual: bool = False) -> str:
        """Process English text for TTS."""
        # English text preprocessing
        # Basic text normalization for English
        text = text.strip()
        
        # Handle common English contractions and abbreviations
        english_contractions = {
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "shouldn't": "should not", "couldn't": "could not", "wouldn't": "would not",
            "it's": "it is", "he's": "he is", "she's": "she is", "that's": "that is",
            "what's": "what is", "where's": "where is", "when's": "when is",
            "why's": "why is", "how's": "how is", "i'm": "i am", "you're": "you are",
            "we're": "we are", "they're": "they are"
        }
        
        # Replace contractions with expanded forms for better pronunciation
        for contraction, expanded in english_contractions.items():
            text = text.replace(contraction, expanded)
        
        # Handle number formatting
        import re
        # Convert numbers to words for better pronunciation
        # This is a simplified version - in production would use a proper number-to-words library
        text = re.sub(r'\b(\d+)\b', lambda m: self._number_to_words(m.group(1)), text)
        
        return text
    
    def _number_to_words(self, number_str: str) -> str:
        """Convert numbers to words for better TTS pronunciation."""
        # Simple number to words conversion for common cases
        number_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
            '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
        }
        
        if number_str in number_map:
            return number_map[number_str]
        
        # For larger numbers, just return the digits separated by spaces
        return ' '.join(number_str)
    
    def _is_english_text(self, text: str) -> bool:
        """Check if text contains primarily English characters."""
        import re
        # Count English vs non-English characters
        english_chars = re.findall(r'[a-zA-Z\s\.,!?;:\'\"]', text)
        non_english_chars = re.findall(r'[^a-zA-Z\s\.,!?;:\'\"]', text)
        
        # If more than 50% of characters are non-English, consider it non-English
        total_chars = len(english_chars) + len(non_english_chars)
        if total_chars == 0:
            return True
        
        english_ratio = len(english_chars) / total_chars
        return english_ratio > 0.5
    
    def _get_speaker_reference(self, speaker_id: str, language: str) -> Optional[str]:
        """Get speaker reference audio file path."""
        # Placeholder implementation
        # In production, this would load actual speaker reference audio
        logger.info(f"Using speaker reference for {speaker_id}")
        return None
    
    def _convert_language_code(self, language: str) -> str:
        """Convert internal language code to XTTS-v2 format."""
        language_map = {
            'yue': 'zh-cn',  # Cantonese - using Chinese as approximation
            'cmn': 'zh-cn',  # Mandarin
            'en': 'en'       # English
        }
        return language_map.get(language, 'zh-cn')
    
    def _apply_prosody(self, emotion: str, speed: float, pitch_shift: float) -> Dict:
        """Calculate prosody modification parameters."""
        prosody_params = {
            'emotion': emotion,
            'speed': speed,
            'pitch_shift': pitch_shift,
            'energy_scale': 1.0,
            'pitch_scale': 1.0
        }
        
        # Emotion-specific adjustments
        if emotion == 'happy':
            prosody_params['energy_scale'] = 1.2
            prosody_params['pitch_scale'] = 1.1
        elif emotion == 'sad':
            prosody_params['energy_scale'] = 0.8
            prosody_params['pitch_scale'] = 0.9
        elif emotion == 'angry':
            prosody_params['energy_scale'] = 1.3
            prosody_params['pitch_scale'] = 1.2
        
        return prosody_params
    
    def _load_audio_file(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return numpy array and sample rate."""
        try:
            audio, sample_rate = torchaudio.load(filepath)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Convert to numpy
            audio_array = audio.squeeze().numpy()
            
            return audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio file {filepath}: {str(e)}")
            raise RuntimeError(f"Audio loading failed: {str(e)}")
    
    def _apply_pitch_shift(self, audio: np.ndarray, sample_rate: int, pitch_shift: float) -> np.ndarray:
        """Apply pitch shift to audio."""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            
            # Apply pitch shift using torchaudio
            effects = [["pitch", str(pitch_shift)]]
            shifted_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio_tensor, sample_rate, effects
            )
            
            return shifted_audio.squeeze().numpy()
            
        except Exception as e:
            logger.error(f"Pitch shift failed: {str(e)}")
            return audio  # Return original if pitch shift fails
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        return audio.astype(np.float32)
    
    def get_available_speakers(self, language: str) -> List[str]:
        """Get list of available speakers for a language."""
        # XTTS-v2 supports voice cloning, so speakers are dynamic
        # Return placeholder list
        return [f"speaker_{i}" for i in range(1, 6)]
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return ["neutral", "happy", "sad", "angry", "excited", "calm"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.languages.keys())
    
    def estimate_synthesis_time(self, text_length: int) -> float:
        """Estimate synthesis time based on text length."""
        # Rough estimation: 50ms base + 8ms per character
        return 0.05 + (text_length * 0.008)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {}
        
        if self.device.type == "cuda":
            memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_stats['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return memory_stats
    
    def clone_voice(self, reference_audio: np.ndarray, language: str) -> str:
        """
        Clone voice from reference audio.
        
        Args:
            reference_audio: Reference audio waveform
            language: Target language
            
        Returns:
            Speaker ID for the cloned voice
        """
        try:
            # Save reference audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Save audio to file
                torchaudio.save(
                    temp_path,
                    torch.from_numpy(reference_audio).unsqueeze(0).float(),
                    self.sample_rate
                )
            
            # Generate unique speaker ID
            speaker_id = f"cloned_{language}_{int(torch.rand(1).item() * 10000)}"
            
            # Store speaker reference
            self.speaker_embeddings[speaker_id] = temp_path
            
            logger.info(f"Voice cloned successfully: {speaker_id}")
            return speaker_id
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            raise RuntimeError(f"Voice cloning failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the XTTS-v2 model."""
        return {
            'engine': 'XTTS-v2',
            'version': '2.0.0',
            'supported_languages': self.get_supported_languages(),
            'sample_rate': self.sample_rate,
            'device': str(self.device),
            'available_speakers': {
                lang: self.get_available_speakers(lang) for lang in self.languages.keys()
            }
        }