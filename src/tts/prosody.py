"""
Prosody Control and Transfer Module

Handles prosodic features including pitch (F0), energy, rhythm, and emotion
for both Cantonese and Mandarin speech synthesis.
"""

import torch
import torchaudio
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ProsodyController:
    """
    Prosody control system for manipulating pitch, energy, rhythm, and emotion
    in speech synthesis for both Cantonese and Mandarin.
    """
    
    def __init__(self, config: DictConfig, device: str = "auto"):
        """
        Initialize prosody controller.
        
        Args:
            config: Configuration with prosody settings
            device: Target device for computation
        """
        self.config = config
        self.device = self._setup_device(device)
        
        # Audio parameters
        self.sample_rate = config.prosody.get('sample_rate', 22050)
        self.hop_length = config.prosody.get('hop_length', 256)
        self.win_length = config.prosody.get('win_length', 1024)
        self.n_mels = config.prosody.get('n_mels', 80)
        self.fmin = config.prosody.get('fmin', 80)
        self.fmax = config.prosody.get('fmax', 8000)
        
        # Language-specific prosody models
        self.language_models = {}
        self.emotion_models = {}
        
        # Prosody extraction tools
        self.f0_extractor = None
        self.energy_extractor = None
        self.rhythm_analyzer = None
        
        self._initialize_components()
        logger.info("Prosody controller initialized")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _initialize_components(self):
        """Initialize prosody analysis and synthesis components."""
        try:
            # Initialize F0 extractor
            self.f0_extractor = self._create_f0_extractor()
            
            # Initialize energy extractor
            self.energy_extractor = self._create_energy_extractor()
            
            # Initialize rhythm analyzer
            self.rhythm_analyzer = self._create_rhythm_analyzer()
            
            # Initialize language-specific prosody models
            for language in ['yue', 'cmn']:
                self.language_models[language] = self._load_language_model(language)
            
            # Initialize emotion models
            for emotion in ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']:
                self.emotion_models[emotion] = self._load_emotion_model(emotion)
                
        except Exception as e:
            logger.error(f"Failed to initialize prosody components: {str(e)}")
            raise RuntimeError(f"Prosody initialization failed: {str(e)}")
    
    def _create_f0_extractor(self):
        """Create fundamental frequency (F0) extractor."""
        class F0Extractor:
            def __init__(self, sample_rate=22050, hop_length=256, fmin=80, fmax=800):
                self.sample_rate = sample_rate
                self.hop_length = hop_length
                self.fmin = fmin
                self.fmax = fmax
            
            def extract(self, audio):
                """Extract F0 using CREPE or similar algorithm."""
                # Placeholder F0 extraction using librosa
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                
                # Fill unvoiced regions with interpolation
                f0 = self._interpolate_f0(f0)
                return f0, voiced_flag
            
            def _interpolate_f0(self, f0):
                """Interpolate missing F0 values."""
                # Simple linear interpolation
                if np.any(np.isnan(f0)):
                    valid = ~np.isnan(f0)
                    if np.any(valid):
                        from scipy.interpolate import interp1d
                        x = np.arange(len(f0))
                        interp_func = interp1d(x[valid], f0[valid], 
                                             kind='linear', bounds_error=False, 
                                             fill_value='extrapolate')
                        f0 = interp_func(x)
                
                # Fill remaining NaNs with median
                if np.any(np.isnan(f0)):
                    median_f0 = np.nanmedian(f0)
                    f0 = np.nan_to_num(f0, nan=median_f0)
                
                return f0
        
        return F0Extractor(self.sample_rate, self.hop_length)
    
    def _create_energy_extractor(self):
        """Create energy extractor for audio signals."""
        class EnergyExtractor:
            def __init__(self, hop_length=256, win_length=1024):
                self.hop_length = hop_length
                self.win_length = win_length
            
            def extract(self, audio, sample_rate):
                """Extract frame-level energy."""
                # Compute STFT
                stft = librosa.stft(
                    audio,
                    n_fft=self.win_length,
                    hop_length=self.hop_length,
                    win_length=self.win_length
                )
                
                # Compute energy as sum of magnitude spectrum
                energy = np.sum(np.abs(stft), axis=0)
                
                # Normalize
                energy = energy / np.max(energy) if np.max(energy) > 0 else energy
                
                return energy
        
        return EnergyExtractor(self.hop_length, self.win_length)
    
    def _create_rhythm_analyzer(self):
        """Create rhythm and timing analyzer."""
        class RhythmAnalyzer:
            def __init__(self, sample_rate=22050, hop_length=256):
                self.sample_rate = sample_rate
                self.hop_length = hop_length
            
            def analyze(self, audio, f0=None):
                """Analyze rhythm patterns and timing."""
                # Extract onset strength
                onset_env = librosa.onset.onset_strength(
                    y=audio, 
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                
                # Detect onsets
                onsets = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                
                # Compute tempo
                tempo, _ = librosa.beat.beat_track(
                    onset_envelope=onset_env,
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                
                # Analyze speech rate if F0 is available
                speech_rate = None
                if f0 is not None:
                    speech_rate = self._estimate_speech_rate(f0)
                
                return {
                    'onset_strength': onset_env,
                    'onset_frames': onsets,
                    'tempo': tempo,
                    'speech_rate': speech_rate
                }
            
            def _estimate_speech_rate(self, f0):
                """Estimate speech rate from F0 contours."""
                # Count syllable-like patterns in F0
                f0_smooth = np.convolve(f0, np.ones(5)/5, mode='same')
                peaks, _ = self._find_peaks(f0_smooth, height=np.mean(f0_smooth))
                
                # Estimate syllables per second
                duration_seconds = len(f0) * self.hop_length / self.sample_rate
                syllables_per_second = len(peaks) / duration_seconds if duration_seconds > 0 else 0
                
                return syllables_per_second
            
            def _find_peaks(self, data, height=None):
                """Simple peak detection."""
                from scipy.signal import find_peaks
                return find_peaks(data, height=height)
        
        return RhythmAnalyzer(self.sample_rate, self.hop_length)
    
    def _load_language_model(self, language: str):
        """Load language-specific prosody model."""
        logger.info(f"Loading prosody model for {language}...")
        
        class LanguageProsodyModel(torch.nn.Module):
            def __init__(self, language):
                super().__init__()
                self.language = language
                
                # Language-specific prosody characteristics
                if language == 'yue':  # Cantonese
                    self.tone_patterns = 6  # 6-9 tones depending on analysis
                    self.pitch_range = [80, 400]  # Typical F0 range
                    self.speech_rate_mean = 5.5  # Syllables per second
                else:  # 'cmn' - Mandarin
                    self.tone_patterns = 4  # 4 main tones + neutral
                    self.pitch_range = [80, 350]  # Typical F0 range
                    self.speech_rate_mean = 6.0  # Syllables per second
                
                # Neural network for prosody prediction
                self.prosody_predictor = torch.nn.Sequential(
                    torch.nn.Linear(80, 256),  # Mel-spectrogram input
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 3)  # F0, energy, duration predictions
                )
            
            def forward(self, mel_spec):
                prosody_features = self.prosody_predictor(mel_spec.transpose(1, 2))
                return prosody_features
            
            def get_language_characteristics(self):
                """Get language-specific prosody characteristics."""
                return {
                    'tone_patterns': self.tone_patterns,
                    'pitch_range': self.pitch_range,
                    'speech_rate_mean': self.speech_rate_mean
                }
        
        model = LanguageProsodyModel(language)
        model.eval()
        if self.device.type == "cuda":
            model = model.cuda()
        
        return model
    
    def _load_emotion_model(self, emotion: str):
        """Load emotion-specific prosody model."""
        logger.info(f"Loading emotion model for {emotion}...")
        
        class EmotionProsodyModel(torch.nn.Module):
            def __init__(self, emotion):
                super().__init__()
                self.emotion = emotion
                
                # Emotion-specific prosody parameters
                self.emotion_params = self._get_emotion_parameters(emotion)
                
                # Neural network for emotion application
                self.emotion_applier = torch.nn.Sequential(
                    torch.nn.Linear(80, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 80)  # Same dimension as mel-spectrogram
                )
            
            def _get_emotion_parameters(self, emotion):
                """Define emotion-specific prosody parameters."""
                params = {
                    'neutral': {'energy_scale': 1.0, 'pitch_scale': 1.0, 'speed_scale': 1.0},
                    'happy': {'energy_scale': 1.2, 'pitch_scale': 1.15, 'speed_scale': 1.1},
                    'sad': {'energy_scale': 0.8, 'pitch_scale': 0.9, 'speed_scale': 0.9},
                    'angry': {'energy_scale': 1.3, 'pitch_scale': 1.2, 'speed_scale': 1.15},
                    'excited': {'energy_scale': 1.25, 'pitch_scale': 1.3, 'speed_scale': 1.2},
                    'calm': {'energy_scale': 0.9, 'pitch_scale': 0.95, 'speed_scale': 0.95}
                }
                return params.get(emotion, params['neutral'])
            
            def forward(self, mel_spec):
                # Apply emotion-specific modifications
                emotion_mel = self.emotion_applier(mel_spec.transpose(1, 2))
                
                # Apply scaling parameters
                params = self.emotion_params
                emotion_mel = emotion_mel * params['energy_scale']
                
                return emotion_mel.transpose(1, 2)
        
        model = EmotionProsodyModel(emotion)
        model.eval()
        if self.device.type == "cuda":
            model = model.cuda()
        
        return model
    
    def analyze_prosody(self, audio: np.ndarray, language: str = 'cmn') -> Dict:
        """
        Analyze prosodic features from audio.
        
        Args:
            audio: Input audio waveform
            language: Language code ('yue' or 'cmn')
            
        Returns:
            Dictionary with prosodic features
        """
        try:
            # Extract F0
            f0, voiced_flag = self.f0_extractor.extract(audio)
            
            # Extract energy
            energy = self.energy_extractor.extract(audio, self.sample_rate)
            
            # Analyze rhythm
            rhythm_features = self.rhythm_analyzer.analyze(audio, f0)
            
            # Language-specific analysis
            lang_model = self.language_models.get(language)
            if lang_model:
                lang_characteristics = lang_model.get_language_characteristics()
            else:
                lang_characteristics = {}
            
            # Compile prosody features
            prosody_features = {
                'f0': {
                    'values': f0,
                    'mean': np.mean(f0[f0 > 0]),  # Exclude unvoiced regions
                    'std': np.std(f0[f0 > 0]),
                    'min': np.min(f0[f0 > 0]),
                    'max': np.max(f0[f0 > 0]),
                    'range': np.max(f0[f0 > 0]) - np.min(f0[f0 > 0])
                },
                'energy': {
                    'values': energy,
                    'mean': np.mean(energy),
                    'std': np.std(energy),
                    'min': np.min(energy),
                    'max': np.max(energy)
                },
                'rhythm': rhythm_features,
                'voiced_ratio': np.sum(voiced_flag) / len(voiced_flag),
                'language_characteristics': lang_characteristics
            }
            
            logger.info(f"Prosody analysis completed for {language}")
            return prosody_features
            
        except Exception as e:
            logger.error(f"Prosody analysis failed: {str(e)}")
            raise RuntimeError(f"Prosody analysis failed: {str(e)}")
    
    def apply_emotion(self, mel_spec: torch.Tensor, emotion: str, intensity: float = 1.0) -> torch.Tensor:
        """
        Apply emotion to mel-spectrogram.
        
        Args:
            mel_spec: Input mel-spectrogram
            emotion: Emotion type
            intensity: Emotion intensity (0.0-2.0)
            
        Returns:
            Emotion-modified mel-spectrogram
        """
        if emotion not in self.emotion_models:
            logger.warning(f"Emotion '{emotion}' not supported, using neutral")
            emotion = 'neutral'
        
        try:
            emotion_model = self.emotion_models[emotion]
            
            with torch.no_grad():
                # Apply emotion model
                emotion_mel = emotion_model(mel_spec)
                
                # Blend with original based on intensity
                if intensity != 1.0:
                    emotion_mel = mel_spec * (1 - intensity) + emotion_mel * intensity
                
                return emotion_mel
                
        except Exception as e:
            logger.error(f"Emotion application failed: {str(e)}")
            return mel_spec  # Return original if failed
    
    def modify_pitch(self, mel_spec: torch.Tensor, pitch_shift: float, 
                    language: str = 'cmn') -> torch.Tensor:
        """
        Modify pitch of mel-spectrogram.
        
        Args:
            mel_spec: Input mel-spectrogram
            pitch_shift: Pitch shift in semitones
            language: Language for pitch range constraints
            
        Returns:
            Pitch-modified mel-spectrogram
        """
        try:
            # Get language-specific pitch constraints
            lang_model = self.language_models.get(language)
            if lang_model:
                lang_chars = lang_model.get_language_characteristics()
                pitch_range = lang_chars.get('pitch_range', [80, 350])
            else:
                pitch_range = [80, 350]
            
            # Apply pitch shift (simplified implementation)
            # In production, this would use more sophisticated pitch modification
            pitch_factor = 2 ** (pitch_shift / 12)  # Convert semitones to frequency ratio
            
            # Apply to higher frequency bins more strongly
            n_bins = mel_spec.shape[1]
            freq_weights = torch.linspace(0.5, 2.0, n_bins).to(mel_spec.device)
            pitch_weights = 1 + (pitch_factor - 1) * freq_weights
            
            modified_mel = mel_spec * pitch_weights.unsqueeze(0).unsqueeze(2)
            
            # Ensure within reasonable pitch range
            modified_mel = torch.clamp(modified_mel, min=0, max=None)
            
            return modified_mel
            
        except Exception as e:
            logger.error(f"Pitch modification failed: {str(e)}")
            return mel_spec  # Return original if failed
    
    def modify_speed(self, mel_spec: torch.Tensor, speed_ratio: float) -> torch.Tensor:
        """
        Modify speaking speed by stretching/compressing mel-spectrogram.
        
        Args:
            mel_spec: Input mel-spectrogram
            speed_ratio: Speed ratio (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
            
        Returns:
            Speed-modified mel-spectrogram
        """
        try:
            # Use interpolation for time stretching
            original_length = mel_spec.shape[2]
            new_length = int(original_length / speed_ratio)
            
            # Interpolate along time dimension
            modified_mel = torch.nn.functional.interpolate(
                mel_spec,
                size=new_length,
                mode='linear',
                align_corners=False
            )
            
            return modified_mel
            
        except Exception as e:
            logger.error(f"Speed modification failed: {str(e)}")
            return mel_spec  # Return original if failed
    
    def transfer_prosody(self, source_mel: torch.Tensor, target_mel: torch.Tensor,
                        transfer_f0: bool = True, transfer_energy: bool = True,
                        transfer_rhythm: bool = False) -> torch.Tensor:
        """
        Transfer prosodic features from source to target mel-spectrogram.
        
        Args:
            source_mel: Source mel-spectrogram (prosody donor)
            target_mel: Target mel-spectrogram (content carrier)
            transfer_f0: Whether to transfer F0/pitch
            transfer_energy: Whether to transfer energy
            transfer_rhythm: Whether to transfer rhythm/timing
            
        Returns:
            Mel-spectrogram with transferred prosody
        """
        try:
            # Convert to numpy for analysis
            source_np = source_mel.cpu().numpy()
            target_np = target_mel.cpu().numpy()
            
            # Extract audio waveforms (approximate)
            source_audio = self._mel_to_audio_approximate(source_np[0])
            target_audio = self._mel_to_audio_approximate(target_np[0])
            
            # Analyze prosody of both source and target
            source_prosody = self.analyze_prosody(source_audio)
            target_prosody = self.analyze_prosody(target_audio)
            
            # Apply prosody transfer
            result_mel = target_mel.clone()
            
            if transfer_f0:
                result_mel = self._transfer_f0(result_mel, source_prosody, target_prosody)
            
            if transfer_energy:
                result_mel = self._transfer_energy(result_mel, source_prosody, target_prosody)
            
            if transfer_rhythm:
                result_mel = self._transfer_rhythm(result_mel, source_prosody, target_prosody)
            
            return result_mel
            
        except Exception as e:
            logger.error(f"Prosody transfer failed: {str(e)}")
            return target_mel  # Return target if transfer fails
    
    def _mel_to_audio_approximate(self, mel_spec: np.ndarray) -> np.ndarray:
        """Approximate audio reconstruction from mel-spectrogram."""
        # Simplified audio reconstruction for analysis purposes
        time_steps = mel_spec.shape[1]
        audio_length = time_steps * self.hop_length
        
        # Generate placeholder audio
        audio = np.random.randn(audio_length) * 0.1
        
        # Add some structure based on mel-spectrogram
        for t in range(time_steps):
            start_idx = t * self.hop_length
            end_idx = min(start_idx + self.hop_length, audio_length)
            
            # Simple amplitude modulation based on mel energy
            mel_energy = np.mean(mel_spec[:, t])
            audio[start_idx:end_idx] += 0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 1, end_idx - start_idx)) * mel_energy
        
        return audio
    
    def _transfer_f0(self, mel_spec: torch.Tensor, source_prosody: Dict, 
                    target_prosody: Dict) -> torch.Tensor:
        """Transfer F0 characteristics from source to target."""
        # Simplified F0 transfer
        source_f0_mean = source_prosody['f0']['mean']
        target_f0_mean = target_prosody['f0']['mean']
        
        if source_f0_mean > 0 and target_f0_mean > 0:
            f0_ratio = source_f0_mean / target_f0_mean
            mel_spec = mel_spec * f0_ratio
        
        return mel_spec
    
    def _transfer_energy(self, mel_spec: torch.Tensor, source_prosody: Dict,
                        target_prosody: Dict) -> torch.Tensor:
        """Transfer energy characteristics from source to target."""
        source_energy_mean = source_prosody['energy']['mean']
        target_energy_mean = target_prosody['energy']['mean']
        
        if target_energy_mean > 0:
            energy_ratio = source_energy_mean / target_energy_mean
            mel_spec = mel_spec * energy_ratio
        
        return mel_spec
    
    def _transfer_rhythm(self, mel_spec: torch.Tensor, source_prosody: Dict,
                        target_prosody: Dict) -> torch.Tensor:
        """Transfer rhythm characteristics from source to target."""
        # This would involve more sophisticated rhythm transfer
        # For now, return the original mel_spec
        return mel_spec
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return list(self.emotion_models.keys())
    
    def get_language_characteristics(self, language: str) -> Dict:
        """Get prosodic characteristics for a language."""
        lang_model = self.language_models.get(language)
        if lang_model:
            return lang_model.get_language_characteristics()
        else:
            return {}
    
    def get_prosody_stats(self, audio: np.ndarray, language: str = 'cmn') -> Dict:
        """Get statistical summary of prosodic features."""
        prosody_features = self.analyze_prosody(audio, language)
        
        stats = {
            'f0_stats': {
                'mean': prosody_features['f0']['mean'],
                'std': prosody_features['f0']['std'],
                'range': prosody_features['f0']['range']
            },
            'energy_stats': {
                'mean': prosody_features['energy']['mean'],
                'std': prosody_features['energy']['std']
            },
            'rhythm_stats': {
                'tempo': prosody_features['rhythm']['tempo'],
                'speech_rate': prosody_features['rhythm']['speech_rate']
            },
            'voiced_ratio': prosody_features['voiced_ratio']
        }
        
        return stats