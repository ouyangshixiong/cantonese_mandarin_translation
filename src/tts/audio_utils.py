"""
Audio Processing Utilities for TTS Module

Provides audio format conversion, preprocessing, quality assessment, and
utility functions for TTS synthesis and voice conversion.
"""

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"


@dataclass
class AudioInfo:
    """Audio file information."""
    sample_rate: int
    channels: int
    duration: float
    frames: int
    format: str
    subtype: str
    bitrate: Optional[int] = None


@dataclass
class AudioQualityMetrics:
    """Audio quality assessment metrics."""
    snr_db: float
    dynamic_range: float
    clipping_ratio: float
    spectral_rolloff: float
    zero_crossing_rate: float
    spectral_centroid: float
    overall_score: float


class AudioProcessor:
    """
    Comprehensive audio processing utilities for TTS applications.
    Handles format conversion, preprocessing, quality assessment, and
    audio manipulation for speech synthesis and voice conversion.
    """
    
    def __init__(self, sample_rate: int = 22050, n_mels: int = 80, 
                 hop_length: int = 256, win_length: int = 1024):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel-frequency bins
            hop_length: Hop length for STFT
            win_length: Window length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = win_length
        
        # Audio transformations
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=self.n_fft
        )
        
        self.inverse_mel_transform = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=win_length // 2 + 1
        )
        
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        
        logger.info(f"Audio processor initialized with SR={sample_rate}, n_mels={n_mels}")
    
    def load_audio(self, filepath: Union[str, Path]) -> Tuple[torch.Tensor, AudioInfo]:
        """
        Load audio file with comprehensive error handling.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, audio_info)
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Audio file not found: {filepath}")
            
            # Load audio using torchaudio
            audio, sample_rate = torchaudio.load(filepath)
            
            # Get file info
            info = torchaudio.info(filepath)
            
            audio_info = AudioInfo(
                sample_rate=info.sample_rate,
                channels=info.num_channels,
                duration=info.num_frames / info.sample_rate,
                frames=info.num_frames,
                format=filepath.suffix.lower().lstrip('.'),
                subtype=getattr(info, 'encoding', 'unknown')
            )
            
            logger.info(f"Loaded audio: {filepath.name} - {audio_info.duration:.2f}s, "
                       f"{audio_info.channels}ch, {info.sample_rate}Hz")
            
            return audio, audio_info
            
        except Exception as e:
            logger.error(f"Failed to load audio {filepath}: {str(e)}")
            raise RuntimeError(f"Audio loading failed: {str(e)}")
    
    def save_audio(self, audio: torch.Tensor, filepath: Union[str, Path], 
                   sample_rate: Optional[int] = None, format: str = 'wav',
                   subtype: str = 'PCM_16') -> bool:
        """
        Save audio file with format options.
        
        Args:
            audio: Audio tensor
            filepath: Output file path
            sample_rate: Sample rate (uses processor default if None)
            format: Output format
            subtype: Audio subtype
            
        Returns:
            Success flag
        """
        try:
            filepath = Path(filepath)
            sample_rate = sample_rate or self.sample_rate
            
            # Ensure audio is 2D (channels, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Handle different formats
            if format.lower() == 'mp3':
                # For MP3, save as WAV first then convert
                temp_wav = filepath.with_suffix('.temp.wav')
                torchaudio.save(temp_wav, audio, sample_rate)
                
                # Convert to MP3 using soundfile
                audio_np, sr = librosa.load(temp_wav, sr=sample_rate)
                sf.write(filepath, audio_np, sr, format='MP3')
                
                # Clean up temp file
                temp_wav.unlink()
            else:
                # Save directly using torchaudio
                torchaudio.save(filepath, audio, sample_rate, format=format)
            
            logger.info(f"Saved audio: {filepath.name} - {audio.shape[-1]/sample_rate:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio {filepath}: {str(e)}")
            return False
    
    def convert_sample_rate(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """
        Convert audio sample rate.
        
        Args:
            audio: Input audio tensor
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio tensor
        """
        if orig_sr == target_sr:
            return audio
        
        try:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            return resampler(audio)
            
        except Exception as e:
            logger.error(f"Sample rate conversion failed: {str(e)}")
            # Fallback to librosa
            audio_np = audio.cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]  # Take first channel
            
            resampled_np = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
            return torch.from_numpy(resampled_np).unsqueeze(0)
    
    def convert_to_mono(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert stereo audio to mono.
        
        Args:
            audio: Input audio tensor (channels, samples)
            
        Returns:
            Mono audio tensor (1, samples)
        """
        if audio.shape[0] == 1:
            return audio
        
        # Average across channels
        mono_audio = torch.mean(audio, dim=0, keepdim=True)
        return mono_audio
    
    def normalize_audio(self, audio: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Input audio tensor
            target_db: Target dB level
            
        Returns:
            Normalized audio tensor
        """
        try:
            # Calculate current RMS
            rms = torch.sqrt(torch.mean(audio ** 2))
            
            if rms > 0:
                # Convert to dB
                current_db = 20 * torch.log10(rms)
                
                # Calculate gain needed
                gain_db = target_db - current_db
                gain_linear = 10 ** (gain_db / 20)
                
                # Apply gain
                normalized_audio = audio * gain_linear
                
                # Prevent clipping
                max_val = torch.max(torch.abs(normalized_audio))
                if max_val > 0.95:
                    normalized_audio = normalized_audio * (0.95 / max_val)
                
                return normalized_audio
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Audio normalization failed: {str(e)}")
            return audio
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Mel-spectrogram tensor
        """
        try:
            # Ensure mono
            if audio.shape[0] > 1:
                audio = self.convert_to_mono(audio)
            
            # Generate mel-spectrogram
            mel_spec = self.mel_transform(audio)
            
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-8)
            
            return mel_spec
            
        except Exception as e:
            logger.error(f"Mel-spectrogram conversion failed: {str(e)}")
            raise RuntimeError(f"Mel-spectrogram conversion failed: {str(e)}")
    
    def mel_to_audio(self, mel_spec: torch.Tensor, 
                    use_griffin_lim: bool = True) -> torch.Tensor:
        """
        Convert mel-spectrogram back to audio.
        
        Args:
            mel_spec: Input mel-spectrogram
            use_griffin_lim: Use Griffin-Lim algorithm
            
        Returns:
            Audio tensor
        """
        try:
            # Convert from log scale
            mel_spec = torch.exp(mel_spec)
            
            if use_griffin_lim:
                # Use Griffin-Lim algorithm
                linear_spec = self.inverse_mel_transform(mel_spec)
                audio = self.griffin_lim(linear_spec)
            else:
                # Simple approximation (lower quality)
                audio = self._simple_mel_to_audio(mel_spec)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {str(e)}")
            # Fallback to simple reconstruction
            return self._simple_mel_to_audio(torch.exp(mel_spec))
    
    def _simple_mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Simple mel-to-audio conversion (lower quality)."""
        # This is a very simplified approach
        # In production, use proper neural vocoders
        
        batch_size, n_mels, time_steps = mel_spec.shape
        
        # Generate placeholder audio
        audio_length = time_steps * self.hop_length
        audio = torch.randn(batch_size, audio_length) * 0.1
        
        # Add harmonic content based on mel-spectrogram
        for b in range(batch_size):
            for t in range(time_steps):
                start_idx = t * self.hop_length
                end_idx = min(start_idx + self.hop_length, audio_length)
                
                # Simple frequency modulation based on mel energy
                mel_energy = mel_spec[b, :, t].mean()
                freq = 200 + (mel_energy * 200)  # 200-400 Hz range
                
                t_vec = torch.linspace(0, 1, end_idx - start_idx)
                audio[b, start_idx:end_idx] += 0.5 * torch.sin(2 * np.pi * freq * t_vec) * mel_energy
        
        return audio
    
    def assess_quality(self, audio: torch.Tensor, 
                      sample_rate: Optional[int] = None) -> AudioQualityMetrics:
        """
        Assess audio quality with various metrics.
        
        Args:
            audio: Input audio tensor
            sample_rate: Sample rate (uses default if None)
            
        Returns:
            Audio quality metrics
        """
        try:
            sample_rate = sample_rate or self.sample_rate
            
            # Convert to numpy for analysis
            if audio.dim() > 1:
                audio_np = audio[0].cpu().numpy()  # Take first channel
            else:
                audio_np = audio.cpu().numpy()
            
            # Signal-to-noise ratio (simplified)
            snr_db = self._estimate_snr(audio_np)
            
            # Dynamic range
            dynamic_range = np.max(np.abs(audio_np)) - np.min(np.abs(audio_np))
            
            # Clipping detection
            clipping_ratio = np.sum(np.abs(audio_np) > 0.95) / len(audio_np)
            
            # Spectral features
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=sample_rate)[0].mean()
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)[0].mean()
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_np)[0].mean()
            
            # Overall quality score (weighted combination)
            overall_score = (
                0.3 * min(snr_db / 30, 1.0) +  # SNR component
                0.2 * min(dynamic_range, 1.0) +  # Dynamic range component
                0.2 * max(0, 1 - clipping_ratio * 10) +  # Clipping component
                0.3 * (1 - abs(zcr - 0.1) * 5)  # ZCR component (speech-like)
            )
            
            metrics = AudioQualityMetrics(
                snr_db=snr_db,
                dynamic_range=dynamic_range,
                clipping_ratio=clipping_ratio,
                spectral_rolloff=spectral_rolloff,
                zero_crossing_rate=zcr,
                spectral_centroid=spectral_centroid,
                overall_score=max(0, min(1, overall_score))
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            # Return default metrics
            return AudioQualityMetrics(
                snr_db=0.0,
                dynamic_range=0.0,
                clipping_ratio=0.0,
                spectral_rolloff=0.0,
                zero_crossing_rate=0.0,
                spectral_centroid=0.0,
                overall_score=0.0
            )
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Very simplified SNR estimation
        # In production, use proper noise estimation
        
        # Split into signal and noise estimates
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise from quieter segments
        sorted_audio = np.sort(np.abs(audio))
        noise_samples = int(len(sorted_audio) * 0.1)  # Bottom 10%
        noise_power = np.mean(sorted_audio[:noise_samples] ** 2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 30.0  # High SNR if no noise detected
        
        return snr_db
    
    def apply_effects(self, audio: torch.Tensor, 
                     effects: List[List[str]]) -> torch.Tensor:
        """
        Apply audio effects using SoX.
        
        Args:
            audio: Input audio tensor
            effects: List of SoX effect commands
            
        Returns:
            Processed audio tensor
        """
        try:
            # Apply effects using torchaudio SoX bindings
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio, self.sample_rate, effects
            )
            
            return augmented
            
        except Exception as e:
            logger.error(f"Audio effects failed: {str(e)}")
            return audio  # Return original if effects fail
    
    def change_speed(self, audio: torch.Tensor, speed_factor: float) -> torch.Tensor:
        """
        Change audio playback speed without affecting pitch.
        
        Args:
            audio: Input audio tensor
            speed_factor: Speed factor (1.0 = normal, 0.5 = half speed, etc.)
            
        Returns:
            Speed-modified audio tensor
        """
        effects = [["speed", str(speed_factor)]]
        return self.apply_effects(audio, effects)
    
    def change_pitch(self, audio: torch.Tensor, pitch_shift: float) -> torch.Tensor:
        """
        Change audio pitch without affecting speed.
        
        Args:
            audio: Input audio tensor
            pitch_shift: Pitch shift in semitones
            
        Returns:
            Pitch-modified audio tensor
        """
        effects = [["pitch", str(pitch_shift)]]
        return self.apply_effects(audio, effects)
    
    def add_reverb(self, audio: torch.Tensor, reverberance: float = 50.0) -> torch.Tensor:
        """
        Add reverb effect to audio.
        
        Args:
            audio: Input audio tensor
            reverberance: Reverberance percentage (0-100)
            
        Returns:
            Audio with reverb
        """
        effects = [["reverb", str(reverberance)]]
        return self.apply_effects(audio, effects)
    
    def trim_silence(self, audio: torch.Tensor, 
                    top_db: int = 20) -> torch.Tensor:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: Input audio tensor
            top_db: Threshold in dB below reference
            
        Returns:
            Trimmed audio tensor
        """
        try:
            # Convert to numpy for librosa
            if audio.dim() > 1:
                audio_np = audio[0].cpu().numpy()
            else:
                audio_np = audio.cpu().numpy()
            
            # Trim silence
            trimmed, _ = librosa.effects.trim(audio_np, top_db=top_db)
            
            return torch.from_numpy(trimmed).unsqueeze(0).float()
            
        except Exception as e:
            logger.error(f"Silence trimming failed: {str(e)}")
            return audio
    
    def split_audio(self, audio: torch.Tensor, 
                   segment_length: float) -> List[torch.Tensor]:
        """
        Split audio into segments of specified length.
        
        Args:
            audio: Input audio tensor
            segment_length: Segment length in seconds
            
        Returns:
            List of audio segments
        """
        try:
            segment_samples = int(segment_length * self.sample_rate)
            total_samples = audio.shape[-1]
            
            segments = []
            for start in range(0, total_samples, segment_samples):
                end = min(start + segment_samples, total_samples)
                segment = audio[..., start:end]
                
                # Only add non-empty segments
                if segment.shape[-1] > 0:
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Audio splitting failed: {str(e)}")
            return [audio]  # Return original as single segment
    
    def get_audio_info(self, filepath: Union[str, Path]) -> AudioInfo:
        """
        Get comprehensive audio file information.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio information
        """
        try:
            filepath = Path(filepath)
            info = torchaudio.info(filepath)
            
            # Get format-specific information
            format_info = self._get_format_info(filepath)
            
            audio_info = AudioInfo(
                sample_rate=info.sample_rate,
                channels=info.num_channels,
                duration=info.num_frames / info.sample_rate,
                frames=info.num_frames,
                format=format_info.get('format', filepath.suffix.lower().lstrip('.')),
                subtype=format_info.get('subtype', 'unknown'),
                bitrate=format_info.get('bitrate')
            )
            
            return audio_info
            
        except Exception as e:
            logger.error(f"Failed to get audio info for {filepath}: {str(e)}")
            raise RuntimeError(f"Audio info extraction failed: {str(e)}")
    
    def _get_format_info(self, filepath: Path) -> Dict[str, any]:
        """Get format-specific audio information."""
        try:
            # Use soundfile for detailed format info
            info = sf.info(filepath)
            
            return {
                'format': info.format,
                'subtype': info.subtype,
                'bitrate': getattr(info, 'bitrate', None)
            }
            
        except Exception:
            # Fallback to basic info
            return {'format': filepath.suffix.lower().lstrip('.')}
    
    def batch_process(self, audio_files: List[Union[str, Path]], 
                     process_func, **kwargs) -> List[Any]:
        """
        Batch process multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            process_func: Processing function to apply
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, filepath in enumerate(audio_files):
            try:
                logger.info(f"Processing file {i+1}/{len(audio_files)}: {filepath}")
                
                # Load audio
                audio, _ = self.load_audio(filepath)
                
                # Apply processing function
                result = process_func(audio, **kwargs)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {str(e)}")
                results.append(None)
        
        return results
    
    def validate_audio_format(self, filepath: Union[str, Path]) -> bool:
        """
        Validate if audio file format is supported.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Validation result
        """
        try:
            filepath = Path(filepath)
            
            # Check file extension
            supported_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            if filepath.suffix.lower() not in supported_extensions:
                return False
            
            # Try to load audio info
            torchaudio.info(filepath)
            return True
            
        except Exception:
            return False