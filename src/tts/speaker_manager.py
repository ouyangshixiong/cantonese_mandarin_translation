"""
Speaker Management and Voice Cloning Module

Handles speaker embedding extraction, voice model management, and few-shot
voice cloning capabilities for personalized TTS synthesis.
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Speaker information data structure."""
    speaker_id: str
    name: str
    language: str
    gender: str
    age_group: str
    embedding: Optional[torch.Tensor] = None
    audio_sample_path: Optional[str] = None
    metadata: Dict[str, Any] = None


class SpeakerManager:
    """
    Manages speaker embeddings, voice models, and provides voice cloning
    capabilities for personalized TTS synthesis.
    """
    
    def __init__(self, config: DictConfig, device: str = "auto"):
        """
        Initialize speaker manager.
        
        Args:
            config: Configuration with speaker management settings
            device: Target device for computation
        """
        self.config = config
        self.device = self._setup_device(device)
        
        # Speaker database
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.speaker_embeddings: Dict[str, torch.Tensor] = {}
        self.voice_models: Dict[str, Any] = {}
        
        # Speaker encoder for embedding extraction
        self.speaker_encoder = None
        
        # Voice cloning models
        self.voice_cloner = None
        self.few_shot_learner = None
        
        # Configuration
        # Handle both config.speaker and config.tts.speaker for compatibility
        speaker_config = getattr(config, 'speaker', None) or getattr(config, 'tts', {}).get('speaker', {})
        tts_config = getattr(config, 'tts', {})
        
        self.embedding_dim = speaker_config.get('embedding_dim', 256)
        self.min_audio_duration = speaker_config.get('min_audio_duration', 5.0)  # seconds
        self.max_audio_duration = speaker_config.get('max_audio_duration', 300.0)  # seconds
        self.sample_rate = speaker_config.get('sample_rate', 22050)
        
        # TTS audio parameters
        self.n_mels = tts_config.get('n_mels', 80)
        self.hop_length = tts_config.get('hop_length', 256)
        
        self._initialize_components()
        logger.info("Speaker manager initialized")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _initialize_components(self):
        """Initialize speaker management components."""
        try:
            # Initialize speaker encoder
            self.speaker_encoder = self._create_speaker_encoder()
            
            # Initialize voice cloner
            self.voice_cloner = self._create_voice_cloner()
            
            # Initialize few-shot learner
            self.few_shot_learner = self._create_few_shot_learner()
            
            # Load pre-defined speakers if available
            self._load_predefined_speakers()
            
        except Exception as e:
            logger.error(f"Failed to initialize speaker components: {str(e)}")
            raise RuntimeError(f"Speaker manager initialization failed: {str(e)}")
    
    def _create_speaker_encoder(self):
        """Create speaker embedding encoder."""
        logger.info("Creating speaker encoder...")
        
        class SpeakerEncoder(torch.nn.Module):
            def __init__(self, embedding_dim=256):
                super().__init__()
                self.embedding_dim = embedding_dim
                
                # Audio processing layers
                self.conv_layers = torch.nn.Sequential(
                    torch.nn.Conv1d(80, 128, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                    torch.nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                    torch.nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool1d(1)
                )
                
                # Embedding projection
                self.embedding_layer = torch.nn.Sequential(
                    torch.nn.Linear(512, embedding_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(embedding_dim, embedding_dim)
                )
                
                # Normalization
                self.instance_norm = torch.nn.InstanceNorm1d(80)
            
            def forward(self, mel_spec):
                """Extract speaker embedding from mel-spectrogram."""
                # Instance normalization
                mel_norm = self.instance_norm(mel_spec)
                
                # Convolutional feature extraction
                conv_features = self.conv_layers(mel_norm)
                conv_features = conv_features.squeeze(-1)  # Remove temporal dimension
                
                # Embedding projection
                embedding = self.embedding_layer(conv_features)
                
                # L2 normalization
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                return embedding
        
        model = SpeakerEncoder(self.embedding_dim)
        model.eval()
        if self.device.type == "cuda":
            model = model.cuda()
        
        return model
    
    def _create_voice_cloner(self):
        """Create voice cloning model for few-shot adaptation."""
        logger.info("Creating voice cloner...")
        
        class VoiceCloner(torch.nn.Module):
            def __init__(self, embedding_dim=256):
                super().__init__()
                self.embedding_dim = embedding_dim
                
                # Meta-learning layers for voice adaptation
                self.adaptation_network = torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, embedding_dim)
                )
                
                # Attention mechanism for few-shot learning
                self.attention = torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=8,
                    batch_first=True
                )
            
            def forward(self, support_embeddings, query_embedding):
                """
                Perform voice cloning using support set and query.
                
                Args:
                    support_embeddings: Speaker embeddings from support audio
                    query_embedding: Target speaker embedding to clone
                """
                # Apply attention mechanism
                attended_embedding, _ = self.attention(
                    query_embedding.unsqueeze(1),
                    support_embeddings.unsqueeze(0),
                    support_embeddings.unsqueeze(0)
                )
                
                # Adaptation network
                adapted_embedding = self.adaptation_network(attended_embedding.squeeze(1))
                
                # Combine with original
                final_embedding = 0.7 * query_embedding + 0.3 * adapted_embedding
                
                return final_embedding
        
        model = VoiceCloner(self.embedding_dim)
        model.eval()
        if self.device.type == "cuda":
            model = model.cuda()
        
        return model
    
    def _create_few_shot_learner(self):
        """Create few-shot learning model for voice adaptation."""
        logger.info("Creating few-shot learner...")
        
        class FewShotLearner(torch.nn.Module):
            def __init__(self, embedding_dim=256, n_way=5, k_shot=1):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.n_way = n_way
                self.k_shot = k_shot
                
                # Prototypical networks for few-shot learning
                self.prototype_encoder = torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, embedding_dim)
                )
                
                # Distance metric learning
                self.distance_metric = torch.nn.CosineSimilarity(dim=1)
            
            def compute_prototypes(self, support_embeddings, support_labels):
                """Compute class prototypes from support set."""
                n_classes = len(torch.unique(support_labels))
                prototypes = torch.zeros(n_classes, self.embedding_dim).to(support_embeddings.device)
                
                for class_idx in range(n_classes):
                    class_mask = support_labels == class_idx
                    class_embeddings = support_embeddings[class_mask]
                    prototypes[class_idx] = torch.mean(class_embeddings, dim=0)
                
                return prototypes
            
            def classify(self, query_embeddings, prototypes):
                """Classify query embeddings based on prototypes."""
                distances = []
                for prototype in prototypes:
                    dist = self.distance_metric(query_embeddings, prototype.unsqueeze(0))
                    distances.append(dist)
                
                distances = torch.stack(distances, dim=1)
                logits = torch.softmax(distances, dim=1)
                
                return logits
            
            def adapt_voice(self, support_embeddings, support_audio_features):
                """Adapt voice characteristics using few-shot learning."""
                # Encode support embeddings
                encoded_support = self.prototype_encoder(support_embeddings)
                
                # Compute adaptive weights
                weights = torch.softmax(encoded_support, dim=1)
                
                # Apply weights to audio features
                adapted_features = support_audio_features * weights.unsqueeze(-1)
                
                return adapted_features
        
        model = FewShotLearner(self.embedding_dim)
        model.eval()
        if self.device.type == "cuda":
            model = model.cuda()
        
        return model
    
    def _load_predefined_speakers(self):
        """Load predefined speaker models and embeddings."""
        logger.info("Loading predefined speakers...")
        
        # Placeholder predefined speakers
        predefined_speakers = [
            SpeakerInfo(
                speaker_id="yue_female_001",
                name="Cantonese Female 1",
                language="yue",
                gender="female",
                age_group="adult",
                metadata={"accent": "hong_kong", "style": "conversational"}
            ),
            SpeakerInfo(
                speaker_id="yue_male_001",
                name="Cantonese Male 1",
                language="yue",
                gender="male",
                age_group="adult",
                metadata={"accent": "guangzhou", "style": "formal"}
            ),
            SpeakerInfo(
                speaker_id="cmn_female_001",
                name="Mandarin Female 1",
                language="cmn",
                gender="female",
                age_group="adult",
                metadata={"accent": "beijing", "style": "news"}
            ),
            SpeakerInfo(
                speaker_id="cmn_male_001",
                name="Mandarin Male 1",
                language="cmn",
                gender="male",
                age_group="adult",
                metadata={"accent": "taiwan", "style": "conversational"}
            )
        ]
        
        # Add to speaker database
        for speaker in predefined_speakers:
            self.speakers[speaker.speaker_id] = speaker
            # Generate placeholder embeddings
            self.speaker_embeddings[speaker.speaker_id] = self._generate_placeholder_embedding()
        
        logger.info(f"Loaded {len(predefined_speakers)} predefined speakers")
    
    def _generate_placeholder_embedding(self) -> torch.Tensor:
        """Generate placeholder speaker embedding."""
        embedding = torch.randn(1, self.embedding_dim)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        if self.device.type == "cuda":
            embedding = embedding.cuda()
        
        return embedding
    
    def extract_speaker_embedding(self, audio_path: str, language: str = 'auto') -> torch.Tensor:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code or 'auto' for automatic detection
            
        Returns:
            Speaker embedding tensor
        """
        try:
            # Load and preprocess audio
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Convert to mel-spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            mel_spec = mel_transform(audio)
            
            # Take logarithm
            mel_spec = torch.log(mel_spec + 1e-8)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.speaker_encoder(mel_spec)
            
            logger.info(f"Extracted speaker embedding from {audio_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Speaker embedding extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract speaker embedding: {str(e)}")
    
    def clone_voice(self, reference_audio_paths: List[str], target_audio_path: str,
                   language: str = 'auto') -> torch.Tensor:
        """
        Clone voice characteristics from reference to target using few-shot learning.
        
        Args:
            reference_audio_paths: List of reference audio files
            target_audio_path: Target audio file to modify
            language: Language code
            
        Returns:
            Cloned speaker embedding
        """
        try:
            # Extract embeddings from reference audio
            reference_embeddings = []
            for ref_path in reference_audio_paths:
                ref_embedding = self.extract_speaker_embedding(ref_path, language)
                reference_embeddings.append(ref_embedding)
            
            # Stack reference embeddings
            reference_embeddings = torch.cat(reference_embeddings, dim=0)
            
            # Extract target embedding
            target_embedding = self.extract_speaker_embedding(target_audio_path, language)
            
            # Perform voice cloning
            with torch.no_grad():
                cloned_embedding = self.voice_cloner(reference_embeddings, target_embedding)
            
            logger.info(f"Voice cloning completed using {len(reference_audio_paths)} reference samples")
            return cloned_embedding
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            raise RuntimeError(f"Voice cloning failed: {str(e)}")
    
    def register_speaker(self, speaker_info: SpeakerInfo, audio_sample_path: str) -> str:
        """
        Register a new speaker with audio sample.
        
        Args:
            speaker_info: Speaker information
            audio_sample_path: Path to audio sample
            
        Returns:
            Speaker ID
        """
        try:
            # Validate audio duration
            audio_info = torchaudio.info(audio_sample_path)
            duration = audio_info.num_frames / audio_info.sample_rate
            
            if duration < self.min_audio_duration:
                raise ValueError(f"Audio too short: {duration:.1f}s < {self.min_audio_duration}s")
            
            if duration > self.max_audio_duration:
                raise ValueError(f"Audio too long: {duration:.1f}s > {self.max_audio_duration}s")
            
            # Extract speaker embedding
            embedding = self.extract_speaker_embedding(audio_sample_path, speaker_info.language)
            
            # Store speaker information
            speaker_info.embedding = embedding
            speaker_info.audio_sample_path = audio_sample_path
            
            self.speakers[speaker_info.speaker_id] = speaker_info
            self.speaker_embeddings[speaker_info.speaker_id] = embedding
            
            logger.info(f"Registered speaker: {speaker_info.speaker_id}")
            return speaker_info.speaker_id
            
        except Exception as e:
            logger.error(f"Speaker registration failed: {str(e)}")
            raise RuntimeError(f"Speaker registration failed: {str(e)}")
    
    def get_speaker_embedding(self, speaker_id: str) -> Optional[torch.Tensor]:
        """Get speaker embedding by ID."""
        return self.speaker_embeddings.get(speaker_id)
    
    def get_speaker_info(self, speaker_id: str) -> Optional[SpeakerInfo]:
        """Get speaker information by ID."""
        return self.speakers.get(speaker_id)
    
    def list_speakers(self, language: Optional[str] = None, 
                     gender: Optional[str] = None) -> List[SpeakerInfo]:
        """
        List speakers with optional filtering.
        
        Args:
            language: Filter by language
            gender: Filter by gender
            
        Returns:
            List of speaker information
        """
        speakers = list(self.speakers.values())
        
        if language:
            speakers = [s for s in speakers if s.language == language]
        
        if gender:
            speakers = [s for s in speakers if s.gender == gender]
        
        return speakers
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker from the database."""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
            del self.speaker_embeddings[speaker_id]
            logger.info(f"Deleted speaker: {speaker_id}")
            return True
        else:
            logger.warning(f"Speaker not found: {speaker_id}")
            return False
    
    def compare_speakers(self, speaker_id1: str, speaker_id2: str) -> float:
        """
        Compare similarity between two speakers.
        
        Args:
            speaker_id1: First speaker ID
            speaker_id2: Second speaker ID
            
        Returns:
            Similarity score (0.0-1.0)
        """
        embedding1 = self.get_speaker_embedding(speaker_id1)
        embedding2 = self.get_speaker_embedding(speaker_id2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)
        
        return similarity.item()
    
    def find_similar_speakers(self, reference_speaker_id: str, 
                            threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find speakers similar to reference speaker.
        
        Args:
            reference_speaker_id: Reference speaker ID
            threshold: Minimum similarity threshold
            
        Returns:
            List of (speaker_id, similarity) tuples
        """
        reference_embedding = self.get_speaker_embedding(reference_speaker_id)
        if reference_embedding is None:
            return []
        
        similar_speakers = []
        
        for speaker_id, embedding in self.speaker_embeddings.items():
            if speaker_id == reference_speaker_id:
                continue
            
            similarity = torch.nn.functional.cosine_similarity(
                reference_embedding, embedding, dim=1
            ).item()
            
            if similarity >= threshold:
                similar_speakers.append((speaker_id, similarity))
        
        # Sort by similarity (descending)
        similar_speakers.sort(key=lambda x: x[1], reverse=True)
        
        return similar_speakers
    
    def save_speaker_database(self, filepath: str):
        """Save speaker database to file."""
        try:
            database = {
                'speakers': self.speakers,
                'embeddings': {k: v.cpu() for k, v in self.speaker_embeddings.items()}
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(database, f)
            
            logger.info(f"Saved speaker database to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save speaker database: {str(e)}")
            raise RuntimeError(f"Failed to save speaker database: {str(e)}")
    
    def load_speaker_database(self, filepath: str):
        """Load speaker database from file."""
        try:
            with open(filepath, 'rb') as f:
                database = pickle.load(f)
            
            self.speakers = database['speakers']
            
            # Move embeddings to device
            for speaker_id, embedding in database['embeddings'].items():
                embedding = embedding.to(self.device)
                self.speaker_embeddings[speaker_id] = embedding
            
            logger.info(f"Loaded speaker database from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load speaker database: {str(e)}")
            raise RuntimeError(f"Failed to load speaker database: {str(e)}")
    
    def validate_audio_quality(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate audio quality for speaker extraction.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Validation results
        """
        try:
            audio_info = torchaudio.info(audio_path)
            
            # Load audio for analysis
            audio, sr = torchaudio.load(audio_path)
            
            # Basic quality metrics
            duration = audio_info.num_frames / audio_info.sample_rate
            channels = audio_info.num_channels
            
            # Audio quality analysis
            audio_mono = audio.mean(dim=0) if channels > 1 else audio[0]
            
            # Signal-to-noise ratio (simplified)
            snr = self._estimate_snr(audio_mono.numpy())
            
            # Clipping detection
            clipping_ratio = torch.sum(torch.abs(audio_mono) > 0.95).item() / len(audio_mono)
            
            # Dynamic range
            dynamic_range = torch.max(audio_mono).item() - torch.min(audio_mono).item()
            
            validation_results = {
                'duration': duration,
                'channels': channels,
                'sample_rate': audio_info.sample_rate,
                'snr_db': snr,
                'clipping_ratio': clipping_ratio,
                'dynamic_range': dynamic_range,
                'is_valid': (
                    duration >= self.min_audio_duration and
                    duration <= self.max_audio_duration and
                    snr > 20 and  # Good SNR
                    clipping_ratio < 0.01  # Minimal clipping
                )
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio (simplified)."""
        # Very simplified SNR estimation
        # In production, use proper noise estimation algorithms
        
        # Assume signal is the higher amplitude components
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise as the lower amplitude components
        sorted_audio = np.sort(np.abs(audio))
        noise_samples = int(len(sorted_audio) * 0.1)  # Bottom 10%
        noise_power = np.mean(sorted_audio[:noise_samples] ** 2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get speaker database statistics."""
        stats = {
            'total_speakers': len(self.speakers),
            'languages': {},
            'genders': {},
            'age_groups': {}
        }
        
        for speaker in self.speakers.values():
            # Language statistics
            if speaker.language not in stats['languages']:
                stats['languages'][speaker.language] = 0
            stats['languages'][speaker.language] += 1
            
            # Gender statistics
            if speaker.gender not in stats['genders']:
                stats['genders'][speaker.gender] = 0
            stats['genders'][speaker.gender] += 1
            
            # Age group statistics
            if speaker.age_group not in stats['age_groups']:
                stats['age_groups'][speaker.age_group] = 0
            stats['age_groups'][speaker.age_group] += 1
        
        return stats