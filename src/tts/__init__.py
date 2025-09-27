"""
Index-TTS2 Integration Module for Cantonese-Mandarin Translation System

This module provides advanced text-to-speech synthesis capabilities using Index-TTS2,
supporting bidirectional voice translation with speaker preservation and cross-lingual
voice conversion.
"""

from .index_tts2 import IndexTTS2Engine
from .voice_converter import CrossLingualVoiceConverter
from .prosody import ProsodyController
from .speaker_manager import SpeakerManager
from .audio_utils import AudioProcessor

__version__ = "2.0.0"
__all__ = [
    "IndexTTS2Engine",
    "CrossLingualVoiceConverter", 
    "ProsodyController",
    "SpeakerManager",
    "AudioProcessor"
]