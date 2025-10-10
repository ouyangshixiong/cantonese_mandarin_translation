"""
XTTS-v2 Integration Module for Cantonese-Mandarin-English Translation System

This module provides advanced text-to-speech synthesis capabilities using Coqui/XTTS-v2,
supporting bidirectional voice translation with speaker preservation and cross-lingual
voice conversion. Supports Cantonese, Mandarin, and English languages.
"""

from .xtts_v2_engine import XTTSV2Engine
from .prosody import ProsodyController
from .speaker_manager import SpeakerManager
from .audio_utils import AudioProcessor

__version__ = "3.0.0"
__all__ = [
    "XTTSV2Engine",
    "ProsodyController",
    "SpeakerManager",
    "AudioProcessor"
]