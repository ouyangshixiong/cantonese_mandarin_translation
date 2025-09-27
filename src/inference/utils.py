"""
Utility functions for inference module
"""
import argparse
import logging
from typing import Optional, Tuple


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cantonese-Mandarin translation inference")
    
    # Input configuration
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--input_file", type=str, help="Input file for batch translation")
    parser.add_argument("--output_file", type=str, help="Output file for batch translation")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Tencent-Hunyuan/Hunyuan-MT-7B",
                       help="Model name or path")
    parser.add_argument("--framework", type=str, choices=["pytorch", "paddle", "auto"], 
                       default="auto", help="Framework to use")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, gpu)")
    
    # Translation configuration
    parser.add_argument("--source_lang", type=str, default="cantonese",
                       choices=["cantonese", "mandarin"], help="Source language")
    parser.add_argument("--target_lang", type=str, default="mandarin", 
                       choices=["cantonese", "mandarin"], help="Target language")
    parser.add_argument("--beam_size", type=int, default=4,
                       help="Beam search size")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    
    # Output configuration
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    # TTS configuration (NEW)
    parser.add_argument("--enable_tts", action="store_true",
                       help="Enable text-to-speech synthesis")
    parser.add_argument("--speaker_id", type=str, default=None,
                       help="Speaker ID for TTS (e.g., yue_female_001, cmn_male_001)")
    parser.add_argument("--audio_format", type=str, default="wav",
                       choices=["wav", "mp3", "flac"], help="Audio output format")
    parser.add_argument("--emotion", type=str, default="neutral",
                       choices=["neutral", "happy", "sad", "angry", "excited", "calm"],
                       help="Emotion for TTS synthesis")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speech speed (0.5-2.0)")
    parser.add_argument("--pitch_shift", type=float, default=0.0,
                       help="Pitch shift in semitones (-6 to +6)")
    parser.add_argument("--voice_output_dir", type=str, default="./outputs/audio",
                       help="Directory for audio output files")
    parser.add_argument("--preserve_voice", action="store_true",
                       help="Preserve speaker voice across translation")
    
    return parser.parse_args()


def check_framework_availability() -> tuple[bool, bool]:
    """Check which frameworks are available"""
    pytorch_available = False
    paddle_available = False
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        pytorch_available = True
    except ImportError:
        pass
    
    try:
        import paddle
        paddle_available = True
    except ImportError:
        pass
    
    return pytorch_available, paddle_available