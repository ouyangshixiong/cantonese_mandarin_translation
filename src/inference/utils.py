"""
Utility functions for inference module
"""
import argparse
import logging
from typing import Optional


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