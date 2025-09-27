"""
Inference module for Cantonese-Mandarin translation
"""

from .translator import CantoneseTranslator
from .utils import parse_args, setup_logging

__all__ = ['CantoneseTranslator', 'parse_args', 'setup_logging']