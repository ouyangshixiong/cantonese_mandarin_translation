# Model registry and factory for Cantonese-Mandarin translation
from .pytorch.cantonese_translation import CantoneseTranslationModule

# Try to import PaddlePaddle model
try:
    from .paddle.cantonese_translation import CantoneseTranslationModel
    PADDLE_MODEL_AVAILABLE = True
except ImportError:
    PADDLE_MODEL_AVAILABLE = False
    CantoneseTranslationModel = None

__all__ = ['CantoneseTranslationModule']

if PADDLE_MODEL_AVAILABLE:
    __all__.append('CantoneseTranslationModel')