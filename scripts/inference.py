#!/usr/bin/env python3
"""
Inference script for Cantonese-Mandarin translation using Hunyuan-MT-7B
Supports both PyTorch and PaddlePaddle frameworks with FastAPI service
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.translator import CantoneseTranslator
from inference.utils import parse_args, setup_logging, check_framework_availability
from inference.cli import interactive_mode, batch_translate_file, handle_single_translation, show_model_info

logger = logging.getLogger(__name__)


def main():
    """Main inference function"""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Check framework availability
        pytorch_available, paddle_available = check_framework_availability()
        
        # Auto-select framework if needed
        if args.framework == "auto":
            if pytorch_available:
                args.framework = "pytorch"
            elif paddle_available:
                args.framework = "paddle"
            else:
                logger.error("No suitable framework available")
                sys.exit(1)
        
        # Initialize translator
        logger.info(f"Initializing translator with {args.framework} framework...")
        translator = CantoneseTranslator(
            model_name=args.model_name,
            framework=args.framework,
            device=args.device
        )
        
        # Show model info if verbose
        if args.verbose:
            show_model_info(translator)
        
        # Handle different input modes
        if args.text:
            # Single text translation
            handle_single_translation(translator, args.text, args)
            
        elif args.input_file and args.output_file:
            # Batch file translation
            batch_translate_file(translator, args.input_file, args.output_file, args)
            
        else:
            # Interactive mode
            interactive_mode(translator)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()