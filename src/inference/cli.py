"""
Command-line interface functions for inference
"""
import logging
import sys
from typing import List
from .translator import CantoneseTranslator

logger = logging.getLogger(__name__)


def interactive_mode(translator: CantoneseTranslator):
    """Interactive translation mode"""
    print("🈵 Cantonese-Mandarin Translation System")
    print("=" * 50)
    print("Commands:")
    print("  t <text> - Translate text")
    print("  i <text> - Translate from input")  
    print("  s - Switch translation direction")
    print("  m - Show model info")
    print("  q - Quit")
    print("=" * 50)
    
    current_source = "cantonese"
    current_target = "mandarin"
    
    while True:
        try:
            user_input = input(f"[{current_source}→{current_target}] > ").strip()
            
            if not user_input:
                continue
                
            command = user_input[0].lower()
            text = user_input[2:].strip() if len(user_input) > 2 else ""
            
            if command == "q":
                print("Goodbye!")
                break
            elif command == "t" and text:
                result = translator.translate(text, current_source, current_target)
                print(f"Translation: {result}")
            elif command == "i" and text:
                result = translator.translate(text, current_source, current_target)
                print(f"Input: {text}")
                print(f"Output: {result}")
            elif command == "s":
                current_source, current_target = current_target, current_source
                print(f"Switched to {current_source}→{current_target}")
            elif command == "m":
                info = translator.get_model_info()
                for key, value in info.items():
                    print(f"{key}: {value}")
            else:
                print("Invalid command. Type 't text' to translate or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def batch_translate_file(translator: CantoneseTranslator, input_file: str, 
                        output_file: str, args) -> None:
    """Batch translate file contents"""
    logger.info(f"Batch translating {input_file} to {output_file}")
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(texts)} texts to translate")
        
        # Batch translate
        results = translator.batch_translate(
            texts, 
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            beam_size=args.beam_size
        )
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
        
        logger.info(f"Translation completed. Results saved to {output_file}")
        
        # Print statistics
        success_count = sum(1 for r in results if r)
        logger.info(f"Successful translations: {success_count}/{len(texts)}")
        
    except Exception as e:
        logger.error(f"Batch translation failed: {str(e)}")
        raise


def handle_single_translation(translator: CantoneseTranslator, text: str, args) -> None:
    """Handle single text translation"""
    try:
        result = translator.translate(
            text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            beam_size=args.beam_size
        )
        print(f"Input: {text}")
        print(f"Translation: {result}")
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise


def show_model_info(translator: CantoneseTranslator) -> None:
    """Display model information"""
    info = translator.get_model_info()
    logger.info("Model information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")