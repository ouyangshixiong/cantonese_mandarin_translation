#!/usr/bin/env python3
"""
Enhanced Inference script with Index-TTS2 integration for Cantonese-Mandarin translation
Supports text translation with speech synthesis, voice cloning, and cross-lingual voice conversion
"""
import logging
import sys
import time
from pathlib import Path
import torch
import numpy as np
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.translator import CantoneseTranslator
from inference.utils import parse_args, setup_logging, check_framework_availability
from inference.cli import interactive_mode, batch_translate_file, handle_single_translation, show_model_info
from tts import IndexTTS2Engine, CrossLingualVoiceConverter, SpeakerManager, AudioProcessor
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class VoiceTranslationSystem:
    """
    Enhanced translation system with Index-TTS2 speech synthesis capabilities.
    Supports bidirectional translation with voice output and cross-lingual voice conversion.
    """
    
    def __init__(self, config, device: str = "auto"):
        """
        Initialize voice translation system.
        
        Args:
            config: Configuration object
            device: Target device for computation
        """
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize components
        self.translator = None
        self.tts_engine = None
        self.voice_converter = None
        self.speaker_manager = None
        self.audio_processor = None
        
        self._initialize_components()
        logger.info("Voice translation system initialized")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        
        return torch.device(device)
    
    def _initialize_components(self):
        """Initialize translation and TTS components."""
        try:
            # Initialize translator
            logger.info("Initializing translator...")
            self.translator = CantoneseTranslator(
                model_name=self.config.model.name,
                framework=self.config.framework.name,
                device=str(self.device)
            )
            
            # Initialize TTS engine if enabled
            if hasattr(self.config, 'tts') and self.config.tts.enabled:
                logger.info("Initializing Index-TTS2 engine...")
                self.tts_engine = IndexTTS2Engine(self.config, str(self.device))
                
                # Initialize voice converter
                logger.info("Initializing voice converter...")
                self.voice_converter = CrossLingualVoiceConverter(self.config, str(self.device))
                
                # Initialize speaker manager
                logger.info("Initializing speaker manager...")
                self.speaker_manager = SpeakerManager(self.config, str(self.device))
                
                # Initialize audio processor
                logger.info("Initializing audio processor...")
                self.audio_processor = AudioProcessor(
                    sample_rate=self.config.tts.sample_rate,
                    n_mels=self.config.tts.n_mels,
                    hop_length=self.config.tts.hop_length,
                    win_length=self.config.tts.win_length
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise RuntimeError(f"Component initialization failed: {str(e)}")
    
    def translate_with_voice(self, text: str, source_lang: str, target_lang: str,
                           speaker_id: Optional[str] = None, emotion: str = "neutral",
                           speed: float = 1.0, pitch_shift: float = 0.0,
                           preserve_voice: bool = False) -> Tuple[str, Optional[np.ndarray], int]:
        """
        Translate text and synthesize speech with Index-TTS2.
        
        Args:
            text: Input text to translate
            source_lang: Source language
            target_lang: Target language
            speaker_id: Speaker ID for TTS
            emotion: Emotion for speech synthesis
            speed: Speech speed multiplier
            pitch_shift: Pitch shift in semitones
            preserve_voice: Whether to preserve speaker voice across translation
            
        Returns:
            Tuple of (translated_text, audio_array, sample_rate)
        """
        try:
            # Perform text translation
            start_time = time.time()
            translated_text = self.translator.translate(text, source_lang, target_lang)
            translation_time = time.time() - start_time
            
            logger.info(f"Translation completed in {translation_time:.3f}s")
            
            # Synthesize speech if TTS is enabled
            audio_array = None
            sample_rate = None
            
            if self.tts_engine and speaker_id:
                # Determine language for TTS
                tts_language = self._get_tts_language(target_lang)
                
                # Get speaker embedding
                speaker_embedding = self.speaker_manager.get_speaker_embedding(speaker_id)
                if speaker_embedding is None:
                    # Use default speaker if not found
                    default_speaker = (self.config.tts.default_speaker_yue if tts_language == 'yue'
                                     else self.config.tts.default_speaker_cmn)
                    speaker_id = default_speaker
                    logger.warning(f"Speaker {speaker_id} not found, using default: {default_speaker}")
                
                # Handle cross-lingual voice preservation
                if preserve_voice and source_lang != target_lang:
                    audio_array, sample_rate = self._synthesize_with_voice_preservation(
                        translated_text, text, source_lang, target_lang, speaker_id,
                        emotion, speed, pitch_shift
                    )
                else:
                    # Standard TTS synthesis
                    start_time = time.time()
                    audio_array, sample_rate = self.tts_engine.synthesize(
                        text=translated_text,
                        language=tts_language,
                        speaker_id=speaker_id,
                        emotion=emotion,
                        speed=speed,
                        pitch_shift=pitch_shift
                    )
                    synthesis_time = time.time() - start_time
                    logger.info(f"TTS synthesis completed in {synthesis_time:.3f}s")
            
            return translated_text, audio_array, sample_rate
            
        except Exception as e:
            logger.error(f"Voice translation failed: {str(e)}")
            raise RuntimeError(f"Voice translation failed: {str(e)}")
    
    def _get_tts_language(self, lang: str) -> str:
        """Convert language name to TTS language code."""
        lang_map = {
            "cantonese": "yue",
            "mandarin": "cmn"
        }
        return lang_map.get(lang, "cmn")
    
    def _synthesize_with_voice_preservation(self, translated_text: str, original_text: str,
                                          source_lang: str, target_lang: str,
                                          speaker_id: str, emotion: str, speed: float,
                                          pitch_shift: float) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech with cross-lingual voice preservation.
        
        Args:
            translated_text: Translated text
            original_text: Original source text
            source_lang: Source language
            target_lang: Target language
            speaker_id: Speaker ID
            emotion: Emotion for synthesis
            speed: Speech speed
            pitch_shift: Pitch shift
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            logger.info(f"Performing cross-lingual voice preservation: {source_lang} → {target_lang}")
            
            # First synthesize original speech
            source_tts_lang = self._get_tts_language(source_lang)
            original_audio, sr = self.tts_engine.synthesize(
                text=original_text,
                language=source_tts_lang,
                speaker_id=speaker_id,
                emotion=emotion,
                speed=speed,
                pitch_shift=pitch_shift
            )
            
            # Perform voice conversion to target language
            target_tts_lang = self._get_tts_language(target_lang)
            converted_audio = self.voice_converter.convert_voice(
                source_audio=original_audio,
                source_language=source_tts_lang,
                target_language=target_tts_lang,
                reference_speaker=None  # Use same speaker characteristics
            )
            
            logger.info("Cross-lingual voice conversion completed")
            return converted_audio, sr
            
        except Exception as e:
            logger.error(f"Voice preservation failed: {str(e)}")
            # Fallback to standard TTS
            target_tts_lang = self._get_tts_language(target_lang)
            return self.tts_engine.synthesize(
                text=translated_text,
                language=target_tts_lang,
                speaker_id=speaker_id,
                emotion=emotion,
                speed=speed,
                pitch_shift=pitch_shift
            )
    
    def save_audio(self, audio_array: np.ndarray, sample_rate: int, 
                   output_path: str, audio_format: str = "wav") -> str:
        """
        Save audio array to file.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate
            output_path: Output file path
            audio_format: Audio format
            
        Returns:
            Path to saved file
        """
        try:
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio_array).float()
            
            # Ensure proper shape (channels, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Save audio
            success = self.audio_processor.save_audio(
                audio=audio_tensor,
                filepath=output_path,
                sample_rate=sample_rate,
                format=audio_format
            )
            
            if success:
                logger.info(f"Audio saved to: {output_path}")
                return output_path
            else:
                raise RuntimeError("Failed to save audio")
                
        except Exception as e:
            logger.error(f"Audio saving failed: {str(e)}")
            raise RuntimeError(f"Audio saving failed: {str(e)}")
    
    def get_available_speakers(self, language: str) -> list:
        """Get available speakers for a language."""
        if not self.speaker_manager:
            return []
        
        speakers = self.speaker_manager.list_speakers(language=language)
        return [s.speaker_id for s in speakers]
    
    def get_model_info(self) -> dict:
        """Get system information."""
        info = {
            "translator_model": self.config.model.name,
            "framework": self.config.framework.name,
            "device": str(self.device),
            "tts_enabled": hasattr(self.config, 'tts') and self.config.tts.enabled
        }
        
        if info["tts_enabled"]:
            info.update({
                "available_speakers_yue": self.get_available_speakers("yue"),
                "available_speakers_cmn": self.get_available_speakers("cmn"),
                "supported_emotions": self.config.tts.prosody.emotions,
                "sample_rate": self.config.tts.sample_rate
            })
        
        return info


def handle_single_voice_translation(voice_system: VoiceTranslationSystem, text: str, args):
    """Handle single text translation with voice synthesis."""
    try:
        logger.info(f"Processing: '{text[:50]}...'")
        
        # Perform voice translation
        translated_text, audio_array, sample_rate = voice_system.translate_with_voice(
            text=text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            speaker_id=args.speaker_id,
            emotion=args.emotion,
            speed=args.speed,
            pitch_shift=args.pitch_shift,
            preserve_voice=args.preserve_voice
        )
        
        # Display results
        print(f"\n📝 Translation:")
        print(f"Input ({args.source_lang}): {text}")
        print(f"Output ({args.target_lang}): {translated_text}")
        
        # Save audio if generated
        if audio_array is not None and args.voice_output_dir:
            output_dir = Path(args.voice_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time() * 1000)
            filename = f"tts_{args.source_lang}_to_{args.target_lang}_{timestamp}.{args.audio_format}"
            output_path = output_dir / filename
            
            # Save audio
            saved_path = voice_system.save_audio(
                audio_array=audio_array,
                sample_rate=sample_rate,
                output_path=str(output_path),
                audio_format=args.audio_format
            )
            
            print(f"\n🔊 Audio saved: {saved_path}")
            print(f"Duration: {len(audio_array) / sample_rate:.2f}s")
            print(f"Sample rate: {sample_rate}Hz")
            
    except Exception as e:
        logger.error(f"Voice translation failed: {str(e)}")
        print(f"Error: {str(e)}")


def interactive_voice_mode(voice_system: VoiceTranslationSystem):
    """Interactive mode with voice synthesis."""
    print("🈵 Cantonese-Mandarin Voice Translation System")
    print("=" * 60)
    print("Commands:")
    print("  t <text> - Translate text")
    print("  v <text> - Translate with voice synthesis")
    print("  s - Switch translation direction")
    print("  l - List available speakers")
    print("  m - Show model info")
    print("  q - Quit")
    print("=" * 60)
    
    current_source = "cantonese"
    current_target = "mandarin"
    current_speaker = None
    
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
                # Text-only translation
                result = voice_system.translator.translate(text, current_source, current_target)
                print(f"Translation: {result}")
                
            elif command == "v" and text:
                # Voice translation
                if not voice_system.tts_engine:
                    print("TTS not available. Please enable --enable_tts")
                    continue
                
                # Auto-select speaker if not set
                tts_lang = "yue" if current_target == "cantonese" else "cmn"
                if current_speaker is None:
                    available_speakers = voice_system.get_available_speakers(tts_lang)
                    if available_speakers:
                        current_speaker = available_speakers[0]
                        print(f"Using speaker: {current_speaker}")
                
                # Perform voice translation
                translated_text, audio_array, sample_rate = voice_system.translate_with_voice(
                    text=text,
                    source_lang=current_source,
                    target_lang=current_target,
                    speaker_id=current_speaker
                )
                
                print(f"Translation: {translated_text}")
                
                if audio_array is not None:
                    print(f"Audio generated: {len(audio_array) / sample_rate:.2f}s")
                    
            elif command == "s":
                current_source, current_target = current_target, current_source
                print(f"Switched to {current_source}→{current_target}")
                current_speaker = None  # Reset speaker
                
            elif command == "l":
                # List available speakers
                if voice_system.tts_engine:
                    yue_speakers = voice_system.get_available_speakers("yue")
                    cmn_speakers = voice_system.get_available_speakers("cmn")
                    print(f"Cantonese speakers: {yue_speakers}")
                    print(f"Mandarin speakers: {cmn_speakers}")
                else:
                    print("TTS not available")
                    
            elif command == "m":
                # Show model info
                info = voice_system.get_model_info()
                print("Model Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                    
        except KeyboardInterrupt:
            print("\nUse 'q' to quit")
        except Exception as e:
            logger.error(f"Interactive mode error: {str(e)}")
            print(f"Error: {str(e)}")


def main():
    """Main inference function with TTS support."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        config = OmegaConf.load(config_path)
        
        # Override config with command line arguments
        if args.enable_tts:
            config.tts.enabled = True
        
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
        
        # Initialize voice translation system
        logger.info("Initializing voice translation system...")
        voice_system = VoiceTranslationSystem(config, args.device)
        
        # Show model info if verbose
        if args.verbose:
            info = voice_system.get_model_info()
            print("System Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # Handle different input modes
        if args.text:
            if args.enable_tts:
                # Voice translation mode
                handle_single_voice_translation(voice_system, args.text, args)
            else:
                # Standard text translation
                from inference.cli import handle_single_translation
                handle_single_translation(voice_system.translator, args.text, args)
                
        elif args.input_file and args.output_file:
            # Batch translation (text only for now)
            logger.warning("Batch voice translation not yet implemented, using text mode")
            from inference.cli import batch_translate_file
            batch_translate_file(voice_system.translator, args.input_file, args.output_file, args)
            
        else:
            # Interactive mode
            if args.enable_tts:
                interactive_voice_mode(voice_system)
            else:
                from inference.cli import interactive_mode
                interactive_mode(voice_system.translator)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Voice inference failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()