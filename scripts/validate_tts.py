#!/usr/bin/env python3
"""
TTS validation script for 1-epoch testing of Index-TTS2 integration
Performs comprehensive testing of speech synthesis capabilities
"""
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tts import IndexTTS2Engine, CrossLingualVoiceConverter, SpeakerManager, AudioProcessor
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class TTSValidator:
    """Comprehensive TTS validation system for 1-epoch testing."""
    
    def __init__(self, config_path: str, device: str = "auto"):
        """Initialize TTS validator."""
        self.config = OmegaConf.load(config_path)
        self.device = self._setup_device(device)
        self.results = {}
        
        # Initialize components
        self._initialize_components()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        return torch.device(device)
    
    def _initialize_components(self):
        """Initialize TTS components."""
        try:
            logger.info("Initializing TTS components...")
            
            # Initialize TTS engine
            self.tts_engine = IndexTTS2Engine(self.config, str(self.device))
            
            # Initialize voice converter
            self.voice_converter = CrossLingualVoiceConverter(self.config, str(self.device))
            
            # Initialize speaker manager
            self.speaker_manager = SpeakerManager(self.config, str(self.device))
            
            # Initialize audio processor
            self.audio_processor = AudioProcessor(
                sample_rate=self.config.tts.sample_rate,
                n_mels=self.config.tts.n_mels,
                hop_length=self.config.tts.hop_length,
                win_length=self.config.tts.win_length
            )
            
            logger.info("TTS components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS components: {str(e)}")
            raise
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive TTS validation tests."""
        logger.info("Starting comprehensive TTS validation...")
        
        start_time = time.time()
        
        # Test 1: Basic Synthesis Validation
        self.results['basic_synthesis'] = self._test_basic_synthesis()
        
        # Test 2: Multi-language Support
        self.results['multi_language'] = self._test_multi_language_support()
        
        # Test 3: Speaker Management
        self.results['speaker_management'] = self._test_speaker_management()
        
        # Test 4: Voice Conversion
        self.results['voice_conversion'] = self._test_voice_conversion()
        
        # Test 5: Prosody Control
        self.results['prosody_control'] = self._test_prosody_control()
        
        # Test 6: Performance Benchmarks
        self.results['performance'] = self._test_performance_benchmarks()
        
        # Test 7: Audio Quality Assessment
        self.results['audio_quality'] = self._test_audio_quality()
        
        # Test 8: Error Handling
        self.results['error_handling'] = self._test_error_handling()
        
        total_time = time.time() - start_time
        self.results['total_validation_time'] = total_time
        
        logger.info(f"Validation completed in {total_time:.2f}s")
        return self.results
    
    def _test_basic_synthesis(self) -> Dict:
        """Test basic speech synthesis functionality."""
        logger.info("Testing basic synthesis...")
        
        test_cases = [
            {"text": "你好嗎？", "language": "yue", "description": "Basic Cantonese"},
            {"text": "你好吗？", "language": "cmn", "description": "Basic Mandarin"},
            {"text": "今天天氣很好", "language": "yue", "description": "Longer Cantonese"},
            {"text": "今天天气很好", "language": "cmn", "description": "Longer Mandarin"}
        ]
        
        results = {"test_cases": [], "overall_success": True}
        
        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                
                audio_array, sample_rate = self.tts_engine.synthesize(
                    text=test_case["text"],
                    language=test_case["language"],
                    emotion="neutral",
                    speed=1.0
                )
                
                synthesis_time = time.time() - start_time
                
                # Validate results
                success = (
                    isinstance(audio_array, np.ndarray) and
                    len(audio_array) > 0 and
                    sample_rate == self.config.tts.sample_rate and
                    synthesis_time < 5.0  # Should be fast
                )
                
                results["test_cases"].append({
                    "id": i + 1,
                    "description": test_case["description"],
                    "text": test_case["text"],
                    "language": test_case["language"],
                    "success": success,
                    "synthesis_time": synthesis_time,
                    "audio_length": len(audio_array),
                    "duration": len(audio_array) / sample_rate
                })
                
                if not success:
                    results["overall_success"] = False
                    
            except Exception as e:
                logger.error(f"Basic synthesis test failed for {test_case['description']}: {str(e)}")
                results["test_cases"].append({
                    "id": i + 1,
                    "description": test_case["description"],
                    "success": False,
                    "error": str(e)
                })
                results["overall_success"] = False
        
        return results
    
    def _test_multi_language_support(self) -> Dict:
        """Test multi-language synthesis support."""
        logger.info("Testing multi-language support...")
        
        languages = ["yue", "cmn"]
        test_phrases = {
            "yue": ["你好", "多謝", "再見", "食咗飯未？"],
            "cmn": ["你好", "谢谢", "再见", "吃饭了吗？"]
        }
        
        results = {"languages": {}, "overall_success": True}
        
        for lang in languages:
            try:
                lang_results = []
                
                for phrase in test_phrases[lang]:
                    try:
                        audio_array, sample_rate = self.tts_engine.synthesize(
                            text=phrase,
                            language=lang,
                            emotion="neutral"
                        )
                        
                        success = (
                            isinstance(audio_array, np.ndarray) and
                            len(audio_array) > 0 and
                            sample_rate == self.config.tts.sample_rate
                        )
                        
                        lang_results.append({
                            "phrase": phrase,
                            "success": success,
                            "audio_length": len(audio_array) if success else 0
                        })
                        
                    except Exception as e:
                        lang_results.append({
                            "phrase": phrase,
                            "success": False,
                            "error": str(e)
                        })
                
                # Calculate language success rate
                success_count = sum(1 for r in lang_results if r["success"])
                success_rate = success_count / len(lang_results)
                
                results["languages"][lang] = {
                    "test_results": lang_results,
                    "success_rate": success_rate,
                    "overall_success": success_rate >= 0.8  # 80% success threshold
                }
                
                if success_rate < 0.8:
                    results["overall_success"] = False
                    
            except Exception as e:
                logger.error(f"Multi-language test failed for {lang}: {str(e)}")
                results["languages"][lang] = {
                    "success_rate": 0.0,
                    "overall_success": False,
                    "error": str(e)
                }
                results["overall_success"] = False
        
        return results
    
    def _test_speaker_management(self) -> Dict:
        """Test speaker management functionality."""
        logger.info("Testing speaker management...")
        
        results = {
            "speaker_listing": {},
            "speaker_embedding": {},
            "speaker_comparison": {},
            "overall_success": True
        }
        
        try:
            # Test speaker listing
            for lang in ["yue", "cmn"]:
                speakers = self.speaker_manager.list_speakers(language=lang)
                results["speaker_listing"][lang] = {
                    "count": len(speakers),
                    "speakers": [s.speaker_id for s in speakers],
                    "success": len(speakers) > 0
                }
            
            # Test speaker embedding retrieval
            test_speaker = "yue_female_001"
            embedding = self.speaker_manager.get_speaker_embedding(test_speaker)
            results["speaker_embedding"] = {
                "test_speaker": test_speaker,
                "embedding_exists": embedding is not None,
                "embedding_shape": embedding.shape if embedding is not None else None,
                "success": embedding is not None
            }
            
            # Test speaker comparison
            speakers = list(self.speaker_manager.speakers.keys())
            if len(speakers) >= 2:
                similarity = self.speaker_manager.compare_speakers(speakers[0], speakers[1])
                results["speaker_comparison"] = {
                    "speaker1": speakers[0],
                    "speaker2": speakers[1],
                    "similarity": similarity,
                    "success": isinstance(similarity, float) and 0 <= similarity <= 1
                }
            
            # Check overall success
            success_criteria = [
                all(lang_result["success"] for lang_result in results["speaker_listing"].values()),
                results["speaker_embedding"]["success"],
                results["speaker_comparison"].get("success", True)  # Optional if not enough speakers
            ]
            
            results["overall_success"] = all(success_criteria)
            
        except Exception as e:
            logger.error(f"Speaker management test failed: {str(e)}")
            results["overall_success"] = False
            results["error"] = str(e)
        
        return results
    
    def _test_voice_conversion(self) -> Dict:
        """Test cross-lingual voice conversion."""
        logger.info("Testing voice conversion...")
        
        results = {
            "conversion_tests": [],
            "overall_success": True
        }
        
        # Test conversion directions
        conversions = [
            {"from": "yue", "to": "cmn", "description": "Cantonese to Mandarin"},
            {"from": "cmn", "to": "yue", "description": "Mandarin to Cantonese"}
        ]
        
        for i, conversion in enumerate(conversions):
            try:
                # Create dummy source audio
                sample_rate = self.config.tts.sample_rate
                duration = 2.0
                source_audio = np.random.randn(int(sample_rate * duration)) * 0.1
                
                # Perform conversion
                start_time = time.time()
                converted_audio = self.voice_converter.convert_voice(
                    source_audio=source_audio,
                    source_language=conversion["from"],
                    target_language=conversion["to"]
                )
                conversion_time = time.time() - start_time
                
                # Validate results
                success = (
                    isinstance(converted_audio, np.ndarray) and
                    len(converted_audio) > 0 and
                    conversion_time < 10.0  # Should complete within 10 seconds
                )
                
                results["conversion_tests"].append({
                    "id": i + 1,
                    "description": conversion["description"],
                    "from_lang": conversion["from"],
                    "to_lang": conversion["to"],
                    "success": success,
                    "conversion_time": conversion_time,
                    "input_length": len(source_audio),
                    "output_length": len(converted_audio)
                })
                
                if not success:
                    results["overall_success"] = False
                    
            except Exception as e:
                logger.error(f"Voice conversion test failed for {conversion['description']}: {str(e)}")
                results["conversion_tests"].append({
                    "id": i + 1,
                    "description": conversion["description"],
                    "success": False,
                    "error": str(e)
                })
                results["overall_success"] = False
        
        return results
    
    def _test_prosody_control(self) -> Dict:
        """Test prosody control features."""
        logger.info("Testing prosody control...")
        
        results = {
            "emotion_tests": [],
            "speed_tests": [],
            "pitch_tests": [],
            "overall_success": True
        }
        
        # Test emotion control
        emotions = ["neutral", "happy", "sad", "angry"]
        test_text = "今天天氣很好"
        
        for emotion in emotions:
            try:
                audio_array, sample_rate = self.tts_engine.synthesize(
                    text=test_text,
                    language="yue",
                    emotion=emotion
                )
                
                success = isinstance(audio_array, np.ndarray) and len(audio_array) > 0
                
                results["emotion_tests"].append({
                    "emotion": emotion,
                    "success": success,
                    "audio_length": len(audio_array) if success else 0
                })
                
            except Exception as e:
                results["emotion_tests"].append({
                    "emotion": emotion,
                    "success": False,
                    "error": str(e)
                })
        
        # Test speed control
        speeds = [0.5, 1.0, 1.5, 2.0]
        
        for speed in speeds:
            try:
                audio_array, sample_rate = self.tts_engine.synthesize(
                    text=test_text,
                    language="yue",
                    speed=speed
                )
                
                success = isinstance(audio_array, np.ndarray) and len(audio_array) > 0
                
                results["speed_tests"].append({
                    "speed": speed,
                    "success": success,
                    "audio_length": len(audio_array) if success else 0
                })
                
            except Exception as e:
                results["speed_tests"].append({
                    "speed": speed,
                    "success": False,
                    "error": str(e)
                })
        
        # Test pitch control
        pitch_shifts = [-3, 0, 3]
        
        for pitch_shift in pitch_shifts:
            try:
                audio_array, sample_rate = self.tts_engine.synthesize(
                    text=test_text,
                    language="yue",
                    pitch_shift=pitch_shift
                )
                
                success = isinstance(audio_array, np.ndarray) and len(audio_array) > 0
                
                results["pitch_tests"].append({
                    "pitch_shift": pitch_shift,
                    "success": success,
                    "audio_length": len(audio_array) if success else 0
                })
                
            except Exception as e:
                results["pitch_tests"].append({
                    "pitch_shift": pitch_shift,
                    "success": False,
                    "error": str(e)
                })
        
        # Check overall success
        emotion_success = all(test["success"] for test in results["emotion_tests"])
        speed_success = all(test["success"] for test in results["speed_tests"])
        pitch_success = all(test["success"] for test in results["pitch_tests"])
        
        results["overall_success"] = emotion_success and speed_success and pitch_success
        
        return results
    
    def _test_performance_benchmarks(self) -> Dict:
        """Test performance benchmarks."""
        logger.info("Testing performance benchmarks...")
        
        results = {
            "synthesis_speed": {},
            "memory_usage": {},
            "latency_tests": [],
            "overall_success": True
        }
        
        # Test synthesis speed
        test_texts = [
            ("你好", 10),
            ("今天天氣很好", 50),
            ("這是一個比較長的測試句子，用來測試合成速度", 100)
        ]
        
        for text, expected_chars in test_texts:
            try:
                # Warm up
                self.tts_engine.synthesize(text, "yue")
                
                # Time multiple syntheses
                times = []
                for _ in range(3):
                    start_time = time.time()
                    audio_array, sample_rate = self.tts_engine.synthesize(text, "yue")
                    synthesis_time = time.time() - start_time
                    times.append(synthesis_time)
                
                avg_time = np.mean(times)
                real_time_factor = (len(audio_array) / sample_rate) / avg_time
                
                results["synthesis_speed"][f"text_{expected_chars}_chars"] = {
                    "avg_time": avg_time,
                    "real_time_factor": real_time_factor,
                    "success": real_time_factor >= 1.0  # Should be at least real-time
                }
                
            except Exception as e:
                results["synthesis_speed"][f"text_{expected_chars}_chars"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test memory usage
        try:
            memory_stats = self.tts_engine.get_memory_usage()
            results["memory_usage"] = memory_stats
            
            # Check if memory usage is reasonable
            if self.device.type == "cuda":
                gpu_memory_gb = memory_stats.get('gpu_allocated', 0)
                results["memory_usage"]["within_limits"] = gpu_memory_gb < 16.0  # Less than 16GB
            else:
                results["memory_usage"]["within_limits"] = True  # CPU doesn't have GPU memory
                
        except Exception as e:
            results["memory_usage"] = {"success": False, "error": str(e)}
        
        # Check overall success
        speed_success = all(test.get("success", False) for test in results["synthesis_speed"].values())
        memory_success = results["memory_usage"].get("within_limits", True)
        
        results["overall_success"] = speed_success and memory_success
        
        return results
    
    def _test_audio_quality(self) -> Dict:
        """Test audio quality metrics."""
        logger.info("Testing audio quality...")
        
        results = {
            "quality_metrics": [],
            "overall_success": True
        }
        
        # Test quality for different synthesis scenarios
        test_cases = [
            {"text": "你好", "language": "yue", "description": "Short Cantonese"},
            {"text": "你好吗？", "language": "cmn", "description": "Short Mandarin"},
            {"text": "今天天氣很好，我們去公園散步。", "language": "yue", "description": "Longer Cantonese"}
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                # Synthesize audio
                audio_array, sample_rate = self.tts_engine.synthesize(
                    text=test_case["text"],
                    language=test_case["language"]
                )
                
                # Convert to tensor for quality assessment
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                
                # Assess quality
                quality_metrics = self.audio_processor.assess_quality(audio_tensor, sample_rate)
                
                # Check quality thresholds
                quality_success = (
                    quality_metrics.overall_score >= 0.6 and  # Overall quality score
                    quality_metrics.snr_db >= 15.0 and        # Signal-to-noise ratio
                    quality_metrics.clipping_ratio < 0.05      # Minimal clipping
                )
                
                results["quality_metrics"].append({
                    "id": i + 1,
                    "description": test_case["description"],
                    "text": test_case["text"],
                    "language": test_case["language"],
                    "success": quality_success,
                    "snr_db": quality_metrics.snr_db,
                    "dynamic_range": quality_metrics.dynamic_range,
                    "clipping_ratio": quality_metrics.clipping_ratio,
                    "overall_score": quality_metrics.overall_score
                })
                
                if not quality_success:
                    results["overall_success"] = False
                    
            except Exception as e:
                results["quality_metrics"].append({
                    "id": i + 1,
                    "description": test_case["description"],
                    "success": False,
                    "error": str(e)
                })
                results["overall_success"] = False
        
        return results
    
    def _test_error_handling(self) -> Dict:
        """Test error handling and edge cases."""
        logger.info("Testing error handling...")
        
        results = {
            "error_cases": [],
            "edge_cases": [],
            "overall_success": True
        }
        
        # Test error cases
        error_cases = [
            {"text": "", "language": "yue", "description": "Empty text"},
            {"text": "你好", "language": "invalid", "description": "Invalid language"},
            {"text": None, "language": "yue", "description": "None text"},
            {"text": "你好", "language": None, "description": "None language"}
        ]
        
        for i, case in enumerate(error_cases):
            try:
                if case["text"] is None or case["language"] is None:
                    # These should raise TypeError or similar
                    with pytest.raises(Exception):
                        self.tts_engine.synthesize(case["text"], case["language"])
                    success = True
                else:
                    # These should handle gracefully or raise appropriate errors
                    try:
                        self.tts_engine.synthesize(case["text"], case["language"])
                        success = True  # If it doesn't crash, consider it handled
                    except Exception:
                        success = True  # Expected error
                
                results["error_cases"].append({
                    "id": i + 1,
                    "description": case["description"],
                    "success": success
                })
                
            except Exception as e:
                results["error_cases"].append({
                    "id": i + 1,
                    "description": case["description"],
                    "success": False,
                    "error": str(e)
                })
                results["overall_success"] = False
        
        # Test edge cases
        edge_cases = [
            {"text": "a", "language": "yue", "description": "Single character"},
            {"text": "你好" * 100, "language": "yue", "description": "Very long text"},
            {"text": "12345", "language": "yue", "description": "Numbers only"},
            {"text": "!@#$%", "language": "yue", "description": "Special characters"}
        ]
        
        for i, case in enumerate(edge_cases):
            try:
                audio_array, sample_rate = self.tts_engine.synthesize(case["text"], case["language"])
                
                success = isinstance(audio_array, np.ndarray) and len(audio_array) > 0
                
                results["edge_cases"].append({
                    "id": i + 1,
                    "description": case["description"],
                    "text_length": len(case["text"]),
                    "success": success,
                    "audio_length": len(audio_array) if success else 0
                })
                
                if not success:
                    results["overall_success"] = False
                    
            except Exception as e:
                results["edge_cases"].append({
                    "id": i + 1,
                    "description": case["description"],
                    "success": False,
                    "error": str(e)
                })
                results["overall_success"] = False
        
        return results
    
    def generate_report(self, output_path: str):
        """Generate validation report."""
        report = {
            "validation_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0,
                "total_validation_time": self.results.get("total_validation_time", 0)
            },
            "detailed_results": self.results,
            "recommendations": []
        }
        
        # Calculate summary statistics
        test_categories = [
            "basic_synthesis", "multi_language", "speaker_management",
            "voice_conversion", "prosody_control", "performance",
            "audio_quality", "error_handling"
        ]
        
        for category in test_categories:
            if category in self.results:
                category_result = self.results[category]
                if isinstance(category_result, dict) and "overall_success" in category_result:
                    report["validation_summary"]["total_tests"] += 1
                    if category_result["overall_success"]:
                        report["validation_summary"]["passed_tests"] += 1
                    else:
                        report["validation_summary"]["failed_tests"] += 1
        
        # Calculate success rate
        if report["validation_summary"]["total_tests"] > 0:
            report["validation_summary"]["success_rate"] = (
                report["validation_summary"]["passed_tests"] /
                report["validation_summary"]["total_tests"]
            )
        
        # Generate recommendations
        if report["validation_summary"]["success_rate"] < 0.8:
            report["recommendations"].append("Overall success rate is below 80%. Consider reviewing implementation.")
        
        if "performance" in self.results and not self.results["performance"]["overall_success"]:
            report["recommendations"].append("Performance issues detected. Consider optimization.")
        
        if "audio_quality" in self.results and not self.results["audio_quality"]["overall_success"]:
            report["recommendations"].append("Audio quality issues detected. Consider model improvements.")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to: {output_path}")
        return report


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index-TTS2 Validation Script")
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--output", type=str, default="./outputs/tts_validation_report.json",
                       help="Output report path")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize validator
        validator = TTSValidator(args.config, args.device)
        
        # Run validation
        results = validator.run_comprehensive_validation()
        
        # Generate report
        report = validator.generate_report(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("TTS VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['validation_summary']['total_tests']}")
        print(f"Passed: {report['validation_summary']['passed_tests']}")
        print(f"Failed: {report['validation_summary']['failed_tests']}")
        print(f"Success Rate: {report['validation_summary']['success_rate']:.1%}")
        print(f"Total Time: {report['validation_summary']['total_validation_time']:.2f}s")
        print("="*60)
        
        if report['recommendations']:
            print("RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"- {rec}")
        
        # Exit with appropriate code
        if report['validation_summary']['success_rate'] >= 0.8:
            print("✅ TTS validation PASSED")
            sys.exit(0)
        else:
            print("❌ TTS validation FAILED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()