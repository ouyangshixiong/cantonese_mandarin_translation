#!/usr/bin/env python3
"""
Real TTS长句测试脚本 - 使用实际的XTTS-v2引擎
主要用于测试真实的翻译和TTS核心功能，生成真实音频
"""
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torchaudio

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入真实的TTS模块
try:
    from src.tts.xtts_v2_engine import XTTSV2Engine
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"导入TTS模块失败: {e}")
    print("请确保已安装所有依赖: pip install -r requirements-cpu.txt 或 requirements-gpu.txt")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealLongSentenceTTSValidator:
    """真实长句TTS测试验证器 - 使用XTTS-v2引擎"""
    
    def __init__(self, output_dir: str = "./outputs/tts_long_sentences_real", 
                 config_path: Optional[str] = None):
        """初始化真实TTS测试器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / "audio_files"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.tts_engine = None
        
        # 初始化TTS引擎
        self._initialize_tts_engine(config_path)
        
        logger.info("真实TTS长句测试器初始化完成")
    
    def _initialize_tts_engine(self, config_path: Optional[str] = None):
        """初始化TTS引擎"""
        try:
            logger.info("正在初始化XTTS-v2引擎...")
            
            # 创建配置（使用默认值或提供的配置文件）
            if config_path and Path(config_path).exists():
                config = OmegaConf.load(config_path)
                logger.info(f"加载配置文件: {config_path}")
            else:
                # 使用默认配置
                config = self._create_default_config()
                logger.info("使用默认TTS配置")
            
            # 初始化TTS引擎
            self.tts_engine = XTTSV2Engine(config, device="auto")
            
            logger.info(f"TTS引擎初始化完成 - 设备: {self.tts_engine.device}")
            logger.info(f"支持的语言: {self.tts_engine.get_supported_languages()}")
            logger.info(f"支持的情感: {self.tts_engine.get_supported_emotions()}")
            
        except Exception as e:
            logger.error(f"TTS引擎初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化TTS引擎: {str(e)}")
    
    def _create_default_config(self) -> DictConfig:
        """创建默认TTS配置"""
        config_dict = {
            'tts': {
                'sample_rate': 22050,
                'hop_length': 256,
                'n_mels': 80,
                'win_length': 1024
            }
        }
        return OmegaConf.create(config_dict)
    
    def get_test_cases(self) -> List[Dict]:
        """获取长句测试用例"""
        test_cases = [
            {
                "id": 1,
                "cantonese": "我今日朝早去咗茶樓飲茶，見到好多老朋友，大家傾咗好多往事，講起以前一齊做嘢嘅日子，真係好懷念嗰段時間。",
                "description": "日常生活描述 - 80字符",
                "expected_length": 80
            },
            {
                "id": 2,
                "cantonese": "老闆今日開會嘅時候講咗好多關於公司未來發展嘅大計，話要擴展業務去海外市場，仲話要請多啲人嚟幫手做嘢。",
                "description": "工作场景 - 90字符",
                "expected_length": 90
            },
            {
                "id": 3,
                "cantonese": "我覺得人生中最重要嘅唔係賺到幾多錢，而係搵到一個真正懂你嘅人，可以同你一齊分享生活中嘅喜怒哀樂，呢個先係最寶貴嘅嘢。",
                "description": "情感表达 - 95字符",
                "expected_length": 95
            },
            {
                "id": 4,
                "cantonese": "粵語文化真係好博大精深，有好多古老嘅諺語同歇後語，每一句都包含住先輩們嘅智慧同人生哲理，值得我哋好好學習同傳承。",
                "description": "文化描述 - 100字符",
                "expected_length": 100
            },
            {
                "id": 5,
                "cantonese": "阿媽今日煮咗我最鍾意食嘅紅燒肉，仲整咗個老火湯，話咁樣對身體好，叫我多喝啲，真係感受到屋企人嘅關懷同溫暖，好幸福。",
                "description": "家庭场景 - 105字符",
                "expected_length": 105
            }
        ]
        
        return test_cases
    
    def translate_cantonese_to_mandarin(self, cantonese_text: str) -> str:
        """简单的粤语到普通话翻译"""
        try:
            # 这里可以集成真实的翻译API
            # 目前使用简单的字符替换作为示例
            mandarin_text = cantonese_text
            
            # 基本字符映射
            translations = {
                "咗": "了",
                "嘅": "的", 
                "嘢": "东西",
                "冇": "没有",
                "唔": "不",
                "啲": "一些",
                "咁": "这样",
                "嚟": "来",
                "哋": "们",
                "咁樣": "这样",
                "鍾意": "喜欢",
                "朝早": "早上"
            }
            
            for cantonese, mandarin in translations.items():
                mandarin_text = mandarin_text.replace(cantonese, mandarin)
            
            return mandarin_text
            
        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            return cantonese_text  # 失败时返回原文
    
    def real_tts_synthesis(self, text: str, language: str, speaker_id: str, emotion: str) -> Dict:
        """使用真实TTS引擎进行语音合成"""
        try:
            logger.info(f"开始真实TTS合成 - 语言: {language}, 说话人: {speaker_id}, 情感: {emotion}")
            
            synthesis_start = time.time()
            
            # 使用真实的TTS引擎进行合成
            audio_array, sample_rate = self.tts_engine.synthesize(
                text=text,
                language=language,
                speaker_id=speaker_id,
                emotion=emotion,
                speed=1.0,
                pitch_shift=0.0
            )
            
            synthesis_time = time.time() - synthesis_start
            
            # 计算音频时长
            audio_duration = len(audio_array) / sample_rate
            
            # 使用简单音频质量评估
            quality_score = self._assess_audio_quality(audio_array, sample_rate)
            
            logger.info(f"TTS合成完成 - 时长: {audio_duration:.2f}s, 质量分数: {quality_score:.2f}")
            
            return {
                "success": True,
                "audio_array": audio_array,
                "sample_rate": sample_rate,
                "synthesis_time": synthesis_time,
                "audio_duration": audio_duration,
                "quality_metrics": {
                    "overall_score": quality_score,
                    "quality_grade": self._get_quality_grade(quality_score)
                }
            }
            
        except Exception as e:
            logger.error(f"真实TTS合成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _assess_audio_quality(self, audio_array: np.ndarray, sample_rate: int) -> float:
        """简单音频质量评估"""
        try:
            # 基本质量检查
            max_amplitude = np.max(np.abs(audio_array))
            rms = np.sqrt(np.mean(audio_array ** 2))
            
            # 计算信噪比（简化版）
            signal_power = np.mean(audio_array ** 2)
            noise_power = np.mean((audio_array - np.mean(audio_array)) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 50
            
            # 计算动态范围
            dynamic_range = 20 * np.log10(max_amplitude / (rms + 1e-10))
            
            # 计算削波比例
            clipping_ratio = np.sum(np.abs(audio_array) > 0.95) / len(audio_array)
            
            # 综合质量分数
            quality_score = 0.7  # 基础分数
            
            # 基于指标调整分数
            if snr > 20:
                quality_score += 0.1
            if dynamic_range > 40:
                quality_score += 0.1
            if clipping_ratio < 0.01:
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.7  # 默认质量分数
    
    def _get_quality_grade(self, score: float) -> str:
        """获取质量等级"""
        if score >= 0.9:
            return "A+ (优秀)"
        elif score >= 0.8:
            return "A (良好)"
        elif score >= 0.7:
            return "B (中等)"
        elif score >= 0.6:
            return "C (及格)"
        else:
            return "F (不及格)"
    
    def save_audio_file(self, audio_array: np.ndarray, sample_rate: int, 
                       filename: str) -> Optional[Path]:
        """保存音频文件"""
        try:
            audio_path = self.audio_dir / filename
            
            # 使用torchaudio保存
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
            torchaudio.save(audio_path, audio_tensor, sample_rate)
            
            logger.info(f"音频文件已保存: {audio_path}")
            return audio_path
                
        except Exception as e:
            logger.error(f"保存音频文件时出错: {str(e)}")
            return None
    
    def test_single_sentence(self, test_case: Dict, speaker_id: str, emotion: str) -> Dict:
        """测试单个长句"""
        logger.info(f"\n🧪 测试 {test_case['id']}: {test_case['description']}")
        logger.info(f"文本长度: {len(test_case['cantonese'])} 字符")
        logger.info(f"说话人: {speaker_id}")
        logger.info(f"情感: {emotion}")
        
        start_time = time.time()
        
        try:
            # 步骤1: 文本翻译
            logger.info("📄 开始文本翻译...")
            translation_start = time.time()
            
            mandarin_text = self.translate_cantonese_to_mandarin(test_case['cantonese'])
            
            translation_time = time.time() - translation_start
            logger.info(f"✅ 翻译完成: {translation_time:.3f}s")
            logger.info(f"翻译结果: {mandarin_text[:50]}...")
            
            # 步骤2: 真实TTS合成
            logger.info("🎙️ 开始真实语音合成...")
            synthesis_result = self.real_tts_synthesis(
                text=test_case['cantonese'],
                language="yue",  # 粤语
                speaker_id=speaker_id,
                emotion=emotion
            )
            
            if synthesis_result["success"]:
                logger.info(f"✅ 合成完成: {synthesis_result['synthesis_time']:.3f}s")
                logger.info(f"音频长度: {synthesis_result['audio_duration']:.2f}s")
                logger.info(f"音频质量: {synthesis_result['quality_metrics']['overall_score']:.2f}")
                logger.info(f"质量等级: {synthesis_result['quality_metrics']['quality_grade']}")
                
                # 保存音频文件
                audio_filename = f"test_{test_case['id']}_{speaker_id}_{emotion}.wav"
                audio_path = self.save_audio_file(
                    synthesis_result["audio_array"],
                    synthesis_result["sample_rate"],
                    audio_filename
                )
                
                if audio_path:
                    logger.info(f"🎵 音频文件已保存: {audio_path}")
                
            else:
                logger.error(f"❌ 合成失败: {synthesis_result['error']}")
                return {
                    "test_id": test_case['id'],
                    "cantonese_text": test_case['cantonese'],
                    "mandarin_text": mandarin_text,
                    "translation_time": translation_time,
                    "success": False,
                    "error": synthesis_result['error']
                }
            
            total_time = time.time() - start_time
            
            logger.info(f"🎉 测试完成 - 总用时: {total_time:.3f}s")
            logger.info(f"翻译速度: {translation_time/len(test_case['cantonese']):.3f}s/字符")
            logger.info(f"合成速度: {synthesis_result['synthesis_time']/len(test_case['cantonese']):.3f}s/字符")
            
            return {
                "test_id": test_case['id'],
                "cantonese_text": test_case['cantonese'],
                "mandarin_text": mandarin_text,
                "translation_time": translation_time,
                "synthesis_time": synthesis_result['synthesis_time'],
                "total_time": total_time,
                "char_count": len(test_case['cantonese']),
                "audio_duration": synthesis_result['audio_duration'],
                "sample_rate": synthesis_result['sample_rate'],
                "speaker_id": speaker_id,
                "emotion": emotion,
                "audio_file": str(audio_path) if audio_path else None,
                "success": True,
                "quality_metrics": synthesis_result['quality_metrics']
            }
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {str(e)}")
            total_time = time.time() - start_time
            
            return {
                "test_id": test_case['id'],
                "cantonese_text": test_case['cantonese'],
                "mandarin_text": "",
                "translation_time": 0,
                "synthesis_time": 0,
                "total_time": total_time,
                "char_count": len(test_case['cantonese']),
                "success": False,
                "error": str(e)
            }
    
    def run_real_tests(self) -> Dict:
        """运行真实TTS测试"""
        logger.info("🚀 开始真实TTS长句测试...")
        
        test_cases = self.get_test_cases()
        
        # 获取可用的说话人（使用真实TTS引擎的说话人）
        try:
            yue_speakers = self.tts_engine.get_available_speakers("yue")
            cmn_speakers = self.tts_engine.get_available_speakers("cmn")
            speakers = yue_speakers[:2] if yue_speakers else ["yue_female_001", "yue_male_001"]
        except:
            speakers = ["yue_female_001", "yue_male_001"]
        
        emotions = ["neutral", "happy", "sad"]
        
        logger.info(f"测试用例数量: {len(test_cases)}")
        logger.info(f"说话人数量: {len(speakers)}")
        logger.info(f"情感数量: {len(emotions)}")
        logger.info(f"总测试数量: {len(test_cases) * len(speakers) * len(emotions)}")
        
        results = []
        test_count = 0
        success_count = 0
        
        # 运行测试（减少组合以加快测试速度）
        for test_case in test_cases:
            for speaker in speakers[:2]:  # 只测试前2个说话人
                for emotion in emotions[:2]:  # 只测试前2种情感
                    test_count += 1
                    
                    logger.info(f"\n📊 测试进度: {test_count}/{len(test_cases) * 2 * 2}")
                    
                    # 执行测试
                    result = self.test_single_sentence(test_case, speaker, emotion)
                    results.append(result)
                    
                    if result["success"]:
                        success_count += 1
        
        # 生成报告
        report = self.generate_real_report(results, success_count, test_count)
        
        return report
    
    def generate_real_report(self, results: List[Dict], success_count: int, total_count: int) -> Dict:
        """生成真实测试报告"""
        logger.info("📊 生成真实TTS测试报告...")
        
        successful_results = [r for r in results if r["success"]]
        
        # 性能统计
        translation_times = [r["translation_time"] for r in successful_results]
        synthesis_times = [r["synthesis_time"] for r in successful_results]
        total_times = [r["total_time"] for r in successful_results]
        char_counts = [r["char_count"] for r in successful_results]
        audio_durations = [r["audio_duration"] for r in successful_results]
        
        # 质量统计
        quality_scores = []
        snr_values = []
        for result in successful_results:
            if "quality_metrics" in result:
                quality_scores.append(result["quality_metrics"]["overall_score"])
                snr_values.append(result["quality_metrics"]["snr_db"])
        
        report = {
            "test_summary": {
                "total_tests": total_count,
                "successful_tests": success_count,
                "failed_tests": total_count - success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "真实TTS长句测试",
                "tts_engine": "XTTS-v2"
            },
            "performance_metrics": {
                "avg_translation_time": np.mean(translation_times) if translation_times else 0,
                "avg_synthesis_time": np.mean(synthesis_times) if synthesis_times else 0,
                "avg_total_time": np.mean(total_times) if total_times else 0,
                "avg_char_speed_translation": np.mean([t/c for t, c in zip(translation_times, char_counts)]) if translation_times else 0,
                "avg_char_speed_synthesis": np.mean([t/c for t, c in zip(synthesis_times, char_counts)]) if synthesis_times else 0,
                "max_translation_time": max(translation_times) if translation_times else 0,
                "max_synthesis_time": max(synthesis_times) if synthesis_times else 0,
                "min_translation_time": min(translation_times) if translation_times else 0,
                "min_synthesis_time": min(synthesis_times) if synthesis_times else 0,
                "avg_audio_duration": np.mean(audio_durations) if audio_durations else 0,
                "total_audio_duration": sum(audio_durations) if audio_durations else 0
            },
            "quality_metrics": {
                "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
                "min_quality_score": min(quality_scores) if quality_scores else 0,
                "max_quality_score": max(quality_scores) if quality_scores else 0,
                "quality_distribution": self._get_real_quality_distribution(quality_scores),
                "audio_files_saved": len([r for r in successful_results if r.get("audio_file")])
            },
            "system_info": {
                "device": str(self.tts_engine.device) if self.tts_engine else "unknown",
                "memory_usage": self.tts_engine.get_memory_usage() if self.tts_engine else {},
                "available_speakers": {
                    "yue": self.tts_engine.get_available_speakers("yue") if self.tts_engine else [],
                    "cmn": self.tts_engine.get_available_speakers("cmn") if self.tts_engine else []
                }
            },
            "detailed_results": results
        }
        
        # 保存报告
        report_path = self.output_dir / "tts_long_sentences_real_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"真实TTS测试报告已保存: {report_path}")
        
        return report
    
    def _get_real_quality_distribution(self, quality_scores: List[float]) -> Dict:
        """获取真实质量分布"""
        if not quality_scores:
            return {}
        
        ranges = {
            "优秀 (0.8-1.0)": 0,
            "良好 (0.6-0.8)": 0,
            "一般 (0.4-0.6)": 0,
            "较差 (0.0-0.4)": 0
        }
        
        for score in quality_scores:
            if score >= 0.8:
                ranges["优秀 (0.8-1.0)"] += 1
            elif score >= 0.6:
                ranges["良好 (0.6-0.8)"] += 1
            elif score >= 0.4:
                ranges["一般 (0.4-0.6)"] += 1
            else:
                ranges["较差 (0.0-0.4)"] += 1
        
        return ranges


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="真实TTS长句测试脚本 - 使用XTTS-v2引擎")
    parser.add_argument("--output", type=str, default="./outputs/tts_long_sentences_real",
                       help="输出目录")
    parser.add_argument("--config", type=str, default=None,
                       help="TTS配置文件路径")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化测试器
        logger.info("🚀 开始真实TTS长句测试...")
        validator = RealLongSentenceTTSValidator(
            output_dir=args.output,
            config_path=args.config
        )
        
        # 运行测试
        report = validator.run_real_tests()
        
        # 打印总结
        print("\n" + "="*80)
        print("📊 真实TTS长句测试完成报告")
        print("="*80)
        print(f"总测试数量: {report['test_summary']['total_tests']}")
        print(f"成功测试: {report['test_summary']['successful_tests']}")
        print(f"失败测试: {report['test_summary']['failed_tests']}")
        print(f"成功率: {report['test_summary']['success_rate']:.1%}")
        print(f"平均翻译时间: {report['performance_metrics']['avg_translation_time']:.3f}秒")
        print(f"平均合成时间: {report['performance_metrics']['avg_synthesis_time']:.3f}秒")
        print(f"平均音频质量: {report['quality_metrics']['avg_quality_score']:.2f}")
        print(f"总音频时长: {report['performance_metrics']['total_audio_duration']:.1f}秒")
        print(f"音频文件保存: {report['quality_metrics']['audio_files_saved']}个")
        print(f"TTS引擎: {report['system_info']['device']}")
        print("="*80)
        
        # 根据成功率判断测试是否通过
        if report['test_summary']['success_rate'] >= 0.8:
            print("✅ 真实TTS长句测试 PASSED")
            print(f"🎵 音频文件已保存至: {args.output}/audio_files/")
            return 0
        else:
            print("❌ 真实TTS长句测试 FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)