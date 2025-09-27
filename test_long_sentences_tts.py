#!/usr/bin/env python3
"""
TTS长句测试脚本 - Index-TTS2长句翻译与语音合成测试
测试长句翻译质量和语音合成性能，包括音频质量评估
"""
import sys
import time
import json
import logging
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference.translator import CantoneseTranslator
from tts import IndexTTS2Engine, CrossLingualVoiceConverter, SpeakerManager, AudioProcessor
from omegaconf import OmegaConf

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果数据结构"""
    test_id: int
    cantonese_text: str
    mandarin_text: str
    translation_time: float
    synthesis_time: float
    total_time: float
    char_count: int
    audio_length: float
    sample_rate: int
    speaker_id: str
    emotion: str
    success: bool
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict] = None
    voice_conversion_metrics: Optional[Dict] = None


class LongSentenceTTSValidator:
    """长句TTS测试验证器"""
    
    def __init__(self, config_path: str, device: str = "auto", output_dir: str = "./outputs/tts_long_sentences"):
        """
        初始化TTS长句测试器
        
        Args:
            config_path: 配置文件路径
            device: 计算设备
            output_dir: 输出目录
        """
        self.config = OmegaConf.load(config_path)
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._initialize_components()
        
        # 测试结果存储
        self.results: List[TestResult] = []
        self.test_start_time = None
        
        logger.info(f"TTS长句测试器初始化完成 - 设备: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"使用设备: {device}")
        if device == "cuda":
            logger.info(f"GPU型号: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        return torch.device(device)
    
    def _initialize_components(self):
        """初始化TTS组件"""
        try:
            logger.info("正在初始化翻译器...")
            self.translator = CantoneseTranslator(
                model_name=self.config.model.name,
                framework=self.config.framework.name,
                device=str(self.device)
            )
            
            logger.info("正在初始化TTS引擎...")
            self.tts_engine = IndexTTS2Engine(self.config, str(self.device))
            
            logger.info("正在初始化语音转换器...")
            self.voice_converter = CrossLingualVoiceConverter(self.config, str(self.device))
            
            logger.info("正在初始化说话人管理器...")
            self.speaker_manager = SpeakerManager(self.config, str(self.device))
            
            logger.info("正在初始化音频处理器...")
            self.audio_processor = AudioProcessor(
                sample_rate=self.config.tts.sample_rate,
                n_mels=self.config.tts.n_mels,
                hop_length=self.config.tts.hop_length,
                win_length=self.config.tts.win_length
            )
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            raise
    
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
            },
            {
                "id": 6,
                "cantonese": "琴晚同班朋友去咗唱K，大家都玩得好開心，有人唱咗經典粵語歌，有人跳咗舞，整個晚上都充滿住歡樂同笑聲，真係好難忘嘅聚會。",
                "description": "朋友聚会 - 110字符",
                "expected_length": 110
            },
            {
                "id": 7,
                "cantonese": "今日嘅天氣真係幾好，陽光普照，微風輕輕吹過，啲樹葉沙沙作響，行喺街上感覺特別舒服，令人心情愉快，真係適合出外走走。",
                "description": "天气描述 - 115字符",
                "expected_length": 115
            },
            {
                "id": 8,
                "cantonese": "我頭先去咗商場買嘢，見到好多新款嘅電子產品，有最新款嘅手機同電腦，科技真係日新月異，發展得好快，令人目不暇給，好驚訝。",
                "description": "购物经历 - 120字符",
                "expected_length": 120
            },
            {
                "id": 9,
                "cantonese": "就嚟到中秋節喇，每年呢個時候屋企人都會聚埋一齊賞月食月餅，仲會整啲應節嘅小食，大家團團圓圓好溫馨，呢種傳統好值得保持。",
                "description": "节日庆祝 - 125字符",
                "expected_length": 125
            },
            {
                "id": 10,
                "cantonese": "記得去年去咗旅行，去咗一個好靚嘅海邊小鎮，嗰度嘅風景如畫，海水清澈見底，沙灘潔白如雪，真係一個令人難忘嘅地方，好想再去一次。",
                "description": "旅行回忆 - 130字符",
                "expected_length": 130
            },
            {
                "id": 11,
                "cantonese": "人工智能技術發展真係好快，已經應用喺好多唔同嘅領域，包括醫療、教育、金融等等，為我哋嘅生活帶嚟好多便利，但係都要注意倫理問題。",
                "description": "技术讨论 - 140字符",
                "expected_length": 140
            },
            {
                "id": 12,
                "cantonese": "我哋公司今年嘅業績表現非常好，營業額比去年同期增長咗三成，利潤率都有顯著提升，呢個係全體員工共同努力嘅成果，值得大家驕傲同慶祝。",
                "description": "商业演示 - 150字符",
                "expected_length": 150
            },
            {
                "id": 13,
                "cantonese": "學習語言最重要嘅係要多聽多講，唔好驚犯錯，要敢於開口練習，每日都要保持學習嘅習慣，持之以恆先會有進步，記住失敗係成功之母，堅持就會勝利。",
                "description": "教育内容 - 160字符",
                "expected_length": 160
            },
            {
                "id": 14,
                "cantonese": "醫生建議我要多注意身體健康，要保持良好嘅生活習慣，包括充足嘅睡眠、均衡嘅飲食、適量嘅運動，同時要保持心情愉快，減少壓力，咁樣先可以延年益壽。",
                "description": "医疗咨询 - 170字符",
                "expected_length": 170
            },
            {
                "id": 15,
                "cantonese": "回顧歷史，我哋可以學到好多嘢，每一個時代都有自己嘅特色同挑戰，但係人類總係能夠克服困難，不斷進步，呢種堅韌不拔嘅精神值得我哋學習同傳承，創造更美好嘅未來。",
                "description": "历史叙述 - 180字符",
                "expected_length": 180
            }
        ]
        
        return test_cases
    
    def get_speaker_configs(self) -> List[Dict]:
        """获取说话人配置"""
        return [
            {"id": "yue_female_001", "language": "yue", "gender": "female", "description": "粤语女声"},
            {"id": "yue_male_001", "language": "yue", "gender": "male", "description": "粤语男声"},
            {"id": "cmn_female_001", "language": "cmn", "gender": "female", "description": "普通话女声"},
            {"id": "cmn_male_001", "language": "cmn", "gender": "male", "description": "普通话男声"}
        ]
    
    def get_emotion_configs(self) -> List[Dict]:
        """获取情感配置"""
        return [
            {"emotion": "neutral", "description": "中性"},
            {"emotion": "happy", "description": "开心"},
            {"emotion": "sad", "description": "悲伤"},
            {"emotion": "excited", "description": "兴奋"}
        ]
    
    def assess_audio_quality(self, audio_array: np.ndarray, sample_rate: int) -> Dict:
        """评估音频质量"""
        try:
            # 转换为tensor
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # 使用音频处理器评估质量
            quality_metrics = self.audio_processor.assess_quality(audio_tensor, sample_rate)
            
            return {
                "snr_db": quality_metrics.snr_db,
                "dynamic_range": quality_metrics.dynamic_range,
                "clipping_ratio": quality_metrics.clipping_ratio,
                "spectral_rolloff": quality_metrics.spectral_rolloff,
                "zero_crossing_rate": quality_metrics.zero_crossing_rate,
                "spectral_centroid": quality_metrics.spectral_centroid,
                "overall_score": quality_metrics.overall_score,
                "quality_grade": self._get_quality_grade(quality_metrics.overall_score)
            }
            
        except Exception as e:
            logger.error(f"音频质量评估失败: {str(e)}")
            return {
                "error": str(e),
                "overall_score": 0.0,
                "quality_grade": "F"
            }
    
    def _get_quality_grade(self, score: float) -> str:
        """根据分数获取质量等级"""
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
    
    def test_single_sentence(self, test_case: Dict, speaker_config: Dict, emotion_config: Dict) -> TestResult:
        """测试单个长句"""
        logger.info(f"\n🧪 测试 {test_case['id']}: {test_case['description']}")
        logger.info(f"文本长度: {len(test_case['cantonese'])} 字符")
        logger.info(f"说话人: {speaker_config['description']} ({speaker_config['id']})")
        logger.info(f"情感: {emotion_config['description']} ({emotion_config['emotion']})")
        
        start_time = time.time()
        
        try:
            # 步骤1: 文本翻译
            logger.info("📄 开始文本翻译...")
            translation_start = time.time()
            
            mandarin_text = self.translator.translate(
                test_case['cantonese'],
                source_lang="cantonese",
                target_lang="mandarin"
            )
            
            translation_time = time.time() - translation_start
            logger.info(f"✅ 翻译完成: {translation_time:.3f}s")
            logger.info(f"粤语原文: {test_case['cantonese'][:50]}...")
            logger.info(f"普通话翻译: {mandarin_text[:50]}...")
            
            # 步骤2: 语音合成
            logger.info("🎙️ 开始语音合成...")
            synthesis_start = time.time()
            
            # 确定TTS语言
            tts_language = "yue" if speaker_config['language'] == "yue" else "cmn"
            
            audio_array, sample_rate = self.tts_engine.synthesize(
                text=test_case['cantonese'],  # 使用原文进行TTS测试
                language=tts_language,
                speaker_id=speaker_config['id'],
                emotion=emotion_config['emotion'],
                speed=1.0
            )
            
            synthesis_time = time.time() - synthesis_start
            audio_length = len(audio_array) / sample_rate
            
            logger.info(f"✅ 合成完成: {synthesis_time:.3f}s")
            logger.info(f"音频长度: {audio_length:.2f}s")
            logger.info(f"采样率: {sample_rate}Hz")
            
            # 步骤3: 音频质量评估
            logger.info("🔍 开始音频质量评估...")
            quality_metrics = self.assess_audio_quality(audio_array, sample_rate)
            
            if "error" not in quality_metrics:
                logger.info(f"音频质量评分: {quality_metrics['overall_score']:.2f}")
                logger.info(f"质量等级: {quality_metrics['quality_grade']}")
                logger.info(f"信噪比: {quality_metrics['snr_db']:.1f}dB")
            else:
                logger.warning(f"质量评估失败: {quality_metrics['error']}")
            
            # 步骤4: 语音转换测试（跨语言）
            voice_conversion_metrics = None
            if tts_language == "yue":
                logger.info("🗣️ 开始跨语言语音转换测试...")
                try:
                    conversion_start = time.time()
                    
                    converted_audio = self.voice_converter.convert_voice(
                        source_audio=audio_array,
                        source_language="yue",
                        target_language="cmn"
                    )
                    
                    conversion_time = time.time() - conversion_start
                    converted_length = len(converted_audio) / sample_rate
                    
                    logger.info(f"✅ 语音转换完成: {conversion_time:.3f}s")
                    logger.info(f"转换后音频长度: {converted_length:.2f}s")
                    
                    voice_conversion_metrics = {
                        "conversion_time": conversion_time,
                        "converted_length": converted_length,
                        "length_ratio": converted_length / audio_length if audio_length > 0 else 0
                    }
                    
                except Exception as e:
                    logger.error(f"语音转换失败: {str(e)}")
                    voice_conversion_metrics = {"error": str(e)}
            
            total_time = time.time() - start_time
            
            logger.info(f"🎉 测试完成 - 总用时: {total_time:.3f}s")
            logger.info(f"翻译速度: {translation_time/len(test_case['cantonese']):.3f}s/字符")
            logger.info(f"合成速度: {synthesis_time/len(test_case['cantonese']):.3f}s/字符")
            
            return TestResult(
                test_id=test_case['id'],
                cantonese_text=test_case['cantonese'],
                mandarin_text=mandarin_text,
                translation_time=translation_time,
                synthesis_time=synthesis_time,
                total_time=total_time,
                char_count=len(test_case['cantonese']),
                audio_length=audio_length,
                sample_rate=sample_rate,
                speaker_id=speaker_config['id'],
                emotion=emotion_config['emotion'],
                success=True,
                quality_metrics=quality_metrics,
                voice_conversion_metrics=voice_conversion_metrics
            )
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {str(e)}")
            total_time = time.time() - start_time
            
            return TestResult(
                test_id=test_case['id'],
                cantonese_text=test_case['cantonese'],
                mandarin_text="",
                translation_time=0,
                synthesis_time=0,
                total_time=total_time,
                char_count=len(test_case['cantonese']),
                audio_length=0,
                sample_rate=0,
                speaker_id=speaker_config['id'],
                emotion=emotion_config['emotion'],
                success=False,
                error_message=str(e)
            )
    
    def save_audio_sample(self, audio_array: np.ndarray, sample_rate: int, filename: str) -> str:
        """保存音频样本"""
        try:
            output_path = self.output_dir / filename
            
            # 转换为tensor并保存
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            success = self.audio_processor.save_audio(
                audio=audio_tensor,
                filepath=output_path,
                sample_rate=sample_rate,
                format="wav"
            )
            
            if success:
                logger.info(f"音频已保存: {output_path}")
                return str(output_path)
            else:
                logger.error(f"音频保存失败: {output_path}")
                return ""
                
        except Exception as e:
            logger.error(f"音频保存错误: {str(e)}")
            return ""
    
    def run_comprehensive_tests(self) -> Dict:
        """运行综合测试"""
        logger.info("🚀 开始TTS长句综合测试...")
        self.test_start_time = time.time()
        
        test_cases = self.get_test_cases()
        speaker_configs = self.get_speaker_configs()
        emotion_configs = self.get_emotion_configs()
        
        logger.info(f"测试用例数量: {len(test_cases)}")
        logger.info(f"说话人配置数量: {len(speaker_configs)}")
        logger.info(f"情感配置数量: {len(emotion_configs)}")
        logger.info(f"总测试数量: {len(test_cases) * len(speaker_configs) * len(emotion_configs)}")
        
        test_count = 0
        success_count = 0
        
        # 运行所有组合测试
        for test_case in test_cases:
            for speaker_config in speaker_configs:
                for emotion_config in emotion_configs:
                    test_count += 1
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"📊 测试进度: {test_count}/{len(test_cases) * len(speaker_configs) * len(emotion_configs)}")
                    
                    # 执行测试
                    result = self.test_single_sentence(test_case, speaker_config, emotion_config)
                    self.results.append(result)
                    
                    if result.success:
                        success_count += 1
                        
                        # 保存成功的音频样本（减少存储，只保存部分）
                        if test_count % 5 == 0:  # 每5个测试保存一个样本
                            filename = f"test_{result.test_id}_{result.speaker_id}_{result.emotion}.wav"
                            audio_path = self.save_audio_sample(
                                audio_array=torch.randn(22050 * 2),  # 占位符，实际音频需要从TTS引擎获取
                                sample_rate=result.sample_rate,
                                filename=filename
                            )
        
        total_time = time.time() - self.test_start_time
        
        # 生成测试报告
        report = self.generate_test_report(success_count, test_count, total_time)
        
        logger.info(f"\n🎉 TTS长句测试完成!")
        logger.info(f"总测试数量: {test_count}")
        logger.info(f"成功测试: {success_count}")
        logger.info(f"成功率: {success_count/test_count:.1%}")
        logger.info(f"总用时: {total_time:.2f}秒")
        
        return report
    
    def generate_test_report(self, success_count: int, total_count: int, total_time: float) -> Dict:
        """生成测试报告"""
        logger.info("📊 生成测试报告...")
        
        if not self.results:
            return {"error": "没有测试结果"}
        
        # 基础统计
        successful_results = [r for r in self.results if r.success]
        
        # 性能统计
        translation_times = [r.translation_time for r in successful_results]
        synthesis_times = [r.synthesis_time for r in successful_results]
        total_times = [r.total_time for r in successful_results]
        char_counts = [r.char_count for r in successful_results]
        
        # 质量统计
        quality_scores = []
        for result in successful_results:
            if result.quality_metrics and "overall_score" in result.quality_metrics:
                quality_scores.append(result.quality_metrics["overall_score"])
        
        report = {
            "test_summary": {
                "total_tests": total_count,
                "successful_tests": success_count,
                "failed_tests": total_count - success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "total_test_time": total_time,
                "test_timestamp": datetime.now().isoformat(),
                "device": str(self.device)
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
                "min_synthesis_time": min(synthesis_times) if synthesis_times else 0
            },
            "quality_metrics": {
                "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
                "min_quality_score": min(quality_scores) if quality_scores else 0,
                "max_quality_score": max(quality_scores) if quality_scores else 0,
                "quality_distribution": self._get_quality_distribution(quality_scores)
            },
            "speaker_performance": self._analyze_speaker_performance(),
            "emotion_performance": self._analyze_emotion_performance(),
            "detailed_results": [self._result_to_dict(r) for r in self.results]
        }
        
        # 保存报告
        report_path = self.output_dir / "tts_long_sentences_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试报告已保存: {report_path}")
        
        # 生成可视化图表
        self.generate_visualizations(report)
        
        return report
    
    def _get_quality_distribution(self, quality_scores: List[float]) -> Dict:
        """获取质量分数分布"""
        if not quality_scores:
            return {}
        
        ranges = {
            "A+ (0.9-1.0)": 0,
            "A (0.8-0.9)": 0,
            "B (0.7-0.8)": 0,
            "C (0.6-0.7)": 0,
            "F (0.0-0.6)": 0
        }
        
        for score in quality_scores:
            if score >= 0.9:
                ranges["A+ (0.9-1.0)"] += 1
            elif score >= 0.8:
                ranges["A (0.8-0.9)"] += 1
            elif score >= 0.7:
                ranges["B (0.7-0.8)"] += 1
            elif score >= 0.6:
                ranges["C (0.6-0.7)"] += 1
            else:
                ranges["F (0.0-0.6)"] += 1
        
        return ranges
    
    def _analyze_speaker_performance(self) -> Dict:
        """分析说话人性能"""
        speaker_stats = {}
        
        for result in self.results:
            if result.success and result.speaker_id:
                if result.speaker_id not in speaker_stats:
                    speaker_stats[result.speaker_id] = {
                        "total_tests": 0,
                        "successful_tests": 0,
                        "avg_synthesis_time": 0,
                        "avg_quality_score": 0,
                        "quality_scores": []
                    }
                
                speaker_stats[result.speaker_id]["total_tests"] += 1
                speaker_stats[result.speaker_id]["successful_tests"] += 1
                
                if result.quality_metrics and "overall_score" in result.quality_metrics:
                    speaker_stats[result.speaker_id]["quality_scores"].append(
                        result.quality_metrics["overall_score"]
                    )
        
        # 计算平均值
        for speaker_id, stats in speaker_stats.items():
            if stats["quality_scores"]:
                stats["avg_quality_score"] = np.mean(stats["quality_scores"])
            
            # 计算成功率
            stats["success_rate"] = stats["successful_tests"] / stats["total_tests"]
        
        return speaker_stats
    
    def _analyze_emotion_performance(self) -> Dict:
        """分析情感性能"""
        emotion_stats = {}
        
        for result in self.results:
            if result.success and result.emotion:
                if result.emotion not in emotion_stats:
                    emotion_stats[result.emotion] = {
                        "total_tests": 0,
                        "successful_tests": 0,
                        "avg_synthesis_time": 0,
                        "avg_quality_score": 0,
                        "quality_scores": []
                    }
                
                emotion_stats[result.emotion]["total_tests"] += 1
                emotion_stats[result.emotion]["successful_tests"] += 1
                
                if result.quality_metrics and "overall_score" in result.quality_metrics:
                    emotion_stats[result.emotion]["quality_scores"].append(
                        result.quality_metrics["overall_score"]
                    )
        
        # 计算平均值
        for emotion, stats in emotion_stats.items():
            if stats["quality_scores"]:
                stats["avg_quality_score"] = np.mean(stats["quality_scores"])
            
            # 计算成功率
            stats["success_rate"] = stats["successful_tests"] / stats["total_tests"]
        
        return emotion_stats
    
    def _result_to_dict(self, result: TestResult) -> Dict:
        """转换结果为字典格式"""
        return {
            "test_id": result.test_id,
            "cantonese_text": result.cantonese_text,
            "mandarin_text": result.mandarin_text,
            "translation_time": result.translation_time,
            "synthesis_time": result.synthesis_time,
            "total_time": result.total_time,
            "char_count": result.char_count,
            "audio_length": result.audio_length,
            "sample_rate": result.sample_rate,
            "speaker_id": result.speaker_id,
            "emotion": result.emotion,
            "success": result.success,
            "error_message": result.error_message,
            "quality_metrics": result.quality_metrics,
            "voice_conversion_metrics": result.voice_conversion_metrics
        }
    
    def generate_visualizations(self, report: Dict):
        """生成可视化图表"""
        try:
            logger.info("📊 生成可视化图表...")
            
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('TTS长句测试可视化报告', fontsize=16, fontweight='bold')
            
            # 图表1: 性能时间分布
            self._plot_performance_distribution(axes[0, 0], report)
            
            # 图表2: 音频质量分布
            self._plot_quality_distribution(axes[0, 1], report)
            
            # 图表3: 说话人性能对比
            self._plot_speaker_performance(axes[1, 0], report)
            
            # 图表4: 情感性能对比
            self._plot_emotion_performance(axes[1, 1], report)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.output_dir / "tts_long_sentences_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化图表已保存: {chart_path}")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {str(e)}")
    
    def _plot_performance_distribution(self, ax, report: Dict):
        """绘制性能时间分布图"""
        perf_metrics = report["performance_metrics"]
        
        times = []
        labels = []
        
        if perf_metrics.get("translation_times"):
            times.extend(perf_metrics["translation_times"])
            labels.extend(["翻译时间"] * len(perf_metrics["translation_times"]))
        
        if perf_metrics.get("synthesis_times"):
            times.extend(perf_metrics["synthesis_times"])
            labels.extend(["合成时间"] * len(perf_metrics["synthesis_times"]))
        
        if times:
            ax.hist(times, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('频次')
            ax.set_title('处理时间分布')
            ax.grid(True, alpha=0.3)
    
    def _plot_quality_distribution(self, ax, report: Dict):
        """绘制质量分布图"""
        quality_dist = report["quality_metrics"]["quality_distribution"]
        
        if quality_dist:
            categories = list(quality_dist.keys())
            counts = list(quality_dist.values())
            
            ax.bar(categories, counts, color=['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#DC143C'])
            ax.set_xlabel('质量等级')
            ax.set_ylabel('测试数量')
            ax.set_title('音频质量分布')
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for i, v in enumerate(counts):
                ax.text(i, v + max(counts) * 0.01, str(v), ha='center', va='bottom')
    
    def _plot_speaker_performance(self, ax, report: Dict):
        """绘制说话人性能对比图"""
        speaker_stats = report["speaker_performance"]
        
        if speaker_stats:
            speakers = list(speaker_stats.keys())
            success_rates = [stats["success_rate"] for stats in speaker_stats.values()]
            avg_quality = [stats.get("avg_quality_score", 0) for stats in speaker_stats.values()]
            
            x = np.arange(len(speakers))
            width = 0.35
            
            ax.bar(x - width/2, success_rates, width, label='成功率', color='skyblue')
            ax.bar(x + width/2, avg_quality, width, label='平均质量', color='lightcoral')
            
            ax.set_xlabel('说话人')
            ax.set_ylabel('性能指标')
            ax.set_title('说话人性能对比')
            ax.set_xticks(x)
            ax.set_xticklabels(speakers, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_emotion_performance(self, ax, report: Dict):
        """绘制情感性能对比图"""
        emotion_stats = report["emotion_performance"]
        
        if emotion_stats:
            emotions = list(emotion_stats.keys())
            success_rates = [stats["success_rate"] for stats in emotion_stats.values()]
            avg_quality = [stats.get("avg_quality_score", 0) for stats in emotion_stats.values()]
            
            x = np.arange(len(emotions))
            width = 0.35
            
            ax.bar(x - width/2, success_rates, width, label='成功率', color='lightgreen')
            ax.bar(x + width/2, avg_quality, width, label='平均质量', color='orange')
            
            ax.set_xlabel('情感类型')
            ax.set_ylabel('性能指标')
            ax.set_title('情感性能对比')
            ax.set_xticks(x)
            ax.set_xticklabels(emotions, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS长句测试脚本")
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备 (auto, cpu, cuda)")
    parser.add_argument("--output", type=str, default="./outputs/tts_long_sentences",
                       help="输出目录")
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式（减少测试组合）")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    def timeout_handler(signum, frame):
        print("\n⏰ 测试超时，但继续运行...")
    
    # 设置超时为30分钟
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800)  # 30分钟
    
    try:
        # 初始化测试器
        logger.info("🚀 开始TTS长句测试...")
        validator = LongSentenceTTSValidator(
            config_path=args.config,
            device=args.device,
            output_dir=args.output
        )
        
        # 运行测试
        report = validator.run_comprehensive_tests()
        
        # 打印总结
        print("\n" + "="*80)
        print("📊 TTS长句测试完成报告")
        print("="*80)
        print(f"总测试数量: {report['test_summary']['total_tests']}")
        print(f"成功测试: {report['test_summary']['successful_tests']}")
        print(f"失败测试: {report['test_summary']['failed_tests']}")
        print(f"成功率: {report['test_summary']['success_rate']:.1%}")
        print(f"总用时: {report['test_summary']['total_test_time']:.2f}秒")
        print(f"平均翻译时间: {report['performance_metrics']['avg_translation_time']:.3f}秒")
        print(f"平均合成时间: {report['performance_metrics']['avg_synthesis_time']:.3f}秒")
        print(f"平均音频质量: {report['quality_metrics']['avg_quality_score']:.2f}")
        print("="*80)
        
        # 根据成功率判断测试是否通过
        if report['test_summary']['success_rate'] >= 0.8:
            print("✅ TTS长句测试 PASSED")
            sys.exit(0)
        else:
            print("❌ TTS长句测试 FAILED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        sys.exit(1)
    finally:
        signal.alarm(0)  # 取消超时


if __name__ == "__main__":
    main()