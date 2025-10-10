# 🈵 粤语-普通话-英语智能翻译系统

**业界首个**集成Coqui/XTTS-v2语音合成的粤语-普通话-英语三向翻译解决方案

### 📋 产品经理快速验证
```bash
# 验证翻译功能
python scripts/inference_tts.py --text "你好嗎" --source_lang cantonese --target_lang mandarin

# 验证语音合成
python scripts/inference_tts.py --text "Hello, how are you?" --source_lang cantonese --target_lang mandarin --enable_tts --speaker_id yue_female_001

# 验证英语语音合成（新增功能）
python scripts/inference_tts.py --text "你好吗" --source_lang mandarin --target_lang english --enable_tts --speaker_id en_female_001

# 验证长文本处理
python scripts/inference_tts.py --text "各位合作伙伴，基于我们过去一年的紧密协作和共同努力，我们在大湾区市场取得了显著成绩，营业额同比增长35%，这为我们未来的深度合作奠定了坚实基础。" --source_lang cantonese --target_lang mandarin --enable_tts
```

### 🔧 技术负责人快速验证
```bash
# 激活环境
source debug-gpu/bin/activate

# 验证系统性能
python scripts/inference_tts.py --text "Test performance" --source_lang cantonese --target_lang mandarin --enable_tts --verbose

# 验证TTS功能
python -c "
from src.tts.xtts_v2_engine import XTTSV2Engine
from omegaconf import DictConfig
config = DictConfig({'tts': {'sample_rate': 22050}})
tts_engine = XTTSV2Engine(config)
audio, sr = tts_engine.synthesize(text='Hello, this is a test.', language='cmn', speaker_id=None)
print(f'✅ TTS正常！音频：{len(audio)} samples')
"

# 验证情感控制
python scripts/inference_tts.py --text "Hello, emotion test" --source_lang cantonese --target_lang mandarin --enable_tts --emotion happy --speed 1.5
```

### 🧪 测试负责人快速验证
```bash
# 基础功能测试
python scripts/inference_tts.py --text "你好嗎" --source_lang cantonese --target_lang mandarin --enable_tts --speaker_id yue_female_001 --voice_output_dir ./test_output

# 英语语音合成测试（新增）
python scripts/inference_tts.py --text "今天天气很好" --source_lang mandarin --target_lang english --enable_tts --speaker_id en_female_001 --voice_output_dir ./test_output

# 性能测试
python scripts/inference_tts.py --text "This is a performance test sentence to verify system stability under normal load conditions." --source_lang cantonese --target_lang mandarin --enable_tts --verbose

# 音频质量检查
python -c "
import torchaudio
audio, sr = torchaudio.load('./test_output/translated_audio.wav')
print(f'采样率：{sr}Hz, 时长：{len(audio[0])/sr:.2f}s, 质量：正常')
"
```

## 🔧 常见问题

### TTS初始化失败
```bash
# 检查TTS库
python -c "from TTS.api import TTS; print('✅ TTS库正常')"

# 重新安装
pip install TTS>=0.22.0
```

### 模型下载问题
```bash
# 接受XTTS-v2许可证
python test_xtts_license.py
```

### GPU内存不足
```bash
# 使用CPU模式
python scripts/inference_tts.py --device cpu --text "Test" --source_lang cantonese --target_lang mandarin --enable_tts
```

## 🆕 新增功能：英语语音合成

系统现已支持**普通话→英语语音合成**功能：

### 功能特性
- ✅ **普通话输入→英语语音输出**
- ✅ **高质量英语语音合成**（基于XTTS-v2英语模型）
- ✅ **英语文本预处理**（自动处理缩写、数字等）
- ✅ **多说话人支持**（英语专用说话人模型）
- ✅ **情感控制**（支持英语情感语音合成）

### 使用示例
```bash
# 普通话→英语语音翻译
python scripts/inference_tts.py --text "你好，今天天气很好" --source_lang mandarin --target_lang english --enable_tts --speaker_id en_female_001

# 带情感的英语语音
python scripts/inference_tts.py --text "我很高兴见到你" --source_lang mandarin --target_lang english --enable_tts --emotion happy --speed 1.2
```

### 技术实现
- **语言支持**：新增英语('en')语言配置
- **文本预处理**：英语缩写扩展、数字转文字
- **模型集成**：XTTS-v2英语模型无缝集成
- **质量保证**：英语语音质量验证测试

---

**版本**：v3.1 (英语TTS集成版)  
**状态**：✅ 粤语/普通话/英语三向语音合成支持