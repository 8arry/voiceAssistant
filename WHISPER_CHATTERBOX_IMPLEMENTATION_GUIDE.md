# 🎤 Whisper + Chatterbox 语音功能替换指南

本指南将帮助您使用开源的 **Whisper (ASR)** 和 **Chatterbox (TTS)** 替换现有的 Google Gemini ADK 语音功能。

## 📋 替换方案概览

### 🔄 **当前架构 vs 新架构**

| 组件 | 当前 (Gemini ADK) | 新方案 (Whisper + Chatterbox) |
|------|------------------|-------------------------------|
| **ASR** | Google ADK 内置 | OpenAI Whisper |
| **TTS** | Google ADK 内置 | Chatterbox (+ 备选方案) |
| **对话** | Google Gemini | 保持 Google Gemini |
| **会话管理** | Redis + ADK | 保持 Redis |
| **通信** | SSE | 保持 SSE |

### 🎯 **优势**
- ✅ **开源**: 完全控制语音处理流程
- ✅ **离线能力**: Whisper 和 Chatterbox 可以离线运行
- ✅ **成本控制**: 减少对 Google 服务的依赖
- ✅ **自定义**: 可以微调模型和参数
- ✅ **隐私**: 语音数据可以在本地处理

## 🚀 实施步骤

### 第一步：安装依赖

```bash
cd voiceAssistant/backend

# 安装新的依赖包
pip install -r requirements_whisper.txt

# 安装 ffmpeg (用于音频处理)
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# 下载 ffmpeg 并添加到 PATH
```

### 第二步：配置环境变量

在您的 `.env` 文件中添加以下配置：

```bash
# 现有的 Gemini 配置保持不变
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

# 新增 Whisper 配置
WHISPER_MODEL_SIZE=base  # 选项: tiny, base, small, medium, large
USE_CUDA=true  # 如果有 GPU 支持

# TTS 配置 (可选)
TTS_ENGINE=chatterbox  # 选项: chatterbox, pyttsx3, gtts
TTS_VOICE=default
TTS_SPEED=150
```

### 第三步：更新后端代码

我已经为您创建了以下新文件：

1. **`app/services/whisper_chatterbox_service.py`** - 核心语音服务
2. **`app/api/voice_whisper.py`** - 新的 API 端点
3. **`requirements_whisper.txt`** - 新的依赖包

### 第四步：更新前端代码

我已经创建了新的前端组件：

1. **`frontend/src/components/WhisperVoiceAssistantOverlay.tsx`** - 新的语音界面

现在您需要更新现有的组件来使用新的端点：

```typescript
// 在 ChatInput.tsx 中添加选择
import WhisperVoiceAssistantOverlay from './WhisperVoiceAssistantOverlay';

// 添加切换按钮
const [useWhisper, setUseWhisper] = useState(false);

// 条件渲染
{useWhisper ? (
  <WhisperVoiceAssistantOverlay
    isOpen={voiceOpen}
    onClose={() => setVoiceOpen(false)}
    isDarkMode={isDarkMode}
  />
) : (
  <VoiceAssistantOverlay
    isOpen={voiceOpen}
    onClose={() => setVoiceOpen(false)}
    isDarkMode={isDarkMode}
  />
)}
```

### 第五步：测试部署

```bash
# 启动后端
cd voiceAssistant/backend
python main.py

# 测试健康检查
curl http://localhost:8000/api/voice-whisper/health

# 期望响应
{
  "status": "healthy",
  "service": "whisper_chatterbox",
  "components": {
    "whisper_asr": true,
    "tts_engine": true,
    "tts_type": "chatterbox",
    "redis": true,
    "gemini": true
  }
}
```

## 🔧 配置选项

### Whisper 模型大小

| 模型 | 参数 | 英语 WER | 多语言 WER | 相对速度 | 内存需求 |
|------|------|----------|------------|----------|----------|
| tiny | 39M | 5.7% | 8.1% | ~32x | ~1GB |
| base | 74M | 4.8% | 7.5% | ~16x | ~1GB |
| small | 244M | 3.6% | 6.2% | ~6x | ~2GB |
| medium | 769M | 2.7% | 5.2% | ~2x | ~5GB |
| large | 1550M | 2.5% | 4.9% | ~1x | ~10GB |

**推荐配置**：
- **开发/测试**: `base` (快速，足够准确)
- **生产环境**: `small` 或 `medium` (平衡性能和准确性)
- **高精度需求**: `large` (需要更多资源)

### TTS 引擎选择

#### 1. Chatterbox (首选)
```python
# 如果 Chatterbox 可用
pip install chatterbox-tts
```

#### 2. pyttsx3 (备选方案 1)
```python
# 本地 TTS，无需网络
pip install pyttsx3
```

#### 3. gTTS (备选方案 2)
```python
# Google TTS，需要网络
pip install gTTS
```

## 🎯 API 端点对比

### 新的 Whisper + Chatterbox 端点

```bash
# SSE 连接
GET /api/voice-whisper/events/{user_id}?is_audio=true

# 发送消息
POST /api/voice-whisper/send/{session_id}
{
  "mime_type": "audio/pcm",
  "data": "base64_audio_data"
}

# 健康检查
GET /api/voice-whisper/health

# 模型信息
GET /api/voice-whisper/models
```

### 原有的 Gemini ADK 端点 (保持不变)

```bash
# SSE 连接
GET /api/voice-redis/events/{user_id}?is_audio=true

# 发送消息
POST /api/voice-redis/send/{session_id}
```

## 🔀 逐步迁移策略

### 阶段 1：并行部署 (推荐)
- 保持现有的 Gemini ADK 功能
- 添加新的 Whisper + Chatterbox 功能
- 用户可以选择使用哪种语音服务
- 逐步收集用户反馈

### 阶段 2：A/B 测试
```typescript
// 在前端添加实验性功能切换
const useExperimentalVoice = user?.experimentalFeatures?.whisperVoice || false;

{useExperimentalVoice ? (
  <WhisperVoiceAssistantOverlay ... />
) : (
  <VoiceAssistantOverlay ... />
)}
```

### 阶段 3：完全切换
- 收集足够的测试数据后
- 将 Whisper + Chatterbox 设为默认
- 保留 Gemini ADK 作为备选方案

## 📊 性能优化

### GPU 加速 (推荐)

```python
# 检查 CUDA 可用性
import torch
print("CUDA available:", torch.cuda.is_available())

# 配置 Whisper 使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)
```

### 音频处理优化

```python
# 批处理音频数据
AUDIO_BATCH_SIZE = 1024 * 4  # 4KB chunks
TRANSCRIPTION_THRESHOLD = 1.0  # 1 second of audio

# 缓存常用响应
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_tts_generation(text: str) -> str:
    return generate_speech(text)
```

## 🐛 故障排除

### 常见问题

#### 1. Whisper 模型下载失败
```bash
# 手动下载模型
python -c "import whisper; whisper.load_model('base')"
```

#### 2. 音频格式不兼容
```python
# 检查音频格式
import librosa
audio, sr = librosa.load(audio_file, sr=16000)
print(f"Sample rate: {sr}, Length: {len(audio)}")
```

#### 3. CUDA 内存不足
```bash
# 使用较小的模型
WHISPER_MODEL_SIZE=tiny

# 或者使用 CPU
USE_CUDA=false
```

#### 4. TTS 输出无声音
```python
# 检查音频设备
import pygame
pygame.mixer.init()
pygame.mixer.get_init()
```

### 日志监控

```python
# 添加详细日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("whisper_chatterbox")
```

## 📈 监控指标

### 关键性能指标 (KPIs)

1. **转录准确率**: 与用户反馈对比
2. **响应延迟**: 从音频到文本的时间
3. **TTS 质量**: 语音自然度评分
4. **系统资源**: CPU/GPU/内存使用率
5. **错误率**: 失败的语音请求比例

### 监控实现

```python
import time
from datetime import datetime

class VoiceMetrics:
    def __init__(self):
        self.transcription_times = []
        self.tts_times = []
        self.error_count = 0
        
    def log_transcription_time(self, duration: float):
        self.transcription_times.append({
            'duration': duration,
            'timestamp': datetime.now()
        })
    
    def get_average_response_time(self) -> float:
        if not self.transcription_times:
            return 0.0
        return sum(t['duration'] for t in self.transcription_times) / len(self.transcription_times)
```

## 🔒 安全考虑

### 数据隐私
- 音频数据在本地处理 (Whisper)
- 不向第三方发送语音数据
- 会话数据加密存储

### 访问控制
- 保持现有的 Supabase 认证
- 会话隔离
- API 速率限制

## 🚀 高级功能

### 多语言支持
```python
# Whisper 自动语言检测
result = whisper_model.transcribe(
    audio_data,
    language=None,  # 自动检测
    task="transcribe"
)
detected_language = result["language"]
```

### 实时流式转录
```python
# 实现流式音频处理
async def stream_transcription(audio_stream):
    buffer = []
    async for audio_chunk in audio_stream:
        buffer.append(audio_chunk)
        if len(buffer) >= CHUNK_SIZE:
            text = await transcribe_chunk(buffer)
            yield text
            buffer = []
```

### 个性化 TTS
```python
# 用户偏好设置
class TTSSettings:
    def __init__(self, user_id: str):
        self.voice_speed = get_user_preference(user_id, 'voice_speed', 150)
        self.voice_type = get_user_preference(user_id, 'voice_type', 'default')
        self.language = get_user_preference(user_id, 'language', 'en')
```

## 📞 支持和反馈

如果您在实施过程中遇到问题：

1. **检查日志**: 查看详细的错误信息
2. **验证依赖**: 确保所有包正确安装
3. **测试组件**: 分别测试 Whisper 和 TTS 功能
4. **性能调优**: 根据您的硬件配置调整参数

---

## 🎉 总结

通过这个实施方案，您将能够：

- ✅ 使用开源的 Whisper 进行语音识别
- ✅ 使用 Chatterbox 进行语音合成
- ✅ 保持现有的对话逻辑 (Gemini)
- ✅ 维持会话管理和认证系统
- ✅ 实现平滑的迁移过程

这种方法让您既能享受开源解决方案的优势，又能保持系统的稳定性和用户体验。

**下一步**: 开始安装依赖并测试新的语音功能！ 