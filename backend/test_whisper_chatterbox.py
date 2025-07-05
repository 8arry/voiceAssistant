#!/usr/bin/env python3
"""
Whisper + Chatterbox 功能测试脚本

这个脚本用于测试 Whisper ASR 和 Chatterbox TTS 功能是否正常工作。
运行此脚本前，请确保已安装所有必要的依赖。

使用方法:
    python test_whisper_chatterbox.py
"""

import os
import sys
import asyncio
import tempfile
import base64
import json
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有必要的包是否已正确安装"""
    print("🔍 测试依赖包导入...")
    
    try:
        import whisper
        print("✅ Whisper 导入成功")
    except ImportError as e:
        print(f"❌ Whisper 导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch 导入成功 (CUDA 可用: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy 导入成功")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")
        return False
    
    # TTS 包测试
    tts_available = []
    
    try:
        from chatterbox import TTSEngine
        print("✅ Chatterbox TTS 可用")
        tts_available.append("chatterbox")
    except ImportError:
        print("⚠️ Chatterbox TTS 不可用")
    
    try:
        import pyttsx3
        print("✅ pyttsx3 TTS 可用")
        tts_available.append("pyttsx3")
    except ImportError:
        print("⚠️ pyttsx3 TTS 不可用")
    
    try:
        from gtts import gTTS
        print("✅ gTTS 可用")
        tts_available.append("gtts")
    except ImportError:
        print("⚠️ gTTS 不可用")
    
    if not tts_available:
        print("❌ 没有可用的 TTS 引擎！")
        return False
    
    try:
        import redis
        print("✅ Redis 客户端可用")
    except ImportError as e:
        print(f"❌ Redis 导入失败: {e}")
        return False
    
    return True

def test_whisper_model():
    """测试 Whisper 模型加载"""
    print("\n🎤 测试 Whisper 模型加载...")
    
    try:
        import whisper
        import torch
        
        # 使用最小的模型进行测试
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        model = whisper.load_model("tiny", device=device)
        print("✅ Whisper tiny 模型加载成功")
        
        # 创建测试音频 (无声音频)
        sample_rate = 16000
        duration = 3  # 3 seconds
        test_audio = np.zeros(sample_rate * duration, dtype=np.float32)
        
        # 测试转录
        result = model.transcribe(test_audio)
        print(f"✅ Whisper 转录测试完成")
        print(f"   结果: '{result['text']}'")
        print(f"   语言: {result.get('language', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper 测试失败: {e}")
        return False

def test_tts_engines():
    """测试所有可用的 TTS 引擎"""
    print("\n🔊 测试 TTS 引擎...")
    
    test_text = "Hello, this is a test of the text-to-speech system."
    
    # 测试 Chatterbox
    try:
        from chatterbox import TTSEngine
        engine = TTSEngine()
        audio_data = engine.synthesize(test_text)
        print(f"✅ Chatterbox TTS 测试成功 (生成了 {len(audio_data)} 字节音频)")
    except Exception as e:
        print(f"⚠️ Chatterbox TTS 测试失败: {e}")
    
    # 测试 pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        engine.save_to_file(test_text, tmp_path)
        engine.runAndWait()
        
        # 检查文件是否生成
        if os.path.exists(tmp_path):
            file_size = os.path.getsize(tmp_path)
            print(f"✅ pyttsx3 TTS 测试成功 (生成了 {file_size} 字节音频文件)")
            os.unlink(tmp_path)  # 清理临时文件
        else:
            print("❌ pyttsx3 TTS 未生成音频文件")
            
    except Exception as e:
        print(f"⚠️ pyttsx3 TTS 测试失败: {e}")
    
    # 测试 gTTS
    try:
        from gtts import gTTS
        from io import BytesIO
        
        tts = gTTS(text=test_text, lang='en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_data = audio_buffer.getvalue()
        
        print(f"✅ gTTS 测试成功 (生成了 {len(audio_data)} 字节音频)")
        
    except Exception as e:
        print(f"⚠️ gTTS 测试失败: {e}")
    
    return True

async def test_redis_connection():
    """测试 Redis 连接"""
    print("\n📡 测试 Redis 连接...")
    
    try:
        import redis.asyncio as redis
        
        # 尝试连接到本地 Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
        
        # 测试连接
        await redis_client.ping()
        print("✅ Redis 连接成功")
        
        # 测试基本操作
        test_key = "whisper_test_key"
        test_value = "test_value"
        
        await redis_client.set(test_key, test_value, ex=10)
        retrieved_value = await redis_client.get(test_key)
        
        if retrieved_value and retrieved_value.decode('utf-8') == test_value:
            print("✅ Redis 读写操作正常")
        else:
            print("❌ Redis 读写操作失败")
        
        # 清理
        await redis_client.delete(test_key)
        await redis_client.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        print("   请确保 Redis 服务正在运行")
        return False

def test_audio_processing():
    """测试音频处理功能"""
    print("\n🎵 测试音频处理...")
    
    try:
        import numpy as np
        
        # 模拟 PCM 音频数据
        sample_rate = 16000
        duration = 2  # 2 seconds
        frequency = 440  # A4 note
        
        # 生成正弦波
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_float = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # 转换为 16-bit PCM
        audio_int16 = (audio_float * 32767).astype(np.int16)
        
        # 转换为字节
        audio_bytes = audio_int16.tobytes()
        
        # Base64 编码
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"✅ 音频处理测试成功")
        print(f"   采样率: {sample_rate} Hz")
        print(f"   时长: {duration} 秒")
        print(f"   数据大小: {len(audio_bytes)} 字节")
        print(f"   Base64 大小: {len(audio_base64)} 字符")
        
        # 测试反向转换
        decoded_bytes = base64.b64decode(audio_base64)
        decoded_audio = np.frombuffer(decoded_bytes, dtype=np.int16)
        
        if len(decoded_audio) == len(audio_int16):
            print("✅ Base64 编解码正常")
        else:
            print("❌ Base64 编解码失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 音频处理测试失败: {e}")
        return False

async def test_service_integration():
    """测试完整的服务集成"""
    print("\n🚀 测试服务集成...")
    
    try:
        # 导入我们的服务
        from app.services.whisper_chatterbox_service import WhisperChatterboxService
        
        # 创建服务实例
        service = WhisperChatterboxService()
        print("✅ 服务初始化成功")
        
        # 测试会话创建
        session_data = await service.create_session("test_user_123", is_audio=True)
        session_id = session_data["session_id"]
        print(f"✅ 会话创建成功: {session_id}")
        
        # 测试文本消息
        text_message = "Hello, this is a test message."
        print(f"📤 发送文本消息: {text_message}")
        
        responses = []
        async for response in service.send_message(session_id, text_message, "text/plain"):
            responses.append(response)
            print(f"📨 收到响应: {response.get('type', 'unknown')}")
        
        print(f"✅ 文本消息处理完成，收到 {len(responses)} 个响应")
        
        # 测试会话清理
        await service.close_session(session_id)
        print("✅ 会话清理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 服务集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_system_info():
    """打印系统信息"""
    print("\n💻 系统信息:")
    print(f"   Python 版本: {sys.version}")
    print(f"   操作系统: {os.name}")
    print(f"   当前目录: {os.getcwd()}")
    
    # GPU 信息
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   GPU: 不可用")
    except:
        pass

async def main():
    """主测试函数"""
    print("🎤 Whisper + Chatterbox 功能测试")
    print("=" * 50)
    
    # 打印系统信息
    print_system_info()
    
    # 运行所有测试
    tests = [
        ("导入测试", test_imports),
        ("Whisper 模型测试", test_whisper_model),
        ("TTS 引擎测试", test_tts_engines),
        ("音频处理测试", test_audio_processing),
        ("Redis 连接测试", test_redis_connection),
        ("服务集成测试", test_service_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！您的 Whisper + Chatterbox 环境配置正确。")
        print("   现在可以启动应用并使用新的语音功能了。")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败。")
        print("   请根据上面的错误信息修复问题。")
        print("   参考 WHISPER_CHATTERBOX_IMPLEMENTATION_GUIDE.md 获取帮助。")

if __name__ == "__main__":
    # 运行测试
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc() 