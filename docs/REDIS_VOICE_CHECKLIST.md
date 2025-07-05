# Redis 语音解决方案 - 部署检查清单

## ✅ 部署前检查

### 1. 环境变量确认
确保您的 `.env` 文件包含以下配置：

```bash
# === 必需的环境变量 ===
GEMINI_API_KEY=your_gemini_api_key
VOICE_MODEL=gemini-2.0-flash-exp
GEMINI_MODEL=gemini-2.0-flash

# === Redis 配置 (二选一) ===
# 选项A: Redis URL (推荐)
REDIS_URL=redis://your-redis-instance

# 选项B: Redis 单独配置
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# === 其他必需变量 ===
SUPABASE_URL=your_supabase_url
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
DATABASE_URL=your_database_url
```

### 2. 依赖安装
```bash
pip install -r requirements.txt
```

### 3. Redis 连接测试
```bash
# 测试 Redis 连接
python -c "
import redis
import os
from dotenv import load_dotenv
load_dotenv()

if os.getenv('REDIS_URL'):
    r = redis.from_url(os.getenv('REDIS_URL'))
else:
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD')
    )

try:
    r.ping()
    print('✅ Redis 连接成功')
except Exception as e:
    print(f'❌ Redis 连接失败: {e}')
"
```

## 🚀 启动服务

### 本地开发
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产部署 (Cloud Run)
```bash
# 构建和推送 Docker 镜像
docker build -t voice-assistant .
docker tag voice-assistant gcr.io/YOUR_PROJECT/voice-assistant
docker push gcr.io/YOUR_PROJECT/voice-assistant

# 部署到 Cloud Run
gcloud run deploy voice-assistant \
  --image gcr.io/YOUR_PROJECT/voice-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🧪 测试新的 Redis 语音功能

### 1. 健康检查
```bash
curl https://your-backend.run.app/api/voice-redis/health
```

**期望响应**:
```json
{
  "status": "healthy",
  "service": "redis_voice",
  "active_sessions": 0,
  "redis_connected": true
}
```

### 2. 创建语音会话
```bash
# 使用您的实际 token 和 user_id
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "https://your-backend.run.app/api/voice-redis/events/YOUR_USER_ID?is_audio=true"
```

**期望响应**: SSE 流包含 `session_created` 事件

### 3. 发送测试消息
```bash
# 使用从上一步获得的 session_id
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mime_type": "text/plain", "data": "Hello, voice assistant!"}' \
  "https://your-backend.run.app/api/voice-redis/send/SESSION_ID"
```

**期望响应**: 流式文本响应

## 🔧 前端集成

### 修改前端代码
在您的前端代码中，将原有的语音 API 路径替换：

```javascript
// === 旧的 API 路径 ===
// const eventSource = new EventSource(`/api/voice/events/${userId}?is_audio=true`);
// fetch(`/api/voice/send/${userId}`, {...})

// === 新的 Redis API 路径 ===
const eventSource = new EventSource(`/api/voice-redis/events/${userId}?is_audio=true`);

// 重要：监听 session_created 事件获取 session_id
let sessionId = null;
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'session_created') {
    sessionId = data.session_id;
    console.log('✅ Voice session created:', sessionId);
  }
  
  // 处理其他事件...
};

// 发送消息时使用 session_id (而不是 user_id)
const sendVoiceMessage = async (message) => {
  if (!sessionId) {
    console.error('❌ No active session');
    return;
  }
  
  const response = await fetch(`/api/voice-redis/send/${sessionId}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      mime_type: 'text/plain',
      data: message
    })
  });
  
  // 处理流式响应...
};
```

## 🐛 故障排除

### 常见问题

1. **Redis 连接失败**
   - 检查 REDIS_URL 或 REDIS_HOST/PORT/PASSWORD 配置
   - 确认 Redis 服务正在运行

2. **"Session not found" 错误**
   - 确认使用的是 session_id 而不是 user_id
   - 检查会话是否已过期 (默认1小时)

3. **Gemini API 错误**
   - 确认 GEMINI_API_KEY 正确
   - 检查模型名称配置

4. **认证失败**
   - 确认 SUPABASE_JWT_SECRET 正确
   - 检查前端传递的 token

### 日志监控
```bash
# 查看 Cloud Run 日志
gcloud run logs tail voice-assistant --limit=100

# 查看特定错误
gcloud run logs read voice-assistant --filter="ERROR"
```

## ✅ 成功指标

当系统正常运行时，您应该看到：

1. ✅ Redis 健康检查通过
2. ✅ 语音会话创建成功
3. ✅ 消息发送和响应正常
4. ✅ 日志中显示 "Created voice session" 和 "REDIS VOICE" 消息
5. ✅ 不再出现 "Voice session not found" 错误

## 📞 需要帮助？

如果遇到问题：
1. 检查上述故障排除部分
2. 查看详细的错误日志
3. 确认所有环境变量正确配置

这个解决方案应该彻底解决 InMemoryRunner 在 Cloud Run 中的问题！ 