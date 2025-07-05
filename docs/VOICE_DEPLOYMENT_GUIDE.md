# Voice Assistant 部署解决方案

## 问题分析

您的架构中 InMemoryRunner 在 Google Cloud Run 中失效的**真正**原因：

- **当前架构**: Frontend (Next.js) → SSE 连接 + HTTP POST → Backend (FastAPI) → ADK Agent (InMemoryRunner)
- **根本问题**: InMemoryRunner 设计用于本地开发，不适合云端无状态部署
- **具体表现**: 
  - ❌ 容器重启导致内存状态丢失
  - ❌ 网络层负载均衡可能路由到不同实例
  - ❌ Cloud Run 的无状态特性与 InMemoryRunner 的有状态设计冲突
- **Teresa 的发现**: 即使单进程部署（`gunicorn -w 1`）+ 会话亲和性也无法解决

## 解决方案对比

### 1. ~~单进程部署~~ ❌ 无效
**Teresa 已验证**: 单进程部署（`gunicorn -w 1`）+ 会话亲和性仍然失败
**原因**: InMemoryRunner 的根本设计不适合 Cloud Run 无状态环境
**结论**: 这个方案无法解决问题，需要采用其他方案

### 2. Redis 状态管理 (推荐立即解决) 🚀
**状态**: ✅ 已准备就绪 - `redis_voice_service.py` + `voice_redis.py`
**更改**: 使用 Redis 替代 InMemoryRunner 存储会话状态

**优点**: 
- 解决 Cloud Run 无状态问题
- 保持现有前端架构 (SSE + HTTP POST)
- 支持多进程和容器重启
- 会话状态持久化

**使用方式**:
```javascript
// 前端只需改变URL路径
// 从: /api/voice/events/USER_ID
// 改为: /api/voice-redis/events/USER_ID

// 从: /api/voice/send/USER_ID  
// 改为: /api/voice-redis/send/SESSION_ID (注意:使用session_id)
```

### 3. WebSocket 架构 🔄
**状态**: ✅ 已准备就绪 - `voice_websocket.py`
**更改**: 将 SSE + HTTP POST 改为单一 WebSocket 连接

**前端修改示例**:
```javascript
// 替代 SSE + fetch 的 WebSocket 连接
const token = localStorage.getItem('supabase_token');
const ws = new WebSocket(`wss://your-backend.run.app/ws/voice/${userId}?token=${token}&is_audio=true`);

// 发送消息
ws.send(JSON.stringify({
  mime_type: "text/plain",
  data: "Hello, voice assistant!"
}));

// 接收响应
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

**优点**: 
- 保证会话一致性
- 更好的实时性能
- 支持多进程部署

### 4. Vertex AI Agent Engine (长期方案) 🌟
**状态**: 🔄 需要配置
**更改**: 迁移到 Google 官方推荐的 ADK 部署平台

**环境变量设置**:
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

**部署步骤**:
```bash
# 1. 启用 Vertex AI Agent Engine API
gcloud services enable aiplatform.googleapis.com

# 2. 创建 Agent Engine 应用
gcloud ai agents create \
  --agent-name="voice-assistant" \
  --model="gemini-2.0-flash-live-001" \
  --location="us-central1"

# 3. 部署 Agent
gcloud ai agents deploy \
  --agent-name="voice-assistant" \
  --source="backend/" \
  --location="us-central1"
```

**优点**: 
- Google 官方支持
- 专门为 ADK 优化
- 自动扩缩容
- 更好的会话管理

## 推荐部署路径

### 阶段 1: 架构重构 (必需) 🔄
单进程部署已证明无效，必须更换架构

### 阶段 2: 架构重构 (必需选择之一)
- **Option A**: Redis 状态管理 (推荐，立即可用) 🚀
- **Option B**: WebSocket 架构 (需要前端修改)
- **Option C**: 迁移到 Vertex AI Agent Engine (长期最佳)

### 阶段 3: 性能优化
根据用户规模调整资源配置

## 当前文件状态

### 已修改的文件
- ✅ `Dockerfile` - 改为单进程部署
- ✅ `voice_websocket.py` - WebSocket 实现
- ✅ `supabase_auth.py` - WebSocket 认证支持
- ✅ `main.py` - 包含 WebSocket 路由
- ✅ `vertex_ai_agent_service.py` - Vertex AI 服务框架
- ✅ `requirements.txt` - 添加 Vertex AI 依赖

### 原有文件 (保持不变)
- `adk_voice_service.py` - 原有 InMemoryRunner 实现
- `voice.py` - 原有 SSE + HTTP 实现

## 测试验证

### 验证单进程修复
```bash
# 检查部署日志
gcloud run logs read voice-assistant --limit=50

# 测试语音功能
curl -X POST "https://your-backend.run.app/api/voice/send/USER_ID" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mime_type": "text/plain", "data": "Hello test"}'
```

### 验证 WebSocket
```bash
# 使用 wscat 测试 WebSocket
npm install -g wscat
wscat -c "wss://your-backend.run.app/ws/voice/USER_ID?token=YOUR_TOKEN&is_audio=true"
```

## 故障排除

### 常见问题
1. **"Voice session not found"** - 检查是否仍在使用多进程部署
2. **WebSocket 连接失败** - 检查 CORS 和认证配置
3. **Vertex AI 权限错误** - 确保服务账号有正确权限

### 日志监控
```bash
# Cloud Run 日志
gcloud run logs tail voice-assistant

# Vertex AI 日志 (如适用)
gcloud logging read "resource.type=vertex_ai_agent"
```

## 下一步

1. **立即**: 重新部署使用修改后的 Dockerfile
2. **短期**: 选择 WebSocket 或 Vertex AI 方案
3. **长期**: 监控性能并根据需要优化

需要协助实施任何解决方案，请告知具体需求！ 