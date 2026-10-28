# OpenClaw-Tracer

[English](README.md)

<p align="center">
  <img src="img/logo.png" alt="OpenClaw-Tracer Logo" width="220">
</p>

OpenClaw-Tracer 是一个基于 LiteLLM 的轻量级 LLM 数据采集与追踪服务，面向 **OpenClaw** 以及其他 Agent / LLM 系统。它作为一个兼容 OpenAI 接口的代理，透明地接收请求并自动捕获完整的对话与追踪数据（包括推理过程、工具调用、token 使用量、时延等），用于 SFT/RL 训练和可观测性。

## 特性

- **通用 LLM 支持**: 通过 LiteLLM 支持 OpenAI、Anthropic、Azure、Google、AWS Bedrock、vLLM、TGI 等
- **完整对话与追踪捕获**: 捕获系统提示词、多轮对话、工具调用、推理过程和 token 使用量
- **时间分片存储**: 数据以 Parquet 格式存储，支持自动按时间分片
- **OpenAI 兼容 API**: 可直接替换任何使用 OpenAI SDK 的客户端
- **Docker 支持**: 支持 Docker 和 Docker Compose 快速部署
- **实时采集**: 可配置缓冲区，支持立即写入或批量写入

## 架构

![架构](img/arch.png)

```
OpenClaw / Chatbox / 其他客户端 → OpenClaw-Tracer → 上游模型提供商
                            ↓
                    捕获: 系统提示、消息、工具调用、
                          推理过程、token 使用量
                            ↓
                    ParquetStore → data/spans/spans_*.parquet
```

## 快速开始

### 方式 1: Python (推荐用于开发)

```bash
# 创建环境
micromamba create -n OpenClaw-Tracer python=3.10 -y
micromamba activate OpenClaw-Tracer

# 安装依赖
pip install -r requirements.txt

# 配置模型（编辑 config/models.json，填入你的 API Key）
# 然后启动代理
python scripts/serve.py --config config/models.json --output-dir ./data
```

代理将启动在 `http://localhost:43886/v1`。

### 方式 2: Docker (推荐用于生产环境)

```bash
# 构建镜像
docker build -t openclaw-tracer:latest .

# 运行容器（需要先编辑配置文件）
docker run -d \
  --name openclaw-tracer \
  -p 43886:43886 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  openclaw-tracer:latest
```

### 方式 3: Docker Compose

```bash
# 使用默认设置启动
docker-compose up -d

# 或使用自定义环境变量
export HOST_PORT=43886
export BUFFER_SIZE=1
export TIME_WINDOW_MINUTES=30
docker-compose up -d
```

## 配置

创建 `config/models.json` 配置文件：

```json
{
    "model_list": [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "os.environ/OPENAI_API_KEY"
            }
        },
        {
            "model_name": "claude-3-5-sonnet",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": "os.environ/ANTHROPIC_API_KEY"
            }
        },
        {
            "model_name": "local-vllm",
            "litellm_params": {
                "model": "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
                "api_base": "http://localhost:8000/v1"
            }
        }
    ]
}
```

### 环境变量

#### 代理鉴权（必填）

| 变量 | 是否必填 | 说明 |
|------|----------|------|
| `PROXY_API_KEY` | **是** | 代理服务的鉴权密钥 |
| `LITELLM_MASTER_KEY` | 否 | LiteLLM 内置 master key。未设置时默认与 `PROXY_API_KEY` 相同 |
| `ADMIN_API_KEY` | 否 | 用于全部 `/admin/**`（routing / flush / batches / config 等）。未设置时默认 `PROXY_API_KEY + "admin"` 并打印 warning；与 `PROXY_API_KEY` 本身仍互不通用 |

**重要：** 如果未设置 `PROXY_API_KEY`，服务将无法启动。热切流与模型 CRUD 见 [docs/USAGE.md §9](docs/USAGE.md#9-热切流与模型-crud)。

#### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `43886` | 代理服务器端口 |
| `BUFFER_SIZE` | `1` | 写入前缓冲的记录数 (1 = 立即写入) |
| `TIME_WINDOW_MINUTES` | `30` | 创建新 Parquet 文件的间隔分钟数 |
| `FLUSH_INTERVAL_SECONDS` | `1800` | 定时 flush 间隔秒数 (0 = 禁用) |

#### 模型配置（基于环境变量）

| 变量 | 是否必填 | 说明 |
|------|----------|------|
| `TARGET_MODEL` | 是 | 要训练的目标模型名称 |
| `ORIGIN_MODEL` | 是 | 原始上游模型名称 |
| `API_MODE` | 是 | 原始 API 类型 (如 `openai`、`anthropic`、`custom`) |
| `API_URL` | 否 | 自定义 API 基础 URL (覆盖默认值) |
| `ACCESS_KEY` | 是 | API 访问密钥 |

#### 批处理模式（滚动窗口）

| 变量 | 是否必填 | 说明 |
|------|----------|------|
| `TRAJECTORY_BUFFER_SIZE` | 否 | 可占用的最大记录数（ready+collecting；0 = 禁用 batch） |

当 `TRAJECTORY_BUFFER_SIZE > 0` 时，每凑满一个 `BUFFER_SIZE` 的 batch 会标记为 **ready** 并保留在磁盘上供训练（可触发 `WORKFLOW_TRIGGER`），**不会自动删除**。训练结束后调用 `POST /admin/batches/{id}/release` 才会删除该批（并触发 `DATA_DELETE_CALLBACK`）。若所有槽位都被未释放的 ready batch 占满，采集会暂停（`BatchCapacityFull`），直到 release。

**双槽流水线（agent 控节奏）**：设 `BUFFER_SIZE > agent BS` 避免自动封批抢跑，`TRAJECTORY_BUFFER_SIZE = 3×BS` 约三槽。Agent 用 `POST /admin/flush` `{"seal_batch":true,"trigger_workflow":true}` 封批；orch 按 `batch_id` claim/release。背压：`inflight_count`（ready+in_training）> 2 时停喂。

**使用环境变量示例：**

```bash
export PROXY_API_KEY="your-secret-api-key"
export TARGET_MODEL="my-model"
export ORIGIN_MODEL="gpt-4"
export API_MODE="openai"
export API_URL="https://api.example.com/v1"
export ACCESS_KEY="sk-xxx"
# Agent BS=3500 → BUFFER_SIZE 更高；trajectory ≈ 3×BS
export BUFFER_SIZE=5000
export TRAJECTORY_BUFFER_SIZE=10500
export FLUSH_INTERVAL_SECONDS=300

python scripts/serve.py
```

## 使用方法

### 发送请求

使用任何兼容 OpenAI 的客户端，**需要在请求中携带鉴权信息**：

**方式 1: 使用 Authorization Header**

```bash
curl http://localhost:43886/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "你好!"}]
  }'
```

**方式 2: 使用 X-API-Key Header**

```bash
curl http://localhost:43886/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "你好!"}]
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:43886/v1",
    api_key="your-secret-api-key"  # 使用 PROXY_API_KEY
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好!"}]
)
```

**注意：** 以下路径无需鉴权：`/health`、`/status`、`/v1/models`

### 查看采集的数据

```bash
# 显示统计信息
python scripts/show_stats.py --output-dir ./data

# 查看最新记录
python scripts/show_spans.py --num 5

# 检查特定文件
python scripts/show_file.py data/spans/spans_20260310_143000.parquet --limit 10
```

## 数据结构

采集的数据存储在 `data/spans/spans_YYYYMMDD_HHMMSS.parquet`，包含以下字段：

| 列名 | 说明 |
|------|------|
| `name` | Span 名称 (如 "llm.completion") |
| `start_time` | 请求开始时间 (Unix 时间戳) |
| `end_time` | 请求结束时间 |
| `attributes` | JSON 字符串，包含所有数据 |
| `rollout_id` | 唯一的 rollout 标识符 |
| `attempt_id` | 唯一的 attempt 标识符 |

### Attributes JSON 结构

```json
{
  "llm.model": "gpt-4",
  "llm.request.system": "你是一个有用的助手...",
  "llm.request.messages": "[{\"role\": \"user\", \"content\": \"...\"}]",
  "llm.request.tools": "[...]",
  "llm.response.content": "响应文本...",
  "llm.response.tool_calls": "[...]",
  "llm.response.reasoning": "思考过程...",
  "llm.usage.prompt_tokens": 20,
  "llm.usage.completion_tokens": 100,
  "llm.usage.total_tokens": 120
}
```

## Docker 高级用法

### 使用自定义标签构建

```bash
docker build -t your-registry/openclaw-tracer:v1.0.0 .
```

### 使用自定义设置运行

```bash
docker run -d \
  --name openclaw-tracer \
  -p 43886:43886 \
  -v $(pwd)/data:/app/data \
  -e TARGET_MODEL="my-exposed-model" \
  -e ORIGIN_MODEL="gpt-4" \
  -e API_MODE="openai" \
  -e API_URL="https://api.openai.com/v1" \
  -e ACCESS_KEY="sk-xxx" \
  -e BUFFER_SIZE=100 \
  -e TRAJECTORY_BUFFER_SIZE=10000 \
  -e PROXY_API_KEY="your-proxy-api-key" \
  pyrominddynamics/openclaw-tracer:latest
```

**环境变量说明：**
| 变量 | 是否必填 | 默认值 | 说明 |
|------|----------|--------|------|
| `TARGET_MODEL` | 是 | - | 暴露给客户端的模型名称 |
| `ORIGIN_MODEL` | 是 | - | 实际要访问的上游模型 |
| `API_MODE` | 是 | - | 上游 API 格式 (如 `openai`、`anthropic`、`custom`) |
| `API_URL` | 是 | - | 上游模型的 API 基础 URL |
| `ACCESS_KEY` | 是 | - | 上游模型的 API 密钥 |
| `BUFFER_SIZE` | 否 | `1` | 刷盘前的缓冲区大小（1 = 立即写入） |
| `TRAJECTORY_BUFFER_SIZE` | 否 | `0` | 可占用的最大记录数（ready+collecting）；训完后 release（0 = 非 batch） |
| `TIME_WINDOW_MINUTES` | 否 | `30` | 定时刷盘间隔分钟数 |
| `PROXY_API_KEY` | 是 | - | 代理鉴权密钥 |
| `ADMIN_API_KEY` | 否 | - | 仅 `/admin/routing/**` |

**缓冲区行为说明：**
| 配置 | 行为 |
|------|------|
| 两者都为空（默认） | 立即写入模式 - 最安全，崩溃不丢失数据 |
| `BUFFER_SIZE=100`, `TRAJECTORY_BUFFER_SIZE=0` | 每 100 条记录写入一次，**数据持续累积不清理** |
| `BUFFER_SIZE=100`, `TRAJECTORY_BUFFER_SIZE=10000` | 批处理模式 — 最多约 100 个 ready 槽位供训练；仅 `/admin/batches/{id}/release` 删除 |

**注意：** ready batch **不会**自动删除；训练侧必须 release（或保持 `TRAJECTORY_BUFFER_SIZE=0` 自行管磁盘）。

**重要提示：**
- 必须设置 `PROXY_API_KEY`，否则容器将无法启动
- 必须设置 `API_MODE`，否则代理无法正常工作
- 默认每 30 分钟自动刷盘一次，防止系统崩溃导致数据丢失（可通过 `TIME_WINDOW_MINUTES` 修改）

### 使用 Docker Compose 配置环境文件

创建 `.env` 文件：
```
HOST_PORT=43886
PORT=43886
PROXY_API_KEY=your-secret-api-key
BUFFER_SIZE=1
TIME_WINDOW_MINUTES=30
```

然后启动：
```bash
docker-compose up -d
```

### 查看日志

```bash
# 容器日志
docker logs -f openclaw-tracer

# HTTP 访问日志
tail -f logs/http.jsonl

# 诊断日志
tail -f logs/diagnostic.log
```

### 停止容器

```bash
# 停止并删除容器
docker stop openclaw-tracer
docker rm openclaw-tracer

# 或使用 docker-compose
docker-compose down
```

## 导出训练数据

将采集的数据导出为 TRL 或 HuggingFace 数据集格式：

```python
import pandas as pd

# 加载 spans
df = pd.read_parquet("data/spans/spans_20260310_143000.parquet")

# 转换为训练格式
# (详细导出说明请参考 docs/USAGE.md)
```

## 健康检查

Docker 容器包含健康检查：

```bash
# 检查健康状态
docker inspect --format='{{.State.Health.Status}}' openclaw-tracer

# 手动健康检查
curl http://localhost:43886/v1/models
```

## 故障排除

### 代理无法启动
- 检查 `config/models.json` 是否存在且格式正确
- 验证环境变量中已设置 API Key

### 没有采集到数据
- 查看日志: `docker logs openclaw-tracer`
- 验证 BUFFER_SIZE 设置 (1 = 立即写入)
- 确保请求发送到正确的端口

### 权限错误
- 确保 `data/` 和 `logs/` 目录可写
- 检查 Docker 卷挂载权限

## Task 追踪

OpenClaw-Tracer 支持 task 追踪和奖励信号捕获，用于 RL 训练数据收集。

### 请求头

| 请求头 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| `X-Task-ID` | string | 否 | 任务标识符，用作 `rollout_id` |
| `X-Previous-Reward` | float | 否 | 上一步的奖励值 |
| `X-Task-No-Trace` | bool | 否 | 为 `1`/`true`/`yes`/`on` 时仍转发上游，但**不落盘** span |

### 示例

```bash
curl http://localhost:43886/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -H "X-Task-ID: my-task-123" \
  -H "X-Previous-Reward: 0.85" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

单次请求不落盘：

```bash
curl http://localhost:43886/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-key" \
  -H "X-Task-No-Trace: true" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"ping"}]}'
```

### API 端点

#### GET /tracer-version

获取 tracer 版本信息（无需认证）。

```bash
curl http://localhost:43886/tracer-version
```

#### POST /end_task

结束指定 task 会话。可选传入本轮 **最终 reward**（episode 级别）：优先使用字段 `final_reward`，若省略则用别名 `reward`；二者同时存在时以 `final_reward` 为准。未传入时响应里 `final_reward` 为 `null`。无效数字返回 400。

可选字段 **`gt`**（别名 `ground_truth`，字符串或 JSON）：写入该 `task_id` 下所有 span 的 `attributes.gt` / `attributes["task.gt"]`。不做 SFT 导出时可不传；若要从 span **单独**重建 `sft_daily` 样本，仅有 `X-Previous-Reward` / `final_reward` 不够，建议带上 GT。

```bash
curl -X POST http://localhost:43886/end_task \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{"task_id": "my-task-123", "final_reward": 1.0, "gt": "<think>...</think>\n\n{\"result\":\"defect\",\"reason\":\"擦伤\",\"boxes\":[]}"}'
```

#### POST /admin/flush

立即将内存中的 span 缓冲刷入 Parquet（需 `ADMIN_API_KEY`）。可选 `trigger_workflow`（query 或 JSON，默认 `false`）同时调度 buffer-full 训练触发脚本。

```bash
curl -X POST http://localhost:43886/admin/flush \
  -H "Authorization: Bearer your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"trigger_workflow": false}'
```

#### GET /admin/batches · POST /admin/batches/{id}/release

列出 held/ready 的 trajectory batch，或在训练完成后释放某个 **ready** batch（需 `ADMIN_API_KEY`）。release 会删除 `spans/<id>/`（含 media），并以 `reason=training_complete` 触发数据删除回调。

```bash
curl http://localhost:43886/admin/batches \
  -H "Authorization: Bearer your-admin-key"

curl -X POST http://localhost:43886/admin/batches/0/release \
  -H "Authorization: Bearer your-admin-key"
```

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `TASK_TIMEOUT_MINUTES` | 10 | task 超时时间（分钟） |

## 回调（Callbacks）

所有回调均为 **异步 / fire-and-forget**（失败只打日志，不影响客户端请求）。

### 1) 训练 workflow 触发（buffer 满 flush 后）

在因 `BUFFER_SIZE` 写满而 flush，或 `POST /admin/flush` 且 `trigger_workflow=true` 时触发。仅因时间窗口切文件的 flush **不会**触发。

| 变量 | 说明 |
|------|------|
| `ENABLE_WORKFLOW_TRIGGER` | 默认开启；设为 `0`/`false` 关闭 |
| `WORKFLOW_TRIGGER_SCRIPT` | shell 脚本路径（见 `workflow_trigger.py`） |

### 2) 路由变更（admin 热切流 / CRUD）

`/admin/routing/**` 成功 `reload_models` 后触发；payload 中 `api_key` 已脱敏。

| 变量 | 说明 |
|------|------|
| `ROUTING_CHANGE_CALLBACK_URL` | 可选，异步 HTTP POST JSON |
| `ROUTING_CHANGE_CALLBACK_TIMEOUT` | HTTP 超时秒（默认 `5`） |
| `ROUTING_CHANGE_CALLBACK_SCRIPT` | 可选 shell；argv 为 JSON 临时文件路径 |
| `ROUTING_CHANGE_CALLBACK_SCRIPT_TIMEOUT` | 脚本超时秒（默认 `30`） |

事件：`{"event":"routing.updated","reason":"...","previous":[],"current":[],"op":{...}}`。

### 3) 数据删除（训练完成后 release）

当 `TRAJECTORY_BUFFER_SIZE > 0`，满 batch 会一直保留，直到 `POST /admin/batches/{id}/release`。删除（parquet + 嵌套 `media/images/`）后以 `reason=training_complete` 调度本回调。

| 变量 | 说明 |
|------|------|
| `DATA_DELETE_CALLBACK_URL` | 可选，异步 HTTP POST JSON |
| `DATA_DELETE_CALLBACK_TIMEOUT` | HTTP 超时秒（默认 `5`） |
| `DATA_DELETE_CALLBACK_SCRIPT` | 可选 shell；argv 为 JSON 临时文件路径 |
| `DATA_DELETE_CALLBACK_SCRIPT_TIMEOUT` | 脚本超时秒（默认 `30`） |

事件示例：

```json
{
  "event": "data.deleted",
  "reason": "training_complete",
  "timestamp": "<iso8601>",
  "paths": ["/app/data/spans/0"],
  "batch_index": 0,
  "extra": {
    "current_batch_index": 3,
    "buffer_size": 100,
    "trajectory_buffer_size": 10000
  }
}
```

## 开发路线 / TODO

### 多轮对话跟踪

**当前状态**: 每个 LLM 请求独立捕获，使用唯一标识符。目前没有内置机制来关联属于同一 agent 会话/对话的多个请求。

**计划功能**:
- **会话管理**: 在单个 agent 对话会话中跟踪多次 LLM 调用
  - 支持 `traceparent` header（OpenTelemetry 标准）以兼容分布式追踪
  - 支持自定义 `X-Session-ID` header 用于基于 SDK 的会话管理
  - 基于 cookie 的服务端会话管理，适用于无 SDK 场景
  - 自动会话超时和清理（可配置，默认 30 分钟）

- **对话重建**: 查询并合并属于同一会话的所有 span
  - 新增查询方法：`store.get_session_spans(session_id)` 获取所有相关 LLM 调用
  - 使用 `sequence_id` 排序自动重建对话流程
  - 导出完整对话作为训练样本（多轮对话格式）
  - 可视化对话树用于调试复杂的 agent 工作流

- **增强元数据**: 额外的会话级别信息
  - 会话开始/结束时间戳
  - 每个会话的总 token 使用量
  - 对话轮次数
  - 会话元数据（用户 ID、agent ID、任务类型等）

**使用场景**:
- agent 进行多次 LLM 调用的多轮对话
- 需要跟踪 LLM 调用与工具执行交错情况的工具使用 agent
- 对话分析和模式识别
- 基于完整对话片段而非孤立轮次的训练

---

### 分布式部署支持

**当前状态**: 仅支持单实例部署。会话状态（如已实现）将存储在内存中，无法兼容负载均衡器后的多实例部署。

**计划功能**:
- **无状态架构**: 移除对内存会话状态的依赖
  - 所有会话上下文通过请求 header 传递（客户端管理会话）
  - 支持外部会话存储（Redis、PostgreSQL、etcd）
  - 通过环境变量配置会话后端

- **可扩展存储**: 分布式 Parquet 文件管理
  - 支持 S3 兼容对象存储（AWS S3、MinIO、GCS、Azure Blob）
  - 按日期/小时自动分区文件以高效查询
  - 可选的预写日志（WAL）用于崩溃恢复
  - 支持列式存储格式（Delta Lake、Apache Iceberg）

- **负载均衡器兼容**: 在负载均衡器后部署多个实例
  - 健康检查端点，包含就绪/存活探针
  - 优雅关闭，完成进行中的请求
  - 请求排空以实现零停机部署
  - 会话亲和性（可选）用于基于 cookie 的会话

- **可观测性**: 生产级监控和日志
  - OpenTelemetry 集成用于分布式追踪
  - Prometheus 指标导出（请求计数、延迟、错误率）
  - 结构化日志，JSON 输出
  - 集成日志聚合器（ELK、Loki、CloudWatch）

- **高可用**: 冗余和故障转移机制
  - 单例任务的领导选举（日志轮转、清理作业）
  - 热备实例和自动故障转移
  - 数据库连接池和熔断器
  - 上游 API 调用的重试逻辑和指数退避

**部署目标**:
- Kubernetes with Helm charts 用于便捷部署
- Docker Swarm 模式用于多主机部署
- AWS ECS/GCP Cloud Run/Azure Container Instances

## 许可证

MIT License - 详见 LICENSE 文件。

## 贡献

欢迎贡献！请随时提交 Pull Request。
