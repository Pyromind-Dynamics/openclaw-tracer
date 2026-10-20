# OpenClaw-Tracer 使用文档

完整的 LLM 数据采集工具使用指南。

---

## 目录

1. [快速开始](#1-快速开始)
2. [Proxy 工作原理](#2-proxy-工作原理)
3. [启动 Proxy](#3-启动-proxy)
4. [配置模型访问](#4-配置模型访问)
5. [数据保存机制](#5-数据保存机制)
6. [OpenClaw 集成](#6-openclaw-集成)
7. [API 参考](#7-api-参考)
8. [常见问题](#8-常见问题)
9. [热切流与模型 CRUD](#9-热切流与模型-crud)

---

## 1. 快速开始

### 1.1 环境准备

```bash
# 创建并激活环境
micromamba create -n OpenClaw-Tracer python=3.10 -y
micromamba activate OpenClaw-Tracer

# 安装依赖
cd /workspace/OpenClaw-Tracer
pip install -r requirements.txt
```

### 1.2 最简单的启动方式

```bash
# 使用 CLI 启动（需要配置环境变量中的 API Key）
OpenClaw-Tracer serve --port 43886 --output-dir ./data
```

### 1.3 使用 Python 启动

```python
import asyncio
import openclaw_tracer as agl

async def main():
    # 配置模型列表
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
            },
        },
    ]

    # 创建存储和代理
    store = agl.ParquetStore(output_dir="data")
    proxy = agl.LLMProxy(port=43886, model_list=model_list, store=store)

    # 启动
    await proxy.start()
    print(f"Proxy running at {proxy.url}")

    # 持续运行
    await proxy.wait()

asyncio.run(main())
```

---

## 2. Proxy 工作原理

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Request                              │
│                    (OpenAI SDK / curl / etc.)                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OpenClaw-Tracer Proxy                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  LiteLLM Proxy Server                                          ││
│  │  - 接收标准 OpenAI API 请求                                    ││
│  │  - 路由到配置的模型                                             ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                 │                                   │
│  ┌──────────────────────────────▼──────────────────────────────────┐│
│  │  SpanLogger (Custom Callback)                                 ││
│  │  - 拦截请求前钩子 (pre_call)                                   ││
│  │  - 捕获请求参数 (messages, temperature, etc.)                   ││
│  │  - 拦截响应钩子 (post_call)                                    ││
│  │  - 捕获响应内容、token 使用量、token_ids                        ││
│  │  - 构造 Span 对象                                              ││
│  └──────────────────────────────┬──────────────────────────────────┘│
│                                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  ParquetStore                                                  ││
│  │  - 缓存 Span 数据                                              ││
│  │  - 批量写入 Parquet 文件                                       ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    External LLM Provider                            │
│  (OpenAI / Anthropic / Azure / vLLM / etc.)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
1. 客户端请求 → Proxy 接收
2. SpanLogger.async_log_success_event() 触发
3. 提取请求参数 → 构造 Span (部分)
4. LiteLLM 转发请求 → 外部 LLM
5. 外部 LLM 返回响应
6. SpanLogger 接收响应 → 完成 Span
7. Span → ParquetStore 缓冲区
8. 缓冲区满或手动 flush → 写入 Parquet 文件
```

### 2.3 Span 数据结构

```python
Span(
    name="llm.completion",
    context=SpanContext(trace_id="...", span_id="..."),
    start_time=1234567890.0,
    end_time=1234567891.5,
    kind=SpanKind.CLIENT,
    status=StatusCode.OK,
    attributes={
        "llm.model": "openai/gpt-4",
        "llm.messages": "[{'role': 'user', 'content': '...'}]",
        "llm.response.content": "Response text...",
        "llm.usage.prompt_tokens": 20,
        "llm.usage.completion_tokens": 100,
        "llm.usage.total_tokens": 120,
        "llm.prompt_token_ids": "[123, 456, ...]",  # 如果模型支持
    }
)
```

---

## 3. 启动 Proxy

### 3.1 使用配置文件启动

创建 `config/models.json`:

```json
{
    "model_list": [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "env:OPENAI_API_KEY"
            }
        },
        {
            "model_name": "claude-3-5-sonnet",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": "env:ANTHROPIC_API_KEY"
            }
        }
    ]
}
```

启动命令：

```bash
OpenClaw-Tracer serve --config config/models.json --output-dir ./data
```

### 3.2 使用 Python 脚本启动

```python
# examples/start_proxy.py
import asyncio
import os
import openclaw_tracer as agl

async def main():
    # 从环境变量读取 API Key
    model_list = []

    if os.getenv("OPENAI_API_KEY"):
        model_list.append({
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4"},
        })

    if os.getenv("ANTHROPIC_API_KEY"):
        model_list.append({
            "model_name": "claude-3-5-sonnet",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022"
            },
        })

    if not model_list:
        raise ValueError("请设置至少一个 API Key (OPENAI_API_KEY 或 ANTHROPIC_API_KEY)")

    # 创建代理
    store = agl.ParquetStore(output_dir="data", buffer_size=50)
    proxy = agl.LLMProxy(port=43886, model_list=model_list, store=store)

    await proxy.start()

    print("=" * 60)
    print("Proxy Server Started")
    print("=" * 60)
    print(f"URL: {proxy.url}")
    print(f"V1 Endpoint: {proxy.v1_url}")
    print(f"Models: {[m['model_name'] for m in model_list]}")
    print("=" * 60)

    # 等待中断信号
    try:
        await proxy.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await proxy.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
micromamba run -n OpenClaw-Tracer python examples/start_proxy.py
```

---

## 4. 配置模型访问

### 4.1 云服务提供商配置

#### OpenAI

```python
{
    "model_name": "gpt-4",
    "litellm_params": {
        "model": "openai/gpt-4",
        "api_key": "env:OPENAI_API_KEY"  # 从环境变量读取
    }
}
```

#### Azure OpenAI

```python
{
    "model_name": "azure-gpt-4",
    "litellm_params": {
        "model": "azure/gpt-4",
        "api_key": "env:AZURE_API_KEY",
        "api_base": "https://your-resource.openai.azure.com",
        "api_version": "2023-05-15"
    }
}
```

#### Anthropic Claude

```python
{
    "model_name": "claude-3-5-sonnet",
    "litellm_params": {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "api_key": "env:ANTHROPIC_API_KEY"
    }
}
```

### 4.2 本地模型配置 (vLLM)

#### 启动 vLLM 服务

```bash
# 启动 vLLM 服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

#### 配置 Proxy 连接本地 vLLM

```python
{
    "model_name": "local-qwen",
    "litellm_params": {
        "model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
        "api_base": "http://localhost:8000/v1"
    }
}
```

### 4.3 完整配置示例

```python
# config/models_full.json
{
    "model_list": [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "env:OPENAI_API_KEY"
            }
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "env:OPENAI_API_KEY"
            }
        },
        {
            "model_name": "claude-sonnet",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": "env:ANTHROPIC_API_KEY"
            }
        },
        {
            "model_name": "local-llama-3",
            "litellm_params": {
                "model": "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
                "api_base": "http://localhost:8000/v1"
            }
        }
    ]
}
```

### 4.4 验证配置

```bash
# 启动 proxy 后，测试连接
curl http://localhost:43886/v1/models

# 发送测试请求
curl http://localhost:43886/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

---

## 5. 数据保存机制

### 5.1 Parquet 文件结构

数据收集后，会在输出目录生成以下文件：

```
data/
├── sft_records.parquet      # SFT 训练数据
├── rl_records.parquet       # RL 训练数据
└── spans.parquet            # 原始 Span 数据
```

### 5.2 数据写入流程

```python
# 1. Span 被捕获
span = Span(...)

# 2. 添加到存储（进入缓冲区）
await store.add_span(span)

# 3. 缓冲区满时自动刷新，或手动刷新
await store.flush()

# 4. 写入 Parquet 文件
#    - 如果文件存在，读取现有数据并追加
#    - 如果文件不存在，创建新文件
```

### 5.3 SFT 数据格式

```python
# sft_records.parquet 的列结构
{
    "id": "abc123",
    "prompt": "What is 2+2?",
    "response": "2+2 equals 4.",
    "thought_chain": null,  # 预留给 OpenClaw
    "model": "gpt-4",
    "rollout_id": "ro-xyz",
    "timestamp": 1234567890.0,
    "metadata": "{}",
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
}
```

### 5.4 RL 数据格式

```python
# rl_records.parquet 的列结构
{
    "rollout_id": "ro-abc",
    "attempt_id": "at-123",
    "trajectory": "[{...}]",  # JSON 序列化的 Span 列表
    "reward": 0.95,
    "query": "原始问题",
    "response": "最终答案",
    "model": "gpt-4",
    "timestamp": 1234567890.0,
    "metadata": "{}"
}
```

### 5.5 缓冲区控制

```python
# 创建带缓冲控制的存储
store = agl.ParquetStore(
    output_dir="data",
    buffer_size=100,       # 缓冲 100 条记录后自动写入
    auto_flush=True        # 自动刷新
)

# 手动刷新
await store.flush()

# 关闭时自动刷新
await store.close()
```

---

## 6. OpenClaw 集成

### 6.1 什么是 OpenClaw

OpenClaw 是一个面向多 Agent 的自治智能编排平台，用于在多个渠道运行长期在线的智能 Agent。它提供统一的 Agent 生命周期管理、工具与技能系统、记忆层（短期与长期记忆）以及模型层抽象，支持复杂多步工作流、任务分解和决策执行。将 OpenClaw 与 OpenClaw-Tracer 集成时，可以在不侵入业务逻辑的前提下，为这些 Agent 的推理过程、工具调用、成本与时延等生成可训练、可分析的追踪数据，用于后续的 SFT/RL 训练和可观测性分析。

### 6.2 数据结构准备

`SFTRecord` 已经预留了 `thought_chain` 字段：

```python
class SFTRecord(BaseModel):
    id: str
    prompt: str
    response: str
    thought_chain: Optional[str] = None  # ← OpenClaw 集成点
    model: Optional[str] = None
    ...
```

### 6.3 集成方式一：后处理

```python
# examples/openclaw_post_process.py
import asyncio
import openclaw_tracer as agl

async def add_thought_chains():
    """将 OpenClaw 的思考链添加到已有数据"""

    # 1. 加载已收集的数据
    store = agl.ParquetStore(output_dir="data")
    records = await store.get_sft_records(limit=1000)

    # 2. 为每条记录关联思考链
    for record in records:
        # 假设你有 OpenClaw 的 API 或存储
        thought_chain = await get_thought_chain_from_openclaw(
            prompt=record.prompt,
            response=record.response
        )

        # 更新记录
        record.thought_chain = thought_chain

    # 3. 保存更新后的数据
    exporter = agl.HFExporter()
    await exporter.export_sft(records, "data/with_thoughts.parquet")

async def get_thought_chain_from_openclaw(prompt: str, response: str) -> str:
    """从 OpenClaw 获取思考链"""
    # TODO: 实现 OpenClaw API 调用
    # 这里需要根据 OpenClaw 的实际 API 来实现
    return "Thinking process..."

asyncio.run(add_thought_chains())
```

### 6.4 集成方式二：实时捕获

```python
# examples/openclaw_realtime.py
import asyncio
from agent_data_collect.types.core import SFTRecord

async def collect_with_openclaw():
    """在数据收集时同时捕获思考链"""

    # 假设 OpenClaw 提供了一个装饰器或中间件
    @openclaw.capture_thought
    async def generate_with_thought(prompt: str, model: str):
        """使用 OpenClaw 包装的生成函数"""
        # 调用 LLM
        response = await call_llm(prompt, model)

        # OpenClaw 会自动捕获思考过程
        return response, openclaw.get_thought_chain()

    # 使用
    prompts = ["What is 2+2?", "Explain quantum computing"]

    for prompt in prompts:
        response, thoughts = await generate_with_thought(prompt, "gpt-4")

        # 创建包含思考链的记录
        record = SFTRecord(
            prompt=prompt,
            response=response,
            thought_chain=thoughts,  # 直接保存思考链
            model="gpt-4"
        )

        await store.add_sft_record(record)

asyncio.run(collect_with_openclaw())
```

### 6.5 导出带思考链的数据

```python
# 导出为 TRL 格式（支持思考链）
exporter = agl.TRLExporter(sft_format="conversational")

# 在 conversational 格式中，思考链可以作为 system 消息
await exporter.export_sft(
    records_with_thoughts,
    "data/trl_with_thoughts.parquet",
    include_thoughts=True  # 包含思考链
)
```

### 6.6 思考链数据在训练中的使用

```python
# 使用包含思考链的数据进行训练
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("parquet", data_files="data/with_thoughts.parquet")

# 方法1：将思考链作为额外的训练信号
def format_with_thoughts(example):
    messages = [
        {"role": "system", "content": example.get("thought_chain", "")},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"messages": messages}

dataset = dataset.map(format_with_thoughts)

trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    dataset_text_field="messages"
)
trainer.train()
```

---

## 7. API 参考

### 7.1 核心类

#### LLMProxy

```python
class LLMProxy:
    def __init__(
        self,
        port: Optional[int] = None,
        model_list: Optional[List[Dict]] = None,
        store: Optional[StorageBackend] = None,
        host: str = "0.0.0.0",
        num_workers: int = 1,
    ): ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def wait(self) -> None: ...

    @property
    def url(self) -> str: ...

    @property
    def v1_url(self) -> str: ...
```

#### ParquetStore

```python
class ParquetStore(StorageBackend):
    def __init__(
        self,
        output_dir: str | Path = "data",
        buffer_size: int = 100,
        auto_flush: bool = True,
    ): ...

    async def add_span(self, span: Span) -> None: ...
    async def add_sft_record(self, record: SFTRecord) -> None: ...
    async def add_rl_record(self, record: RLRecord) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...

    @property
    def stats(self) -> Dict[str, Any]: ...
```

#### TRLExporter

```python
class TRLExporter(DataExporter):
    def __init__(self, sft_format: str = "conversational"): ...

    async def export_sft(
        self,
        records: List[SFTRecord],
        output_path: str | Path,
        **kwargs
    ) -> None: ...
```

### 7.2 CLI 命令

```bash
# 启动代理
OpenClaw-Tracer serve [OPTIONS]

# 导出数据
OpenClaw-Tracer export INPUT_PARQUET OUTPUT_PATH [OPTIONS]

# 检查数据
OpenClaw-Tracer inspect PARQUET_PATH [OPTIONS]

# 查看统计
OpenClaw-Tracer stats [OPTIONS]
```

---

## 8. 常见问题

### Q1: 如何查看收集了多少数据？

```bash
OpenClaw-Tracer stats --output-dir ./data
```

### Q2: 如何修改缓冲区大小？

```python
store = agl.ParquetStore(output_dir="data", buffer_size=500)
```

### Q3: 支持哪些 LLM 提供商？

查看 LiteLLM 支持的提供商：https://docs.litellm.ai/

支持包括但不限于：
- OpenAI
- Anthropic
- Azure OpenAI
- Google (Vertex AI, Gemini)
- AWS Bedrock
- vLLM (本地部署)
- TGI (Text Generation Inference)

### Q4: 数据如何用于 TRL 训练？

```python
from datasets import load_dataset
from trl import SFTTrainer

# 加载导出的数据
dataset = load_dataset("parquet", data_files="data/trl_format/train.parquet")

# 训练
trainer = SFTTrainer(
    "Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset["train"],
    dataset_text_field="messages"
)
trainer.train()
```

### Q5: 如何同时收集多个模型的数据？

在 model_list 中配置多个模型：

```python
model_list = [
    {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
    {"model_name": "claude", "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}},
    {"model_name": "local", "litellm_params": {"model": "hosted_vllm/llama-3", "api_base": "http://localhost:8000/v1"}},
]
```

请求时使用对应的 model_name。

### Q6: 数据存储在哪里？

默认存储在 `output_dir` 指定的目录下：
- `sft_records.parquet` - SFT 训练数据
- `rl_records.parquet` - RL 训练数据
- `spans.parquet` - 原始 Span 数据

### Q7: 如何处理大量数据？

```python
# 1. 增大缓冲区，减少 I/O 次数
store = agl.ParquetStore(output_dir="data", buffer_size=1000)

# 2. 定期手动刷新
await store.flush()

# 3. 按时间或大小分割文件
# （需要自行实现文件轮转逻辑）
```

### Q8: OpenClaw 思考链占用多少存储？

思考链通常是文本数据，大小取决于思考的复杂度：
- 简单问题：~1KB
- 复杂推理：~10-50KB

建议：
1. 只在需要时保存完整思考链
2. 考虑压缩或摘要
3. 分开存储（thinking 字段单独一个 Parquet 文件）

---

## 9. 热切流与模型 CRUD

对外 **逻辑模型名**（`model_name`，客户端 `/v1/models` 与请求里的 `model`）保持稳定；运维通过 Admin API **增删改查后端 deployment**（`api_base` + 上游 model + `weight`），实现 A/B、热迁移、热切流。

### 9.1 鉴权分离

| 密钥 | 用途 |
|------|------|
| `PROXY_API_KEY` | 客户端调 `/v1/chat/completions` 等 |
| `ACCESS_KEY` | 上游模型服务密钥（写在 deployment `litellm_params.api_key`） |
| `ADMIN_API_KEY` | **仅** `/admin/routing/**`；未设置则 admin 返回 503 |

`PROXY_API_KEY` 与 `ADMIN_API_KEY` **互不通用**。

```bash
export ADMIN_API_KEY="your-admin-key"
```

### 9.2 推荐切流顺序（对外名不变）

1. `POST /admin/routing/deployments` 追加新后端  
2. （可选）`PATCH /admin/routing/weights` 灰度迁移  
3. `DELETE /admin/routing/deployments/{id}` 删除旧后端（默认 `drain=true`，等在途请求结束）

### 9.3 curl 示例

```bash
BASE=http://localhost:43886
ADMIN_HDR="Authorization: Bearer ${ADMIN_API_KEY}"

# 全量查询
curl -s -H "$ADMIN_HDR" "$BASE/admin/routing" | jq .

# 同一逻辑名下追加后端（A/B）
curl -s -X POST -H "$ADMIN_HDR" -H "Content-Type: application/json" \
  "$BASE/admin/routing/deployments" \
  -d '{
    "reason": "ab_test",
    "deployment": {
      "model_name": "pcb_avi_sft_merge_v10",
      "litellm_params": {
        "model": "openai/pcb_avi_sft_merge_v10",
        "api_base": "https://backend-b.example/v1",
        "api_key": "'"${ACCESS_KEY}"'",
        "weight": 50
      }
    }
  }' | jq .

# 热迁移改权重
curl -s -X PATCH -H "$ADMIN_HDR" -H "Content-Type: application/json" \
  "$BASE/admin/routing/weights" \
  -d '{
    "reason": "migration",
    "model_name": "pcb_avi_sft_merge_v10",
    "weights": [
      {"id": "<deploy-a-uuid>", "weight": 10},
      {"id": "<deploy-b-uuid>", "weight": 90}
    ]
  }' | jq .

# 删除旧后端（默认 drain）
curl -s -X DELETE -H "$ADMIN_HDR" -H "Content-Type: application/json" \
  "$BASE/admin/routing/deployments/<deploy-a-uuid>" \
  -d '{"reason":"cutover","drain":true}' | jq .
```

### 9.4 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ADMIN_API_KEY` | （空） | 启用 admin CRUD |
| `ROUTING_STATE_PATH` | `{output_dir}/routing_state.json` | 路由落盘；启动优先恢复 |
| `ROUTING_DRAIN_TIMEOUT_SECONDS` | `120` | drain 超时；超时返回 409 且不提交 |
| `ROUTING_CHANGE_CALLBACK_URL` | （空） | 变更成功后异步 POST JSON（密钥已脱敏） |
| `ROUTING_CHANGE_CALLBACK_TIMEOUT` | `5` | HTTP 回调超时（秒） |
| `ROUTING_CHANGE_CALLBACK_SCRIPT` | （空） | 变更成功后异步执行 shell，argv 传入 JSON 临时文件路径 |
| `ROUTING_CHANGE_CALLBACK_SCRIPT_TIMEOUT` | `30` | shell 回调超时（秒） |

URL 与 SCRIPT 都空则跳过回调；都配置则先调度 HTTP 再调度 shell。
