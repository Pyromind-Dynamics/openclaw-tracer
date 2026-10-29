# EchoMind 产品文档待补充接口

EchoMind 实例底层即 **openclaw-tracer**。控制台 / SDK 文档目前几乎只写了「创建实例 → 满 Training Batch Size → 跑脚本」，下列 **实例内 API** 需要补进产品文档（建议独立「Tracer / 实例内 API」章节）。

鉴权：`PROXY` = 实例 `secret_key`（`PROXY_API_KEY`）；`ADMIN` = `ADMIN_API_KEY`（未设置时默认 `PROXY_API_KEY + "admin"`；与 PROXY 本身互不通用）；`—` = 无需认证。

字段映射：控制台 **Training Batch Size** ↔ `BUFFER_SIZE` / `buffer_size`；**Trajectory Buffer Size** ↔ `TRAJECTORY_BUFFER_SIZE`；**自定义运行脚本** ↔ `WORKFLOW_TRIGGER_SCRIPT`。

---

## P0 — 训练闭环

| 方法 | 路径 | 鉴权 | 要点 |
|------|------|------|------|
| `GET` | `/status` | — | 采集水位（capacity / held / ready / inflight），**不是**实例 running/paused |
| `POST` | `/admin/flush` | ADMIN | body：`seal_batch`、`trigger_workflow`；槽满 `409` |
| `GET` | `/admin/batches` | ADMIN | 列出 ready / in_training / collecting |
| `POST` | `/admin/batches/{id}/claim` | ADMIN | ready → in_training |
| `POST` | `/admin/batches/{id}/release` | ADMIN | 训完删盘；触发 data-delete 回调 |

闭环：`满批 →（可选 trigger）→ claim → 训练 → release`。

---

## P0 — 运行时配置（热更新）

| 方法 | 路径 | 鉴权 | 要点 |
|------|------|------|------|
| `GET` | `/admin/config` | ADMIN | 读当前 `buffer_size` / `trajectory_buffer_size` / `workflow_trigger_script` |
| `PATCH` | `/admin/config` | ADMIN | 热更新上述三项（可部分更新；亦接受 `BUFFER_SIZE` 等大写别名） |

```bash
# 读取
curl -s http://$TRACER/admin/config \
  -H "Authorization: Bearer $ADMIN_API_KEY"

# 更新（示例）
curl -X PATCH http://$TRACER/admin/config \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "buffer_size": 8000,
    "trajectory_buffer_size": 24000,
    "workflow_trigger_script": "/workspace/trigger/train.sh"
  }'
```

约束：

- `trajectory_buffer_size > 0` 时须 `>= buffer_size`
- 仍有 ready/in_training 时不可把 `trajectory_buffer_size` 设为 `0`（先 release）
- 非法参数返回 `400` + `code: invalid_config`

产品侧应对齐：创建实例写入这些值；控制台/SDK 变更训练参数时调 `PATCH /admin/config`（不必重建实例）。

---

## P1 — 鉴权与脚本

| 项 | 说明 |
|----|------|
| `PROXY_API_KEY` | = 创建后下发的 `secret_key`；调 `/v1/*`、`/end_task` |
| `ADMIN_API_KEY` | 全部 `/admin/**`（flush、batches、config、routing、spans）；未设则默认 `PROXY_API_KEY + "admin"` |
| `LITELLM_MASTER_KEY` | LiteLLM 启动必需；未设则默认与 `PROXY_API_KEY` 相同（entrypoint / `serve.py`） |
| `WORKFLOW_TRIGGER_SCRIPT` | = `custom_runtime_script_path`；满批或 flush+trigger 时异步执行 |
| `POST /admin/spans/release` | ADMIN；`TRAJECTORY_BUFFER_SIZE=0` 时按 shard 释放 |

---

## P2 — 热切流

| 方法 | 路径 | 鉴权 |
|------|------|------|
| `GET` / `PUT` | `/admin/routing` | ADMIN |
| `POST` | `/admin/routing/deployments` | ADMIN |
| `GET` / `PUT` / `PATCH` / `DELETE` | `/admin/routing/deployments/{id}` | ADMIN |
| `GET` | `/admin/routing/models/{model_name}` | ADMIN |
| `PATCH` | `/admin/routing/weights` | ADMIN |

对外逻辑模型名不变；删除默认 `drain=true`。

---

## P2 — Task / RL

| 项 | 鉴权 | 要点 |
|----|------|------|
| Header `X-Task-ID` | PROXY | → `rollout_id` |
| Header `X-Previous-Reward` | PROXY | 逐步 reward |
| Header `X-Task-Trace` | PROXY | **opt-in 落盘**；缺省只转发不写 span |
| `POST /end_task` | PROXY | `final_reward` / `gt` |
| `GET /tracer-version` | — | 版本探测 |

---

## P3 — 回调（附录即可）

| 回调 | 触发 |
|------|------|
| Workflow（`WORKFLOW_TRIGGER_SCRIPT`） | 满 `BUFFER_SIZE` 或 flush+`trigger_workflow`；时间窗 flush **不**触发 |
| `DATA_DELETE_CALLBACK_*` | `release` 后 |
| `ROUTING_CHANGE_CALLBACK_*` | 热切成功后 |

---

## 不必写入「实例内 Tracer API」的内容

- 平台 SDK：`create` / `list` / `pause` / `resume` / `delete` / `get_internal_ip` / `update`
- `training_round`、`time_per_round`、`training_save_path`（编排层）
- Automatic 执行模式（尚未落地）
