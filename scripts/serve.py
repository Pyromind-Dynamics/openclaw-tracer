#!/usr/bin/env python3
"""
启动 OpenClaw-Tracer 代理服务器

替代命令: agl-collect serve --config CONFIG --output-dir DIR --port PORT
"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# INFO级别日志透出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)


import openclaw_tracer


# 全局变量用于信号处理
_shutdown_event = asyncio.Event()
_store_ref = None


def _signal_handler(signum, frame):
    """处理中断信号，确保数据保存

    当收到 SIGINT (Ctrl+C) 或 SIGTERM 信号时，确保 buffer 中的数据被刷新到磁盘。
    """
    print(f"\n收到信号 {signum}，正在保存数据...")
    global _store_ref
    if _store_ref:
        try:
            # 尝试获取当前运行的事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，创建一个刷新任务
                loop.call_soon(lambda: asyncio.create_task(_store_ref.flush()))
            else:
                # 如果循环已停止，运行刷新
                loop.run_until_complete(_store_ref.flush())
        except Exception as e:
            print(f"刷新数据时出错: {e}")
    _shutdown_event.set()


# 注册信号处理器
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


async def main(
    config: str | None = None,
    output_dir: str = "data",
    port: int | None = None,
    host: str = "0.0.0.0",
    buffer_size: int = 1,
    time_window_minutes: int = 5,
    log_file: str | None = None,
    trajectory_buffer_size: int = 0,
    flush_interval_seconds: int = 1800,
    proxy_api_key: str | None = None,
    task_timeout_minutes: int = 10,
):
    """启动代理服务器

    Args:
        config: 配置文件路径 (models.json)
        output_dir: 数据输出目录
        port: 监听端口 (默认自动选择)
        host: 监听地址
        buffer_size: 缓冲区大小 (默认: 1，每次请求立即写入)
        time_window_minutes: 时间窗口分钟数 (默认: 5)
        log_file: HTTP访问日志文件路径 (JSONL格式)
        trajectory_buffer_size: 总保留数据条数 (0=不启用分批)
        flush_interval_seconds: 定时 flush 间隔秒数 (默认: 1800 = 30分钟, 0=禁用)
        proxy_api_key: 代理鉴权密钥 (必填，从环境变量 PROXY_API_KEY 读取)
        task_timeout_minutes: 任务超时分钟数 (默认: 10)
    """
    # 获取并校验 PROXY_API_KEY
    if proxy_api_key is None:
        proxy_api_key = os.getenv("PROXY_API_KEY")

    if not proxy_api_key:
        print("✗ 错误: 未设置 PROXY_API_KEY 环境变量")
        print("  请设置环境变量: export PROXY_API_KEY=your-api-key")
        print("  或通过参数传入: --proxy-api-key your-api-key")
        sys.exit(1)

    print(f"✓ 代理鉴权已启用 (PROXY_API_KEY: {proxy_api_key[:8]}...)")

    # LiteLLM refuses to boot without a master key; default to PROXY_API_KEY so
    # EchoMind / k8s deployments that only inject PROXY_API_KEY still start.
    if not (os.getenv("LITELLM_MASTER_KEY") or "").strip():
        os.environ["LITELLM_MASTER_KEY"] = proxy_api_key
        print("⚠ LITELLM_MASTER_KEY 未设置，默认与 PROXY_API_KEY 相同")

    admin_api_key = (os.getenv("ADMIN_API_KEY") or "").strip() or None
    if admin_api_key:
        print("✓ Admin API 已启用 (ADMIN_API_KEY set)")
    else:
        admin_api_key = f"{proxy_api_key}admin"
        print("⚠ ADMIN_API_KEY 未设置，默认使用 PROXY_API_KEY + 'admin'")

    # 加载模型配置
    model_list = []

    if config and Path(config).exists():
        with open(config) as f:
            config_data = json.load(f)
            model_list = config_data.get("model_list", config_data.get("models", []))
        print(f"✓ 加载配置: {config}")
        print(f"  模型数量: {len(model_list)}")
    else:
        # 从环境变量自动检测
        if os.getenv("OPENAI_API_KEY"):
            model_list.append({
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4"},
            })
            print("✓ 检测到 OPENAI_API_KEY，添加 gpt-4")

        if os.getenv("ANTHROPIC_API_KEY"):
            model_list.append({
                "model_name": "claude-3-5-sonnet",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
            })
            print("✓ 检测到 ANTHROPIC_API_KEY，添加 claude-3-5-sonnet")

    # 读取环境变量配置的模型（合并模式）
    target_model = os.getenv("TARGET_MODEL")
    api_url = os.getenv("API_URL")
    access_key = os.getenv("ACCESS_KEY", "EMPTY")
    origin_model = os.getenv("ORIGIN_MODEL")
    api_mode = os.getenv("API_MODE")

    if target_model and origin_model:
        litellm_model = f"{api_mode}/{origin_model}" if api_mode else origin_model
        env_model_entry = {
            "model_name": target_model,
            "litellm_params": {
                "model": litellm_model,
                "api_key": access_key,
            },
        }
        if api_url:
            env_model_entry["litellm_params"]["api_base"] = api_url
        model_list.append(env_model_entry)
        print(f"✓ 环境变量配置模型: {target_model} -> {litellm_model}")

    # 环境变量覆盖 buffer 参数
    env_buffer = os.getenv("BUFFER_SIZE")
    env_trajectory = os.getenv("TRAJECTORY_BUFFER_SIZE")
    env_flush_interval = os.getenv("FLUSH_INTERVAL_SECONDS")
    env_task_timeout = os.getenv("TASK_TIMEOUT_MINUTES")
    if env_buffer:
        buffer_size = int(env_buffer)
    if env_trajectory:
        trajectory_buffer_size = int(env_trajectory)
    if env_flush_interval:
        flush_interval_seconds = int(env_flush_interval)
    if env_task_timeout:
        task_timeout_minutes = int(env_task_timeout)

    # 校验
    if trajectory_buffer_size > 0:
        assert buffer_size > 0, "BUFFER_SIZE must be > 0 when TRAJECTORY_BUFFER_SIZE is set"
        assert trajectory_buffer_size >= buffer_size, (
            f"TRAJECTORY_BUFFER_SIZE ({trajectory_buffer_size}) must be >= BUFFER_SIZE ({buffer_size})"
        )

    if not model_list:
        print("⚠ 没有配置任何模型，服务将以空配置启动")
        print("  可通过以下方式配置模型:")
        print("  1. 提供 --config 参数指定配置文件")
        print("  2. 设置环境变量 (TARGET_MODEL + ORIGIN_MODEL)")
        print("  3. 设置环境变量 (OPENAI_API_KEY, ANTHROPIC_API_KEY 等)")

    # 创建存储和代理
    store = openclaw_tracer.ParquetStore(
        output_dir=output_dir,
        buffer_size=buffer_size,
        time_window_minutes=time_window_minutes,
        trajectory_buffer_size=trajectory_buffer_size,
        flush_interval_seconds=flush_interval_seconds,
    )

    # 保存 store 引用供信号处理使用
    global _store_ref
    _store_ref = store

    proxy = openclaw_tracer.LLMProxy(
        port=port,
        host=host,
        model_list=model_list,
        store=store,
        log_file=log_file,
        proxy_api_key=proxy_api_key,
        task_timeout_minutes=task_timeout_minutes,
        admin_api_key=admin_api_key,
    )

    # 启动
    await proxy.start()

    print("\n" + "=" * 60)
    print("代理服务器已启动")
    print("=" * 60)
    print(f"  URL:        http://infer.pyromind.ai/v1")
    print(f"  V1 API:     http://infer.pyromind.ai/chat/completions")
    print(f"  Models:     {[m['model_name'] for m in model_list]}")
    print(f"  Output:     {output_dir}")
    print(f"  Buffer:     {buffer_size} (每 {buffer_size} 条记录刷新)")
    if trajectory_buffer_size > 0:
        print(f"  Trajectory: {trajectory_buffer_size} (held until /admin/batches/{{id}}/release)")
        print(
            f"  Batch Mode: on ({buffer_size}/batch, "
            f"max {trajectory_buffer_size // buffer_size} slots; no auto-delete)"
        )
    print(f"  Time Window: {time_window_minutes} 分钟")
    print(f"  Flush间隔:  {flush_interval_seconds}s ({flush_interval_seconds // 60}min)")
    print(f"  Auth:       启用 (PROXY_API_KEY)")
    admin_src = (
        "ADMIN_API_KEY"
        if (os.getenv("ADMIN_API_KEY") or "").strip()
        else "PROXY_API_KEY + 'admin'"
    )
    print(f"  Admin:      启用 ({admin_src})")
    if log_file:
        print(f"  HTTP Log:   {log_file}")
    print("=" * 60)
    print("\n按 Ctrl+C 停止服务器\n")

    try:
        # 等待 shutdown_event 或 proxy.wait()
        await asyncio.wait(
            [
                asyncio.create_task(_shutdown_event.wait()),
                asyncio.create_task(proxy.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
    except KeyboardInterrupt:
        print("\n\n检测到键盘中断...")
    finally:
        # 确保 flush 被调用
        print("\n正在停止服务器并保存数据...")
        await proxy.stop()
        await store.flush()
        print("✓ 服务器已停止")
        print(f"✓ 数据已保存到: {output_dir}")
        print(f"\n统计信息:")
        for key, value in store.stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="启动 OpenClaw-Tracer 代理服务器")
    parser.add_argument("--config", "-c", help="配置文件路径 (models.json)")
    parser.add_argument("--output-dir", "-o", default="data", help="数据输出目录")
    parser.add_argument("--port", "-p", type=int, default=None, help="监听端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--buffer-size", type=int, default=1, help="缓冲区大小 (默认: 1)")
    parser.add_argument("--time-window", type=int, default=5, dest="time_window_minutes",
                        help="时间窗口分钟数 (默认: 5)")
    parser.add_argument("--log-file", "-l", help="HTTP访问日志文件路径 (JSONL格式)")
    parser.add_argument("--trajectory-buffer-size", type=int, default=0,
                        help="总保留数据条数，启用分批存储 (默认: 0, 不启用)")
    parser.add_argument("--flush-interval", type=int, default=1800, dest="flush_interval_seconds",
                        help="定时 flush 间隔秒数 (默认: 1800 = 30分钟, 0=禁用)")
    parser.add_argument("--proxy-api-key", help="代理鉴权密钥 (默认从环境变量 PROXY_API_KEY 读取)")
    parser.add_argument("--task-timeout-minutes", type=int, default=10,
                        help="任务超时分钟数 (默认: 10, 可通过环境变量 TASK_TIMEOUT_MINUTES 设置)")

    args = parser.parse_args()

    asyncio.run(main(
        config=args.config,
        output_dir=args.output_dir,
        port=args.port,
        host=args.host,
        buffer_size=args.buffer_size,
        time_window_minutes=args.time_window_minutes,
        log_file=args.log_file,
        trajectory_buffer_size=args.trajectory_buffer_size,
        flush_interval_seconds=args.flush_interval_seconds,
        proxy_api_key=args.proxy_api_key,
        task_timeout_minutes=args.task_timeout_minutes,
    ))
