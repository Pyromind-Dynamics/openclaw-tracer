# ============================================
# Stage 1: Builder - install Python deps into venv
# ============================================
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# Build deps for wheels that may not have manylinux binaries (pyarrow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .



RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements-prod.txt \
    && /venv/bin/pip install --no-cache-dir "prisma>=0.11.0" \
    && /venv/bin/pip install --no-cache-dir "pyromind-sdk==0.1.11" \
    && /venv/bin/python -c "import prisma; print('prisma-ok')"
    
# ============================================
# Stage 2: Runtime - slim image, no compiler
# ============================================
FROM python:3.10-slim-bookworm AS final

# 设置工作目录
WORKDIR /app

# curl: healthcheck; procps: ps; libjpeg/zlib: Pillow image externalization
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /venv /venv

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=43886 \
    BUFFER_SIZE=8000 \
    TRAJECTORY_BUFFER_SIZE=24000 \
    TIME_WINDOW_MINUTES=30 \
    ENABLE_WORKFLOW_TRIGGER=1 \
    WORKFLOW_TRIGGER_SCRIPT=/app/scripts/trigger_orch_round.sh

# 代理鉴权密钥 (必填，运行时通过 docker run -e 或 docker-compose 设置)
# ENV PROXY_API_KEY=your-api-key-here

COPY openclaw_tracer/ /app/openclaw_tracer/
COPY scripts/ /app/scripts/

# 创建必要的目录
RUN mkdir -p /app/config /app/data /app/logs /app/trigger

# 暴露端口
EXPOSE 43886

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:43886/v1/models || exit 1

# 设置 docker-entrypoint.sh 可执行权限
RUN chmod +x /app/scripts/docker-entrypoint.sh

# 启动命令（使用 docker-entrypoint.sh）
CMD ["/app/scripts/docker-entrypoint.sh"]
