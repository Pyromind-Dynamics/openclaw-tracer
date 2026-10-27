# Copyright (c) 2025 OpenClaw-Tracer
# Runtime /admin/config GET+PATCH for buffer / trajectory / workflow script.

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from openclaw_tracer.admin_ops import register_admin_ops_routes
from openclaw_tracer.storage.parquet_store import ParquetStore


def _client(store: ParquetStore, admin_key: str = "admin-secret") -> TestClient:
    app = FastAPI()
    proxy = MagicMock()
    proxy.admin_api_key = admin_key
    proxy.store = store
    register_admin_ops_routes(app, proxy)
    return TestClient(app)


def test_get_and_patch_config(tmp_path: Path) -> None:
    script = tmp_path / "trigger.sh"
    script.write_text("#!/bin/bash\necho ok\n")
    script.chmod(0o755)

    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=100,
        trajectory_buffer_size=300,
        time_window_minutes=9999,
        enable_workflow_trigger=True,
        workflow_trigger_script=script,
    )
    client = _client(store)
    headers = {"Authorization": "Bearer admin-secret"}

    get_resp = client.get("/admin/config", headers=headers)
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["ok"] is True
    assert data["buffer_size"] == 100
    assert data["trajectory_buffer_size"] == 300
    assert data["workflow_trigger_script"] == str(script)

    new_script = tmp_path / "other.sh"
    new_script.write_text("#!/bin/bash\necho other\n")
    new_script.chmod(0o755)

    patch_resp = client.patch(
        "/admin/config",
        headers=headers,
        json={
            "buffer_size": 50,
            "trajectory_buffer_size": 150,
            "workflow_trigger_script": str(new_script),
        },
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["buffer_size"] == 50
    assert data["trajectory_buffer_size"] == 150
    assert data["workflow_trigger_script"] == str(new_script)
    assert store.buffer_size == 50
    assert store.trajectory_buffer_size == 150
    assert str(store._workflow_trigger_script_path) == str(new_script)
    assert os.environ.get("BUFFER_SIZE") == "50"
    assert os.environ.get("TRAJECTORY_BUFFER_SIZE") == "150"
    assert os.environ.get("WORKFLOW_TRIGGER_SCRIPT") == str(new_script)


def test_patch_config_env_aliases(tmp_path: Path) -> None:
    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=100,
        trajectory_buffer_size=300,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )
    client = _client(store)
    headers = {"Authorization": "Bearer admin-secret"}
    resp = client.patch(
        "/admin/config",
        headers=headers,
        json={"BUFFER_SIZE": 80, "TRAJECTORY_BUFFER_SIZE": 240},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["buffer_size"] == 80
    assert data["trajectory_buffer_size"] == 240


def test_patch_config_rejects_traj_lt_buffer(tmp_path: Path) -> None:
    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=100,
        trajectory_buffer_size=300,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )
    client = _client(store)
    headers = {"Authorization": "Bearer admin-secret"}
    resp = client.patch(
        "/admin/config",
        headers=headers,
        json={"trajectory_buffer_size": 10},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_config"
    assert store.trajectory_buffer_size == 300


def test_update_runtime_config_direct(tmp_path: Path) -> None:
    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=10,
        trajectory_buffer_size=30,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )
    cfg = store.update_runtime_config(buffer_size=20, trajectory_buffer_size=60)
    assert cfg["buffer_size"] == 20
    assert cfg["trajectory_buffer_size"] == 60
    assert cfg["max_batch_slots"] == 3


@pytest.mark.asyncio
async def test_cannot_disable_batch_while_inflight(tmp_path: Path) -> None:
    from openclaw_tracer.types.core import (
        Resource,
        Span,
        SpanContext,
        SpanKind,
    )

    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=2,
        trajectory_buffer_size=4,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )

    for i in range(2):
        await store.add_span(
            Span(
                name="llm.completion",
                context=SpanContext(trace_id=f"{i:032x}", span_id=f"{i:016x}"),
                start_time=float(i),
                end_time=float(i) + 1,
                kind=SpanKind.CLIENT,
                status="OK",
                attributes={},
                resource=Resource(attributes={}),
                sequence_id=i,
            )
        )

    with pytest.raises(ValueError, match="inflight"):
        store.update_runtime_config(trajectory_buffer_size=0)
    await store.close()
