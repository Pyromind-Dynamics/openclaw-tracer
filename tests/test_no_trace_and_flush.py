# Copyright (c) 2025 OpenClaw-Tracer
# Tests for X-Task-No-Trace parsing, flush API helper, and data-delete callbacks.

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from openclaw_tracer.data_callback import (
    build_data_delete_event,
    schedule_data_delete_callback,
)
from openclaw_tracer.middleware.task_middleware import TaskMiddleware
from openclaw_tracer.storage.parquet_store import ParquetStore
from openclaw_tracer.types.core import (
    Resource,
    Span,
    SpanContext,
    SpanKind,
)


def test_extract_no_trace_truthy() -> None:
    mw = TaskMiddleware(app=MagicMock(), task_manager=MagicMock())
    assert mw._extract_bool_header({"X-Task-No-Trace": "true"}, "x-task-no-trace") is True
    assert mw._extract_bool_header({"x-task-no-trace": "1"}, "x-task-no-trace") is True
    assert mw._extract_bool_header({"x-task-no-trace": "yes"}, "x-task-no-trace") is True
    assert mw._extract_bool_header({"x-task-no-trace": "0"}, "x-task-no-trace") is False
    assert mw._extract_bool_header({}, "x-task-no-trace") is False


@pytest.mark.asyncio
async def test_flush_accepts_schedule_workflow(tmp_path: Path) -> None:
    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=100,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )
    span = Span(
        name="llm.completion",
        context=SpanContext(trace_id="a" * 32, span_id="b" * 16),
        start_time=1.0,
        end_time=2.0,
        kind=SpanKind.CLIENT,
        status="OK",
        attributes={},
        resource=Resource(attributes={}),
    )
    await store.add_span(span)
    assert len(store._span_buffer) == 1
    await store.flush(schedule_workflow=False)
    assert len(store._span_buffer) == 0
    await store.close()


@pytest.mark.asyncio
async def test_batch_delete_schedules_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DELETE_CALLBACK_URL", "http://callback.test/del")
    monkeypatch.delenv("DATA_DELETE_CALLBACK_SCRIPT", raising=False)

    scheduled: List[Any] = []

    def capture(payload: Any) -> None:
        scheduled.append(payload)

    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=2,
        trajectory_buffer_size=4,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )

    def make_span(i: int) -> Span:
        return Span(
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

    with patch(
        "openclaw_tracer.data_callback.schedule_data_delete_callback", capture
    ):
        # Fill batches until an old one is deleted:
        # max_batches = 4//2 = 2; delete when current_batch_index - 2 - 1 >= 0
        # advance at 2,4,6 → at advance to batch 3, delete batch 0
        for i in range(8):
            await store.add_span(make_span(i))

    assert any(p.get("event") == "data.deleted" for p in scheduled), scheduled
    assert any(p.get("reason") == "batch_advance" for p in scheduled)
    await store.close()


def test_prepare_media_images_prefix_tracks_batch(tmp_path: Path) -> None:
    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=2,
        trajectory_buffer_size=4,
        time_window_minutes=9999,
        enable_workflow_trigger=False,
    )
    assert store.prepare_media_images_prefix() == "spans/0/media/images"
    store._current_batch_collected = 2
    assert store.prepare_media_images_prefix() == "spans/1/media/images"
    assert store._current_batch_index == 1


@pytest.mark.asyncio
async def test_schedule_data_delete_skipped_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATA_DELETE_CALLBACK_URL", raising=False)
    monkeypatch.delenv("DATA_DELETE_CALLBACK_SCRIPT", raising=False)
    with patch("openclaw_tracer.data_callback.asyncio.get_running_loop") as loop:
        schedule_data_delete_callback(
            build_data_delete_event(reason="batch_advance", paths=["/tmp/x"], batch_index=0)
        )
        loop.assert_not_called()
