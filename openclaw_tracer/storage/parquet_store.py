# Copyright (c) 2025 OpenClaw-Tracer
# Parquet file storage backend with time-based sharding

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from openclaw_tracer.storage.base import StorageBackend
from openclaw_tracer.types.core import Span
from openclaw_tracer.workflow_trigger import (
    resolve_workflow_script_path,
    schedule_workflow_trigger,
)

logger = logging.getLogger(__name__)

BATCH_META_FILENAME = "batch.json"


class BatchCapacityFull(Exception):
    """Ready batches fill TRAJECTORY_BUFFER_SIZE; release trained batches first."""


# PyArrow schema for span data with explicit types
_SPAN_SCHEMA = pa.schema([
    ("name", pa.string()),
    ("trace_id", pa.string()),
    ("span_id", pa.string()),
    ("parent_id", pa.string()),
    ("start_time", pa.float64()),
    ("end_time", pa.float64()),
    ("kind", pa.string()),
    ("status", pa.string()),
    ("attributes", pa.string()),
    ("rollout_id", pa.string()),
    ("attempt_id", pa.string()),
    ("sequence_id", pa.int64()),
    ("resource_attributes", pa.string()),
    ("previous_reward", pa.float64()),
    ("final_reward", pa.float64()),
])


def _ensure_final_reward_column(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy shards may omit ``final_reward``; fill with NaN for schema alignment."""
    if "final_reward" not in df.columns:
        df = df.copy()
        df["final_reward"] = float("nan")
    return df


def _coerce_stored_final_reward(value: Any) -> Optional[float]:
    """Parquet float NaN means unset."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    try:
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _get_time_window(interval_minutes: int = 30) -> str:
    """Get the current time window identifier.

    Args:
        interval_minutes: Minutes per time window (default: 30)

    Returns:
        Time window string like "20260303_143000" representing the window start
    """
    now = datetime.now(timezone.utc)
    window_seconds = (now.hour * 3600 + now.minute * 60 + now.second) // (interval_minutes * 60) * (interval_minutes * 60)
    window_hour = window_seconds // 3600
    window_minute = (window_seconds % 3600) // 60
    return now.strftime(f"%Y%m%d_{window_hour:02d}{window_minute:02d}00")


class ParquetStore(StorageBackend):
    """Parquet file storage backend with time-based sharding.

    This backend stores span data in Parquet format:
    - Raw spans: spans/spans_YYYYMMDD_HHMMSS.parquet

    Flushing behavior:
    1. Auto-flush when buffer reaches buffer_size (default: 1 for immediate write)
    2. Creates a new file every time_window_minutes (default: 5)
    3. Manual flush via flush() method

    Args:
        output_dir: Directory to write Parquet files.
        buffer_size: Number of records to buffer before auto-flushing (default: 1).
        auto_flush: Whether to automatically flush when buffer is full.
        time_window_minutes: Minutes per time shard (default: 5).
    """

    # File prefix for span files
    SPAN_PREFIX = "spans_"

    def __init__(
        self,
        output_dir: str | Path = "data",
        buffer_size: int = 1,
        auto_flush: bool = True,
        time_window_minutes: int = 5,
        trajectory_buffer_size: int = 0,
        flush_interval_seconds: int = 1800,
        enable_workflow_trigger: bool | None = None,
        workflow_trigger_script: str | Path | None = None,
    ):
        """Initialize the Parquet store.

        Args:
            output_dir: Directory to write Parquet files.
            buffer_size: Number of records to buffer before auto-flushing (default: 1).
            auto_flush: Whether to automatically flush when buffer is full.
            time_window_minutes: Minutes per time shard (default: 5).
            trajectory_buffer_size: Max records held across ready+collecting batches (0 = disabled).
                Full batches stay on disk until released via admin after training.
            flush_interval_seconds: Periodic flush interval in seconds (default: 1800 = 30min).
                                    0 = disabled.
            enable_workflow_trigger: If set, enables/disables post-flush workflow hook; if None,
                uses ENABLE_WORKFLOW_TRIGGER env (default on unless 0/false/no/off).
            workflow_trigger_script: Absolute path to shell script; None uses WORKFLOW_TRIGGER_SCRIPT
                env or /app/trigger/start_trigger.sh.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        self.time_window_minutes = time_window_minutes
        self.trajectory_buffer_size = trajectory_buffer_size

        # Batch mode: enabled when trajectory_buffer_size > 0
        self._batch_mode = trajectory_buffer_size > 0
        self._current_batch_index: int = 0
        self._current_batch_collected: int = 0

        # Create spans directory (with batch subdirectory if needed)
        spans_dir = self.output_dir / "spans"
        spans_dir.mkdir(exist_ok=True)

        if self._batch_mode:
            self._restore_batch_state(spans_dir)
            self._write_batch_meta(
                self._current_batch_index,
                state=(
                    "ready"
                    if self._current_batch_collected >= self.buffer_size
                    else "collecting"
                ),
                collected=self._current_batch_collected,
            )
        else:
            # Non-batch mode: spans written directly to data/spans/
            pass

        # Current time window for spans
        self._current_span_window = _get_time_window(time_window_minutes)

        # Span buffer
        self._span_buffer: List[Dict[str, Any]] = []

        # Periodic flush
        self.flush_interval_seconds = flush_interval_seconds
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "spans_added": 0,
            "spans_flushed": 0,
            "span_files_created": 0,
        }

        if enable_workflow_trigger is not None:
            self._workflow_trigger_enabled = enable_workflow_trigger
        else:
            ev = os.getenv("ENABLE_WORKFLOW_TRIGGER", "1").strip().lower()
            self._workflow_trigger_enabled = ev not in ("0", "false", "no", "off")

        self._workflow_trigger_script_path = resolve_workflow_script_path(
            workflow_trigger_script
        )

    def _iter_span_parquet_paths(self) -> List[Path]:
        """All span shard files under ``output_dir/spans`` (including batch subdirs)."""
        spans_root = self.output_dir / "spans"
        if not spans_root.exists():
            return []
        paths: List[Path] = []
        if self._batch_mode:
            for sub in sorted(spans_root.iterdir(), key=lambda p: p.name):
                if sub.is_dir() and sub.name.isdigit():
                    paths.extend(sorted(sub.glob(f"{self.SPAN_PREFIX}*.parquet")))
        else:
            paths.extend(sorted(spans_root.glob(f"{self.SPAN_PREFIX}*.parquet")))
        return paths

    def _rewrite_parquet_final_reward_for_rollout(
        self, path: Path, rollout_id: str, value: float
    ) -> bool:
        """Update ``final_reward`` for all rows with ``rollout_id``. Returns True if file changed."""
        try:
            df = pq.read_table(path).to_pandas()
        except Exception as exc:
            logger.warning("Skipping parquet %s: %s", path, exc)
            return False
        df = _ensure_final_reward_column(df)
        if "rollout_id" not in df.columns:
            return False
        mask = df["rollout_id"].astype(str) == str(rollout_id)
        if not mask.any():
            return False
        df.loc[mask, "final_reward"] = value
        table = pa.Table.from_pandas(df, schema=_SPAN_SCHEMA, preserve_index=False)
        pq.write_table(table, path)
        return True

    async def apply_rollout_final_reward(
        self, rollout_id: str, final_reward: Optional[float]
    ) -> None:
        """Set ``final_reward`` on all spans for ``rollout_id`` (buffer + flushed Parquet shards)."""
        value = float("nan") if final_reward is None else float(final_reward)

        buffer_updates = 0
        for row in self._span_buffer:
            if row.get("rollout_id") == rollout_id:
                row["final_reward"] = value
                buffer_updates += 1

        file_updates = 0
        for path in self._iter_span_parquet_paths():
            changed = await asyncio.to_thread(
                self._rewrite_parquet_final_reward_for_rollout, path, rollout_id, value
            )
            if changed:
                file_updates += 1

        logger.info(
            "[ParquetStore] apply_rollout_final_reward rollout_id=%s buffer_rows=%s parquet_files=%s",
            rollout_id,
            buffer_updates,
            file_updates,
        )

    def _restore_batch_state(self, spans_dir: Path) -> None:
        """Restore batch index and collected count from existing directories on restart."""
        existing_batches = sorted(
            (int(d.name) for d in spans_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        )

        if existing_batches:
            self._current_batch_index = existing_batches[-1]
            # Count rows in the current batch directory
            batch_dir = spans_dir / str(self._current_batch_index)
            row_count = 0
            for pf in batch_dir.glob("*.parquet"):
                try:
                    table = pq.read_table(pf)
                    row_count += table.num_rows
                except Exception:
                    continue
            self._current_batch_collected = row_count

            # If current batch is already full, keep it as-is.
            # The advance will happen when the next add_span is called.

            logger.info(
                f"Restored batch state: batch_index={self._current_batch_index}, "
                f"batch_collected={self._current_batch_collected}"
            )
        else:
            self._current_batch_index = 0
            self._current_batch_collected = 0

        # Ensure current batch directory exists
        (spans_dir / str(self._current_batch_index)).mkdir(exist_ok=True)

    async def add_span(self, span: Span) -> None:
        """Add a single span to storage.

        In batch mode a full batch is marked ``ready`` (held for training) and a
        new collecting batch is opened. Ready batches are never auto-deleted;
        call ``release_batch`` after training. If all TRAJECTORY_BUFFER_SIZE
        slots are occupied by unreleased batches, raises ``BatchCapacityFull``.
        """
        # Current batch already full → hold it for training, open a new slot
        if self._batch_mode and self._current_batch_collected >= self.buffer_size:
            self._mark_batch_ready(self._current_batch_index)
            self._advance_batch()

        span_dict = self._span_to_dict(span)
        self._span_buffer.append(span_dict)
        self._stats["spans_added"] += 1

        if self._batch_mode:
            self._current_batch_collected += 1
            self._write_batch_meta(
                self._current_batch_index,
                state=(
                    "ready"
                    if self._current_batch_collected >= self.buffer_size
                    else "collecting"
                ),
                collected=self._current_batch_collected,
            )

        # Check if we need to flush (buffer full, time window changed, or batch ready)
        current_window = _get_time_window(self.time_window_minutes)
        batch_ready = self._batch_mode and self._current_batch_collected >= self.buffer_size
        buffer_full_flush = self.auto_flush and len(self._span_buffer) >= self.buffer_size
        should_flush = (
            buffer_full_flush
            or current_window != self._current_span_window
            or batch_ready
        )
        schedule_workflow = batch_ready or (not self._batch_mode and buffer_full_flush)

        if should_flush:
            await self._flush_spans(
                new_window=current_window != self._current_span_window,
                schedule_workflow=schedule_workflow,
            )
            if batch_ready:
                self._mark_batch_ready(self._current_batch_index)

    async def add_spans(self, spans: List[Span]) -> None:
        """Add multiple spans to storage."""
        for span in spans:
            span_dict = self._span_to_dict(span)
            self._span_buffer.append(span_dict)
            self._stats["spans_added"] += 1

        # Check if we need to flush
        current_window = _get_time_window(self.time_window_minutes)
        buffer_full_flush = self.auto_flush and len(self._span_buffer) >= self.buffer_size
        should_flush = buffer_full_flush or current_window != self._current_span_window

        if should_flush:
            await self._flush_spans(
                new_window=current_window != self._current_span_window,
                schedule_workflow=buffer_full_flush,
            )

    async def query_spans(
        self,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Span]:
        """Query stored spans from all Parquet shard files.

        Args:
            rollout_id: Filter by rollout ID.
            attempt_id: Filter by attempt ID.
            limit: Maximum number of spans to return.

        Returns:
            List of Span objects matching the filters.
        """
        spans_dir = self.output_dir / "spans"
        if not spans_dir.exists():
            return []

        # Read all span shard files
        all_spans = []
        for shard_file in sorted(spans_dir.glob("*.parquet")):
            try:
                df = pq.read_table(shard_file).to_pandas()
                all_spans.append(df)
            except Exception as e:
                # Skip corrupted files
                continue

        if not all_spans:
            return []

        # Combine all shards
        combined_df = pd.concat(all_spans, ignore_index=True)

        # Apply filters
        if rollout_id is not None:
            combined_df = combined_df[combined_df["rollout_id"] == rollout_id]
        if attempt_id is not None:
            combined_df = combined_df[combined_df["attempt_id"] == attempt_id]

        # Limit results
        combined_df = combined_df.head(limit)

        # Convert back to Span objects
        spans = []
        for _, row in combined_df.iterrows():
            spans.append(self._dict_to_span(row.to_dict()))

        return spans

    def start_periodic_flush(self) -> None:
        """Start the background periodic flush task.

        Must be called from within a running event loop (e.g. after server starts).
        Safe to call multiple times — only one task will be created.
        """
        if self.flush_interval_seconds <= 0:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())
        logger.info(f"Periodic flush started (interval={self.flush_interval_seconds}s)")

    async def _periodic_flush_loop(self) -> None:
        """Background loop that flushes the buffer at fixed intervals."""
        while True:
            await asyncio.sleep(self.flush_interval_seconds)
            if self._span_buffer:
                logger.info(f"Periodic flush: {len(self._span_buffer)} spans in buffer")
                await self.flush()

    async def flush(self, schedule_workflow: bool = False) -> None:
        """Flush all buffered data to storage.

        Args:
            schedule_workflow: If True, also fire the buffer-full workflow trigger.
        """
        current_window = _get_time_window(self.time_window_minutes)
        await self._flush_spans(
            new_window=current_window != self._current_span_window,
            schedule_workflow=schedule_workflow,
        )

    async def close(self) -> None:
        """Close the storage backend."""
        # Stop periodic flush task
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()

    async def merge_spans(self, output_path: Optional[str | Path] = None) -> Path:
        """Merge all span shards into a single Parquet file.

        Args:
            output_path: Output file path. If None, uses spans_merged.parquet in output_dir.

        Returns:
            Path to the merged file.
        """
        if output_path is None:
            output_path = self.output_dir / "spans_merged.parquet"
        else:
            output_path = Path(output_path)

        spans_dir = self.output_dir / "spans"
        if not spans_dir.exists():
            return output_path

        # Read and combine all shards
        all_dfs = []
        for shard_file in sorted(spans_dir.glob("*.parquet")):
            try:
                df = pq.read_table(shard_file).to_pandas()
                all_dfs.append(df)
            except Exception:
                continue

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_parquet(output_path, index=False)

        return output_path

    @property
    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        spans_dir = self.output_dir / "spans"
        span_files = list(spans_dir.glob("*.parquet")) if spans_dir.exists() else []

        return {
            **self._stats,
            "output_dir": str(self.output_dir),
            "span_buffer_size": len(self._span_buffer),
            "span_shard_files": len(span_files),
            "current_time_window": _get_time_window(self.time_window_minutes),
            "time_window_minutes": self.time_window_minutes,
        }

    # ========== Private Helper Methods ==========

    async def _flush_spans(self, new_window: bool = False, *, schedule_workflow: bool = False) -> None:
        """Flush buffered spans to Parquet file.

        Args:
            new_window: Whether the time window has changed (creates new file).
            schedule_workflow: If True, spawn workflow trigger after successful write (buffer-full flush).
        """
        if not self._span_buffer:
            return

        # Update time window if changed
        current_window = _get_time_window(self.time_window_minutes)
        if new_window or current_window != self._current_span_window:
            self._current_span_window = current_window

        # Determine target directory
        if self._batch_mode:
            target_dir = self.output_dir / "spans" / str(self._current_batch_index)
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.output_dir / "spans"

        # Write to time-sharded file
        spans_path = target_dir / f"{self.SPAN_PREFIX}{self._current_span_window}.parquet"
        df = pd.DataFrame(self._span_buffer)

        # If file exists, append to it (same time window)
        if spans_path.exists():
            existing_df = pq.read_table(spans_path).to_pandas()
            existing_df = _ensure_final_reward_column(existing_df)
            df = _ensure_final_reward_column(df)
            df = pd.concat([existing_df, df], ignore_index=True)

        df = _ensure_final_reward_column(df)

        # Convert to PyArrow table with explicit schema before writing
        table = pa.Table.from_pandas(df, schema=_SPAN_SCHEMA, preserve_index=False)
        pq.write_table(table, spans_path)
        self._stats["spans_flushed"] += len(self._span_buffer)
        if new_window or not spans_path.exists():
            self._stats["span_files_created"] += 1
        self._span_buffer.clear()

        if not schedule_workflow and not self._workflow_trigger_enabled:
            logger.info(f"not schedule_workflow and not self._workflow_trigger_enabled")
        if schedule_workflow and not self._workflow_trigger_enabled:
            logger.info(f"schedule_workflow and not self._workflow_trigger_enabled")
        if not schedule_workflow and self._workflow_trigger_enabled:
            logger.info(f"not schedule_workflow and self._workflow_trigger_enabled")
        if schedule_workflow and self._workflow_trigger_enabled:
            logger.info(f"schedule_workflow and self._workflow_trigger_enabled")
            logger.info(f"Scheduling workflow trigger: {self._workflow_trigger_script_path}")
            schedule_workflow_trigger(self._workflow_trigger_script_path)

    def prepare_media_images_prefix(self) -> str:
        """Return relative media dir for the batch that the next span will join.

        Advances when the current batch is already full (same rule as ``add_span``),
        so inline images land under the same ``spans/<n>/`` tree as the parquet
        shard. Full batches are held until ``release_batch`` after training.
        """
        if self._batch_mode and self._current_batch_collected >= self.buffer_size:
            self._mark_batch_ready(self._current_batch_index)
            try:
                self._advance_batch()
            except BatchCapacityFull:
                # Stay on the held ready batch; add_span will raise the same error
                return f"spans/{self._current_batch_index}/media/images"
        if self._batch_mode:
            return f"spans/{self._current_batch_index}/media/images"
        return "spans/media/images"

    def _max_batch_slots(self) -> int:
        if self.buffer_size <= 0:
            return 0
        return max(1, self.trajectory_buffer_size // self.buffer_size)

    def _batch_dir(self, batch_index: int) -> Path:
        return self.output_dir / "spans" / str(batch_index)

    def _batch_meta_path(self, batch_index: int) -> Path:
        return self._batch_dir(batch_index) / BATCH_META_FILENAME

    def _list_batch_indices(self) -> List[int]:
        spans_dir = self.output_dir / "spans"
        if not spans_dir.exists():
            return []
        return sorted(
            int(d.name) for d in spans_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )

    def _read_batch_meta(self, batch_index: int) -> Dict[str, Any]:
        path = self._batch_meta_path(batch_index)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.exception("Failed to read batch meta %s", path)
        # Infer from live counters / fullness
        batch_dir = self._batch_dir(batch_index)
        collected = 0
        if batch_dir.is_dir():
            for pf in batch_dir.glob("*.parquet"):
                try:
                    collected += pq.read_table(pf).num_rows
                except Exception:
                    continue
        if batch_index == self._current_batch_index:
            collected = max(collected, self._current_batch_collected)
        state = "ready" if collected >= self.buffer_size else "collecting"
        return {
            "index": batch_index,
            "state": state,
            "collected": collected,
            "buffer_size": self.buffer_size,
        }

    def _write_batch_meta(
        self,
        batch_index: int,
        *,
        state: str,
        collected: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._batch_mode:
            return
        batch_dir = self._batch_dir(batch_index)
        batch_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "index": batch_index,
            "state": state,
            "collected": (
                self._current_batch_collected
                if collected is None and batch_index == self._current_batch_index
                else (collected if collected is not None else 0)
            ),
            "buffer_size": self.buffer_size,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if extra:
            payload.update(extra)
        path = self._batch_meta_path(batch_index)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _mark_batch_ready(self, batch_index: int) -> None:
        meta = self._read_batch_meta(batch_index)
        if meta.get("state") == "ready":
            return
        ready_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._write_batch_meta(
            batch_index,
            state="ready",
            collected=int(meta.get("collected") or self.buffer_size),
            extra={"ready_at": ready_at},
        )
        logger.info("Batch %s marked ready (held for training)", batch_index)

    def _advance_batch(self) -> None:
        """Open the next collecting batch. Never deletes ready batches.

        Raises:
            BatchCapacityFull: if TRAJECTORY_BUFFER_SIZE slots are all occupied
                by unreleased batches.
        """
        occupied = len(self._list_batch_indices())
        # Current full batch already occupies a slot; opening another needs room
        if occupied >= self._max_batch_slots():
            raise BatchCapacityFull(
                f"trajectory buffer full: {occupied}/{self._max_batch_slots()} batch "
                f"slots held for training; POST /admin/batches/{{id}}/release after train"
            )

        self._current_batch_index += 1
        self._current_batch_collected = 0

        new_batch_dir = self._batch_dir(self._current_batch_index)
        new_batch_dir.mkdir(parents=True, exist_ok=True)
        self._write_batch_meta(
            self._current_batch_index,
            state="collecting",
            collected=0,
        )

        self._current_span_window = _get_time_window(self.time_window_minutes)
        logger.info(f"Advanced to batch {self._current_batch_index} (held ready batches retained)")

    def list_batches(self) -> Dict[str, Any]:
        """Return ready/collecting batch inventory for admin APIs."""
        batches = [self._read_batch_meta(i) for i in self._list_batch_indices()]
        ready = [b for b in batches if b.get("state") == "ready"]
        return {
            "batch_mode": self._batch_mode,
            "buffer_size": self.buffer_size,
            "trajectory_buffer_size": self.trajectory_buffer_size,
            "max_batch_slots": self._max_batch_slots() if self._batch_mode else None,
            "current_batch_index": self._current_batch_index,
            "current_batch_collected": self._current_batch_collected,
            "ready_count": len(ready),
            "batches": batches,
        }

    async def release_batch(self, batch_index: int) -> Dict[str, Any]:
        """Delete a ready batch after training completes; fire data-delete callback.

        Raises:
            FileNotFoundError: batch directory missing
            ValueError: batch is still collecting (not ready for release)
        """
        if not self._batch_mode:
            raise ValueError("batch release requires TRAJECTORY_BUFFER_SIZE > 0")

        batch_dir = self._batch_dir(batch_index)
        if not batch_dir.exists():
            raise FileNotFoundError(f"batch {batch_index} not found")

        meta = self._read_batch_meta(batch_index)
        state = str(meta.get("state") or "collecting")
        if state != "ready":
            raise ValueError(
                f"batch {batch_index} is '{state}'; only ready batches can be released "
                f"after training"
            )

        # Flush any buffered rows that still belong to this batch before delete
        if batch_index == self._current_batch_index and self._span_buffer:
            await self._flush_spans(schedule_workflow=False)

        deleted_path = str(batch_dir.resolve())
        shutil.rmtree(batch_dir)
        logger.info("Released (deleted) trained batch %s at %s", batch_index, deleted_path)

        # If we released the current index, reopen it as empty collecting
        if batch_index == self._current_batch_index:
            self._current_batch_collected = 0
            batch_dir.mkdir(parents=True, exist_ok=True)
            self._write_batch_meta(batch_index, state="collecting", collected=0)
            self._current_span_window = _get_time_window(self.time_window_minutes)

        try:
            from openclaw_tracer.data_callback import (
                build_data_delete_event,
                schedule_data_delete_callback,
            )

            schedule_data_delete_callback(
                build_data_delete_event(
                    reason="training_complete",
                    paths=[deleted_path],
                    batch_index=batch_index,
                    extra={
                        "current_batch_index": self._current_batch_index,
                        "buffer_size": self.buffer_size,
                        "trajectory_buffer_size": self.trajectory_buffer_size,
                    },
                )
            )
        except Exception:
            logger.exception(
                "Failed to schedule data delete callback for released batch %s",
                batch_index,
            )

        return {
            "released_batch_index": batch_index,
            "deleted_path": deleted_path,
            "status": self.get_collection_status(),
        }

    def get_collection_status(self) -> Dict[str, Any]:
        """Return collection progress and held-batch summary."""
        if self._batch_mode:
            batches_info = self.list_batches()
            held_records = sum(
                int(b.get("collected") or 0) for b in batches_info["batches"]
            )
            return {
                "collected_data_size": self._current_batch_collected,
                "total_data_size": held_records,
                "batch_mode": True,
                "current_batch_index": self._current_batch_index,
                "ready_count": batches_info["ready_count"],
                "max_batch_slots": batches_info["max_batch_slots"],
                "trajectory_buffer_size": self.trajectory_buffer_size,
            }
        total = self._current_batch_collected
        # non-batch: approximate from stats
        return {
            "collected_data_size": len(self._span_buffer),
            "total_data_size": self._stats.get("spans_flushed", 0) + len(self._span_buffer),
            "batch_mode": False,
        }

    def _span_to_dict(self, span: Span) -> Dict[str, Any]:
        """Convert a Span object to a dictionary for Parquet storage."""
        return {
            "name": span.name,
            "trace_id": span.context.trace_id,
            "span_id": span.context.span_id,
            "parent_id": span.parent_id,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "kind": span.kind.value,
            "status": span.status,
            "attributes": json.dumps(span.attributes),
            "rollout_id": span.rollout_id,
            "attempt_id": span.attempt_id,
            "sequence_id": span.sequence_id,
            "resource_attributes": json.dumps(span.resource.attributes),
            "previous_reward": span.previous_reward,
            "final_reward": float("nan")
            if span.final_reward is None
            else float(span.final_reward),
        }

    def _dict_to_span(self, d: Dict[str, Any]) -> Span:
        """Convert a dictionary back to a Span object."""
        from openclaw_tracer.types.core import (
            Attributes,
            Resource,
            SpanContext,
            SpanKind,
        )

        return Span(
            name=d["name"],
            context=SpanContext(
                trace_id=d["trace_id"],
                span_id=d["span_id"],
            ),
            parent_id=d.get("parent_id"),
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            kind=SpanKind(d.get("kind", "INTERNAL")),
            status=d.get("status", "UNSET"),
            attributes=json.loads(d.get("attributes", "{}")),
            rollout_id=d["rollout_id"],
            attempt_id=d["attempt_id"],
            sequence_id=d.get("sequence_id", 0),
            resource=Resource(attributes=json.loads(d.get("resource_attributes", "{}"))),
            previous_reward=d.get("previous_reward"),
            final_reward=_coerce_stored_final_reward(d.get("final_reward")),
        )
