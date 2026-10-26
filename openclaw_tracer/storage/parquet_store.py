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

        # Current time window for spans. ``_window_generation`` bumps when an admin
        # asks to seal the live shard (so a consumed window can be deleted without
        # swallowing rows written from then on); it resets on a real window change.
        self._current_span_window = _get_time_window(time_window_minutes)
        self._window_generation = 0

        # Span buffer
        self._span_buffer: List[Dict[str, Any]] = []
        # /end_task may arrive before the LLM success callback writes the span.
        # Stash reward/gt keyed by rollout_id and apply when the span lands.
        self._pending_final_reward: Dict[str, float] = {}
        self._pending_gt: Dict[str, str] = {}
        # Serialize parquet rewrites from concurrent /end_task calls.
        self._rollout_rewrite_lock = asyncio.Lock()

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
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)
        return True

    async def apply_rollout_final_reward(
        self, rollout_id: str, final_reward: Optional[float]
    ) -> None:
        """Set ``final_reward`` on all spans for ``rollout_id`` (buffer + flushed Parquet shards)."""
        value = float("nan") if final_reward is None else float(final_reward)

        async with self._rollout_rewrite_lock:
            buffer_updates = 0
            for row in self._span_buffer:
                if row.get("rollout_id") == rollout_id:
                    row["final_reward"] = value
                    buffer_updates += 1

            file_updates = 0
            for path in self._iter_span_parquet_paths():
                changed = await asyncio.to_thread(
                    self._rewrite_parquet_final_reward_for_rollout,
                    path,
                    rollout_id,
                    value,
                )
                if changed:
                    file_updates += 1

            if buffer_updates == 0 and file_updates == 0 and final_reward is not None:
                self._pending_final_reward[str(rollout_id)] = float(final_reward)
                logger.info(
                    "[ParquetStore] pending final_reward rollout_id=%s (span not written yet)",
                    rollout_id,
                )
            elif str(rollout_id) in self._pending_final_reward:
                self._pending_final_reward.pop(str(rollout_id), None)

        logger.info(
            "[ParquetStore] apply_rollout_final_reward rollout_id=%s buffer_rows=%s parquet_files=%s",
            rollout_id,
            buffer_updates,
            file_updates,
        )

    @staticmethod
    def _merge_gt_into_attributes(raw_attrs: Any, gt: str) -> str:
        """Merge ground-truth text into span ``attributes`` JSON under ``gt`` / ``task.gt``."""
        attrs: Dict[str, Any]
        if isinstance(raw_attrs, dict):
            attrs = dict(raw_attrs)
        elif isinstance(raw_attrs, str) and raw_attrs.strip():
            try:
                parsed = json.loads(raw_attrs)
                attrs = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                attrs = {}
        else:
            attrs = {}
        attrs["gt"] = gt
        attrs["task.gt"] = gt
        return json.dumps(attrs, ensure_ascii=False)

    def _rewrite_parquet_gt_for_rollout(
        self, path: Path, rollout_id: str, gt: str
    ) -> bool:
        """Merge ``gt`` into ``attributes`` for all rows with ``rollout_id``."""
        try:
            df = pq.read_table(path).to_pandas()
        except Exception as exc:
            logger.warning("Skipping parquet %s: %s", path, exc)
            return False
        if "rollout_id" not in df.columns or "attributes" not in df.columns:
            return False
        mask = df["rollout_id"].astype(str) == str(rollout_id)
        if not mask.any():
            return False
        df.loc[mask, "attributes"] = df.loc[mask, "attributes"].map(
            lambda raw: self._merge_gt_into_attributes(raw, gt)
        )
        table = pa.Table.from_pandas(df, schema=_SPAN_SCHEMA, preserve_index=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)
        return True

    async def apply_rollout_gt(self, rollout_id: str, gt: str) -> None:
        """Persist ground-truth text onto all spans for ``rollout_id``.

        Reward headers alone are not enough for SFT/GRPO export — clients should
        POST ``gt`` (or ``ground_truth``) on ``/end_task`` so exporters can rebuild
        ``sft_daily`` samples without joining an external source jsonl.
        """
        text = str(gt)
        async with self._rollout_rewrite_lock:
            buffer_updates = 0
            for row in self._span_buffer:
                if row.get("rollout_id") == rollout_id:
                    row["attributes"] = self._merge_gt_into_attributes(
                        row.get("attributes"), text
                    )
                    buffer_updates += 1

            file_updates = 0
            for path in self._iter_span_parquet_paths():
                changed = await asyncio.to_thread(
                    self._rewrite_parquet_gt_for_rollout, path, rollout_id, text
                )
                if changed:
                    file_updates += 1

            if buffer_updates == 0 and file_updates == 0:
                self._pending_gt[str(rollout_id)] = text
                logger.info(
                    "[ParquetStore] pending gt rollout_id=%s chars=%s (span not written yet)",
                    rollout_id,
                    len(text),
                )
            elif str(rollout_id) in self._pending_gt:
                self._pending_gt.pop(str(rollout_id), None)

        logger.info(
            "[ParquetStore] apply_rollout_gt rollout_id=%s chars=%s buffer_rows=%s parquet_files=%s",
            rollout_id,
            len(text),
            buffer_updates,
            file_updates,
        )

    def _apply_pending_to_span_dict(self, span_dict: Dict[str, Any]) -> None:
        """Merge any pending /end_task reward or gt into a just-buffered span row."""
        rid = str(span_dict.get("rollout_id") or "")
        if not rid:
            return
        if rid in self._pending_final_reward:
            span_dict["final_reward"] = self._pending_final_reward.pop(rid)
        if rid in self._pending_gt:
            gt = self._pending_gt.pop(rid)
            span_dict["attributes"] = self._merge_gt_into_attributes(
                span_dict.get("attributes"), gt
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

        In batch mode a full batch is sealed as ``ready``, a new collecting
        directory is opened immediately, then ``WORKFLOW_TRIGGER`` may fire.
        Ready batches are never auto-deleted; call ``release_batch`` after train.

        If all trajectory slots are occupied, the span is **skipped** (logged)
        and this method returns without raising — chat/inference must stay 200.
        """
        try:
            await self._add_span_impl(span)
        except BatchCapacityFull:
            logger.warning(
                "Batch capacity full (%s/%s slots); skip span record "
                "(inference response unaffected)",
                len(self._list_batch_indices()),
                self._max_batch_slots() if self._batch_mode else 0,
            )

    async def _add_span_impl(self, span: Span) -> None:
        # Stuck on a full/ready batch → finish seal (flush→ready→advance→trigger).
        if self._batch_mode and self._current_batch_collected >= self.buffer_size:
            await self.seal_current_batch(schedule_workflow=True)

        span_dict = self._span_to_dict(span)
        self._apply_pending_to_span_dict(span_dict)
        self._span_buffer.append(span_dict)
        self._stats["spans_added"] += 1

        if self._batch_mode:
            self._current_batch_collected += 1
            self._write_batch_meta(
                self._current_batch_index,
                state="collecting",
                collected=self._current_batch_collected,
            )

        current_window = _get_time_window(self.time_window_minutes)
        batch_full = self._batch_mode and self._current_batch_collected >= self.buffer_size
        buffer_full_flush = self.auto_flush and len(self._span_buffer) >= self.buffer_size
        window_changed = current_window != self._current_span_window

        if batch_full:
            # flush → ready → advance → trigger (isolates parquet for training)
            await self.seal_current_batch(schedule_workflow=True)
        elif buffer_full_flush or window_changed:
            await self._flush_spans(
                new_window=window_changed,
                schedule_workflow=(not self._batch_mode and buffer_full_flush),
            )

    async def add_spans(self, spans: List[Span]) -> None:
        """Add multiple spans to storage."""
        for span in spans:
            span_dict = self._span_to_dict(span)
            self._apply_pending_to_span_dict(span_dict)
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

    async def flush(
        self,
        schedule_workflow: bool = False,
        *,
        seal_batch: bool = False,
    ) -> Dict[str, Any]:
        """Flush buffered spans; optionally seal the collecting batch.

        Args:
            schedule_workflow: Fire WORKFLOW_TRIGGER after a successful seal
                (or after a non-seal flush when not sealing).
            seal_batch: Ops path — seal current collecting as ready even when
                ``collected < buffer_size``, then immediately open the next
                collecting directory (same pipeline as auto-full). Agent feeders
                should not call this; they only write spans.
        """
        current_window = _get_time_window(self.time_window_minutes)
        await self._flush_spans(
            new_window=current_window != self._current_span_window,
            schedule_workflow=schedule_workflow and not seal_batch,
        )
        result: Dict[str, Any] = {
            "sealed_batch_index": None,
            "current_batch_index": self._current_batch_index,
            "current_batch_collected": self._current_batch_collected,
            "batch_mode": self._batch_mode,
        }
        if seal_batch and self._batch_mode:
            sealed = await self.seal_current_batch(schedule_workflow=schedule_workflow)
            result["sealed_batch_index"] = sealed
            result["current_batch_index"] = self._current_batch_index
            result["current_batch_collected"] = self._current_batch_collected
        return result

    async def seal_current_batch(self, *, schedule_workflow: bool = False) -> Optional[int]:
        """Seal collecting batch: flush → ready → advance → optional trigger.

        Used by auto-full (BUFFER_SIZE reached) and by admin
        ``flush(seal_batch=true)``. Isolates training parquet under
        ``spans/<id>/`` while new traffic goes to the next collecting dir.

        Returns the sealed batch index, or None if there was nothing to seal.
        Raises ``BatchCapacityFull`` if a new collecting slot cannot be opened.
        """
        if not self._batch_mode:
            raise ValueError("seal_batch requires TRAJECTORY_BUFFER_SIZE > 0")

        if self._span_buffer:
            await self._flush_spans(schedule_workflow=False)

        meta = self._read_batch_meta(self._current_batch_index)
        state = str(meta.get("state") or "collecting")
        collected = int(meta.get("collected") or self._current_batch_collected or 0)
        if state == "ready":
            # Already sealed; must advance so new spans do not append to ready.
            self._advance_batch()
            return (
                self._current_batch_index - 1 if self._current_batch_index > 0 else 0
            )

        if collected <= 0 and not any(
            self._batch_dir(self._current_batch_index).glob("*.parquet")
        ):
            return None

        sealed_index = self._current_batch_index
        self._mark_batch_ready(sealed_index, collected=collected)
        self._advance_batch()

        if schedule_workflow and self._workflow_trigger_enabled:
            try:
                schedule_workflow_trigger(self._workflow_trigger_script_path)
            except Exception:  # noqa: BLE001
                logger.exception("seal_batch workflow trigger failed")

        return sealed_index

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

        # Update time window if changed (a real window change resets the seal)
        current_window = _get_time_window(self.time_window_minutes)
        if new_window or current_window != self._current_span_window:
            self._current_span_window = current_window
            self._window_generation = 0

        # Determine target directory
        if self._batch_mode:
            target_dir = self.output_dir / "spans" / str(self._current_batch_index)
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.output_dir / "spans"

        # Write to time-sharded file
        spans_path = target_dir / self._shard_name()
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

        When the current batch is already full, prefer the next collecting
        index if a slot is free; otherwise keep pointing at the held ready
        batch (``add_span`` will skip recording when capacity is full).
        """
        if self._batch_mode and self._current_batch_collected >= self.buffer_size:
            meta = self._read_batch_meta(self._current_batch_index)
            if str(meta.get("state") or "") != "ready":
                self._mark_batch_ready(self._current_batch_index)
            try:
                self._advance_batch()
            except BatchCapacityFull:
                return f"spans/{self._current_batch_index}/media/images"
        if self._batch_mode:
            return f"spans/{self._current_batch_index}/media/images"
        return "spans/media/images"

    def _max_batch_slots(self) -> int:
        """Max simultaneous batch dirs (ready + collecting + in_training).

        Dual-slot sizing uses ``TRAJECTORY_BUFFER_SIZE ≈ 3× agent BS`` while
        ``BUFFER_SIZE > BS`` so auto-ready does not race agent flush. Ceiling
        division keeps ~3 slots when trajectory is 3×BS but BUFFER_SIZE is larger
        (e.g. BS=3500, BUFFER=5000, trajectory=10500 → 3 slots, not 2).
        """
        if self.buffer_size <= 0 or self.trajectory_buffer_size <= 0:
            return 0
        return max(
            1,
            (self.trajectory_buffer_size + self.buffer_size - 1) // self.buffer_size,
        )

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

    def _mark_batch_ready(
        self, batch_index: int, *, collected: Optional[int] = None
    ) -> None:
        meta = self._read_batch_meta(batch_index)
        if meta.get("state") in {"ready", "in_training"}:
            return
        ready_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if collected is None:
            collected = int(
                meta.get("collected")
                or (
                    self._current_batch_collected
                    if batch_index == self._current_batch_index
                    else self.buffer_size
                )
            )
        self._write_batch_meta(
            batch_index,
            state="ready",
            collected=collected,
            extra={"ready_at": ready_at},
        )
        logger.info(
            "Batch %s marked ready (held for training, collected=%s)",
            batch_index,
            collected,
        )

    def claim_batch(self, batch_index: int) -> Dict[str, Any]:
        """Mark a ready batch as ``in_training`` so only one orch round claims it."""
        if not self._batch_mode:
            raise ValueError("claim requires TRAJECTORY_BUFFER_SIZE > 0")
        batch_dir = self._batch_dir(batch_index)
        if not batch_dir.exists():
            raise FileNotFoundError(f"batch {batch_index} not found")
        meta = self._read_batch_meta(batch_index)
        state = str(meta.get("state") or "collecting")
        if state == "in_training":
            return {**meta, "claimed": False, "note": "already_in_training"}
        if state != "ready":
            raise ValueError(
                f"batch {batch_index} is '{state}'; only ready batches can be claimed"
            )
        claimed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._write_batch_meta(
            batch_index,
            state="in_training",
            collected=int(meta.get("collected") or 0),
            extra={
                "ready_at": meta.get("ready_at"),
                "claimed_at": claimed_at,
            },
        )
        return {**self._read_batch_meta(batch_index), "claimed": True}

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
        """Return ready/collecting/in_training batch inventory for admin APIs."""
        batches = [self._read_batch_meta(i) for i in self._list_batch_indices()]
        ready = [b for b in batches if b.get("state") == "ready"]
        in_training = [b for b in batches if b.get("state") == "in_training"]
        collecting = [b for b in batches if b.get("state") == "collecting"]
        return {
            "batch_mode": self._batch_mode,
            "buffer_size": self.buffer_size,
            "trajectory_buffer_size": self.trajectory_buffer_size,
            "max_batch_slots": self._max_batch_slots() if self._batch_mode else None,
            "current_batch_index": self._current_batch_index,
            "current_batch_collected": self._current_batch_collected,
            "ready_count": len(ready),
            "in_training_count": len(in_training),
            "collecting_count": len(collecting),
            #: Agent backpressure: stop feeding when this exceeds the pipeline depth.
            "inflight_count": len(ready) + len(in_training),
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
        if state not in {"ready", "in_training"}:
            raise ValueError(
                f"batch {batch_index} is '{state}'; only ready/in_training batches "
                f"can be released after training"
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

    def _media_images_root(self) -> Path:
        """``spans/media/images/<trace_id>/`` holds inline images per trace."""
        return self.output_dir / "spans" / "media" / "images"

    def _shard_name(self) -> str:
        """File name of the live shard; ``_g<n>`` marks shards opened by a seal."""
        suffix = f"_g{self._window_generation}" if self._window_generation else ""
        return f"{self.SPAN_PREFIX}{self._current_span_window}{suffix}.parquet"

    def _current_window_path(self) -> Path:
        """The shard the tracer may still be appending to (non-batch mode)."""
        if self._batch_mode:
            return self.output_dir / "spans" / str(self._current_batch_index) / self._shard_name()
        return self.output_dir / "spans" / self._shard_name()

    def _seal_current_shard(self) -> str:
        """Open a fresh shard so the live one can be deleted without losing writes."""
        self._window_generation += 1
        sealed = self._current_window_path()
        logger.info("Sealed span shard; next writes go to %s", sealed.name)
        return sealed.name

    def _trace_ids_in(self, path: Path) -> set:
        """Trace ids of one shard; empty when the shard has no such column."""
        try:
            table = pq.read_table(path, columns=["trace_id"])
        except Exception:  # noqa: BLE001 - missing column or corrupt shard
            return set()
        return {str(v) for v in table.column("trace_id").to_pylist() if v}

    def _resolve_span_file(self, name: str) -> Path:
        """Map a caller-supplied shard to a path inside ``spans/``.

        The orchestrator reports host paths (``datas/openclaw-val-eval/spans/x``)
        while the tracer sees the same tree mounted elsewhere (``/app/data``), so
        an absolute or relative join cannot match; fall back to the
        relative-to-``spans/`` suffix, then to a unique basename. Unknown names
        resolve to a non-existent path inside ``spans/`` and are reported as
        ``missing`` — release must stay idempotent across retries.
        """
        spans_root = (self.output_dir / "spans").resolve()

        def _inside(path: Path) -> Path:
            resolved = path.resolve()
            try:
                resolved.relative_to(spans_root)
            except ValueError:
                raise ValueError(
                    f"refusing to release {name}: outside {spans_root}"
                ) from None
            return resolved

        raw = Path(str(name))
        if raw.is_absolute() and raw.exists():
            return _inside(raw)
        for cand in (self.output_dir / raw, self.output_dir / "spans" / raw):
            if cand.exists():
                return _inside(cand)

        shards = self._iter_span_parquet_paths()
        normalized = "/" + str(raw).replace("\\", "/").lstrip("./")
        rel = normalized.split("/spans/", 1)[1] if "/spans/" in normalized else None
        if rel:
            for shard in shards:
                # Prefer the shard whose path relative to spans/ matches exactly,
                # so ``<batch>/spans_x.parquet`` never collapses onto a root shard.
                if str(shard.resolve().relative_to(spans_root)).replace("\\", "/") == rel:
                    return _inside(shard)
        by_name = [s for s in shards if s.name == raw.name]
        if len(by_name) == 1:
            return _inside(by_name[0])
        # Unknown shard: validate anyway so ``..`` in the name cannot escape.
        return _inside(spans_root / (rel or raw.name))

    async def release_spans(
        self,
        files: Optional[List[str]] = None,
        *,
        keep_current: bool = True,
        reason: str = "training_complete",
    ) -> Dict[str, Any]:
        """Delete consumed span shards (and their images) so they cannot be reread.

        This is the consumption watermark: the orchestrator names the shards its
        export actually read, and they are removed from the store instead of
        being left for the next round to pick up again. Works with or without
        batch mode; batch mode requires explicit ``files`` so a batch that is
        still collecting is never wiped.

        ``keep_current`` defaults to true and merely spares the live shard. Pass
        false when the export consumed it too: the shard is sealed first (writes
        move to a fresh ``_g<n>`` file) and then deleted, so no row written from
        that moment on is lost.

        Media is pruned only for trace ids that no surviving shard still needs,
        so images of a newer, not-yet-consumed trace are never lost.

        Raises:
            ValueError: path outside ``spans/``, or batch mode without ``files``.
        """
        if self._batch_mode and not files:
            raise ValueError(
                "batch mode requires explicit files; "
                "use POST /admin/batches/{id}/release instead"
            )

        if files:
            candidates = [self._resolve_span_file(f) for f in files]
        else:
            candidates = [p.resolve() for p in self._iter_span_parquet_paths()]

        deleted_files: List[str] = []
        skipped: List[Dict[str, Any]] = []
        deleted_rows = 0
        deleted_traces: set = set()
        sealed: Optional[str] = None

        for path in candidates:
            if path == self._current_window_path().resolve():
                if keep_current:
                    # The tracer is still appending here; deleting it would drop
                    # the rows written between the export and now.
                    skipped.append({"path": str(path), "reason": "current_window"})
                    continue
                if path.is_file():
                    sealed = self._seal_current_shard()
            if not path.is_file():
                skipped.append({"path": str(path), "reason": "missing"})
                continue
            try:
                deleted_rows += pq.read_table(path).num_rows
            except Exception:  # noqa: BLE001
                logger.warning("release_spans: cannot count rows in %s", path)
            deleted_traces |= self._trace_ids_in(path)
            path.unlink()
            deleted_files.append(str(path))
            logger.info("Released (deleted) consumed span shard %s", path)

        media_root = self._media_images_root()
        remaining_traces: set = set()
        if deleted_traces and media_root.is_dir():
            for path in self._iter_span_parquet_paths():
                remaining_traces |= self._trace_ids_in(path)
        deleted_media: List[str] = []
        if deleted_traces and media_root.is_dir():
            for folder in sorted(media_root.iterdir()):
                if not folder.is_dir():
                    continue
                if folder.name not in deleted_traces or folder.name in remaining_traces:
                    continue
                shutil.rmtree(folder, ignore_errors=True)
                deleted_media.append(str(folder))
            if deleted_media:
                logger.info("Released (deleted) %d consumed media dirs", len(deleted_media))

        if deleted_files or deleted_media:
            try:
                from openclaw_tracer.data_callback import (
                    build_data_delete_event,
                    schedule_data_delete_callback,
                )

                schedule_data_delete_callback(
                    build_data_delete_event(
                        reason=reason,
                        paths=deleted_files + deleted_media,
                        extra={
                            "deleted_rows": deleted_rows,
                            "deleted_traces": len(deleted_traces),
                            "batch_mode": self._batch_mode,
                        },
                    )
                )
            except Exception:
                logger.exception("Failed to schedule data delete callback for released spans")

        return {
            "deleted_files": deleted_files,
            "deleted_file_count": len(deleted_files),
            "deleted_rows": deleted_rows,
            "deleted_media_dirs": deleted_media,
            "deleted_media_count": len(deleted_media),
            "skipped": skipped,
            "sealed_shard": sealed,
            "kept_files": [str(p) for p in self._iter_span_parquet_paths()],
            "batch_mode": self._batch_mode,
        }

    def get_collection_status(self) -> Dict[str, Any]:
        """Return collection progress for public ``GET /status``.

        Batch mode exposes ``capacity`` (trajectory budget) and ``held``
        (records already pressed into batch dirs) so producers can pace
        feeding without admin APIs. Chat remains available when held is full;
        only recording may skip.
        """
        if self._batch_mode:
            batches_info = self.list_batches()
            held_records = sum(
                int(b.get("collected") or 0) for b in batches_info["batches"]
            )
            return {
                "capacity": self.trajectory_buffer_size,
                "held": held_records,
                "buffer_size": self.buffer_size,
                "collected_data_size": self._current_batch_collected,
                "total_data_size": held_records,
                "batch_mode": True,
                "current_batch_index": self._current_batch_index,
                "ready_count": batches_info["ready_count"],
                "in_training_count": batches_info["in_training_count"],
                "collecting_count": batches_info["collecting_count"],
                "inflight_count": batches_info["inflight_count"],
                "max_batch_slots": batches_info["max_batch_slots"],
                "trajectory_buffer_size": self.trajectory_buffer_size,
            }
        held = self._stats.get("spans_flushed", 0) + len(self._span_buffer)
        return {
            "capacity": None,
            "held": held,
            "buffer_size": self.buffer_size,
            "collected_data_size": len(self._span_buffer),
            "total_data_size": held,
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
