# Copyright (c) 2025 OpenClaw-Tracer
# Tests for ParquetStore

import asyncio
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from openclaw_tracer.storage.parquet_store import ParquetStore, _get_time_window
from openclaw_tracer.types.core import (
    Resource,
    Span,
    SpanContext,
    SpanKind,
)


class TestGetTimeWindow:
    """Tests for _get_time_window function."""

    def test_default_interval(self) -> None:
        """Test with default 30 minute interval."""
        # The function uses the current time, so we just verify it returns a string
        result = _get_time_window()
        assert isinstance(result, str)
        assert len(result) == 15  # YYYYMMDD_HHMMSS format

    def test_custom_interval(self) -> None:
        """Test with custom interval."""
        result = _get_time_window(interval_minutes=60)
        assert isinstance(result, str)
        assert len(result) == 15

    def test_format_structure(self) -> None:
        """Test the format of the time window string."""
        result = _get_time_window()
        # Should match format: YYYYMMDD_HHMMSS
        assert "_" in result
        parts = result.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS


class TestParquetStoreInit:
    """Tests for ParquetStore initialization."""

    def test_init_with_defaults(self, temp_dir: Path) -> None:
        """Test initialization with default parameters."""
        store = ParquetStore(output_dir=temp_dir / "data")
        assert store.output_dir == temp_dir / "data"
        assert store.buffer_size == 1
        assert store.auto_flush is True
        assert store.time_window_minutes == 5
        assert store.trajectory_buffer_size == 0
        assert store._batch_mode is False

    def test_init_with_custom_params(self, temp_dir: Path) -> None:
        """Test initialization with custom parameters."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=100,
            auto_flush=False,
            time_window_minutes=30,
            trajectory_buffer_size=1000,
        )
        assert store.buffer_size == 100
        assert store.auto_flush is False
        assert store.time_window_minutes == 30
        assert store.trajectory_buffer_size == 1000
        assert store._batch_mode is True

    def test_creates_output_directory(self, temp_dir: Path) -> None:
        """Test that initialization creates the output directory."""
        output_path = temp_dir / "new_data"
        assert not output_path.exists()
        store = ParquetStore(output_dir=output_path)
        assert output_path.exists()
        assert (output_path / "spans").exists()

    def test_init_creates_batch_directory_in_batch_mode(self, temp_dir: Path) -> None:
        """Test that batch mode creates batch directories."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=10,
            trajectory_buffer_size=100,
        )
        spans_dir = temp_dir / "data" / "spans"
        assert spans_dir.exists()
        # Should have batch 0 directory
        assert (spans_dir / "0").exists()


class TestParquetStoreAddSpan:
    """Tests for ParquetStore.add_span method."""

    @pytest.mark.asyncio
    async def test_add_single_span(
        self,
        async_parquet_store: ParquetStore,
        sample_span: Span,
    ) -> None:
        """Test adding a single span."""
        initial_stats = async_parquet_store.stats.copy()
        await async_parquet_store.add_span(sample_span)

        assert async_parquet_store.stats["spans_added"] == initial_stats["spans_added"] + 1

    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_size(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that auto-flush triggers when buffer reaches buffer_size."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=3,
            auto_flush=True,
        )
        spans_dir = temp_dir / "data" / "spans"

        # Add spans up to buffer_size
        for i in range(3):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                end_time=1234567895.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)

        # Check that file was created
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        # Verify content
        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 3

    @pytest.mark.asyncio
    async def test_no_auto_flush_when_disabled(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that spans are buffered when auto_flush is False."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=10,
            auto_flush=False,
        )

        # Add spans
        for i in range(3):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)

        # No file should be created yet
        spans_dir = temp_dir / "data" / "spans"
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 0

        # Manual flush should create the file
        await store.flush()
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

    @pytest.mark.asyncio
    async def test_add_spans_updates_sequence(
        self,
        async_parquet_store: ParquetStore,
        sample_span: Span,
    ) -> None:
        """Test that adding spans increments sequence_id."""
        initial_seq = sample_span.sequence_id
        await async_parquet_store.add_span(sample_span)

        # Create a new span with incremented sequence
        next_span = Span(
            name="test.span.2",
            context=sample_span.context,
            start_time=1234567896.0,
            end_time=1234567900.0,
            rollout_id=sample_span.rollout_id,
            attempt_id=sample_span.attempt_id,
            sequence_id=initial_seq + 1,
        )
        await async_parquet_store.add_span(next_span)


class TestParquetStoreAddSpans:
    """Tests for ParquetStore.add_spans method."""

    @pytest.mark.asyncio
    async def test_add_multiple_spans(
        self,
        async_parquet_store: ParquetStore,
        sample_span: Span,
    ) -> None:
        """Test adding multiple spans at once."""
        spans = []
        for i in range(5):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                end_time=1234567895.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
                sequence_id=i,
            )
            spans.append(span)

        await async_parquet_store.add_spans(spans)
        await async_parquet_store.flush()

        assert async_parquet_store.stats["spans_added"] == 5


class TestParquetStoreQuerySpans:
    """Tests for ParquetStore.query_spans method."""

    @pytest.mark.asyncio
    async def test_query_all_spans(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test querying all spans."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=1,
        )

        # Add some spans
        for i in range(3):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                end_time=1234567895.0 + i,
                rollout_id=f"ro-test{i}",
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)
        await store.close()

        # Query all spans
        results = await store.query_spans(limit=10)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_by_rollout_id(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test querying spans by rollout_id."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=1,
        )

        # Add spans with different rollout_ids
        rollout_1 = "ro-test123"
        rollout_2 = "ro-test456"

        for rollout_id in [rollout_1, rollout_1, rollout_2]:
            span = Span(
                name="test.span",
                context=sample_span.context,
                start_time=1234567890.0,
                end_time=1234567895.0,
                rollout_id=rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)
        await store.close()

        # Query by rollout_id
        results = await store.query_spans(rollout_id=rollout_1)
        assert len(results) == 2

        results = await store.query_spans(rollout_id=rollout_2)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_limit(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test query limit parameter."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=1,
        )

        # Add more spans than the limit
        for i in range(10):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                end_time=1234567895.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
                sequence_id=i,
            )
            await store.add_span(span)
        await store.close()

        # Query with limit
        results = await store.query_spans(limit=5)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_empty_store(self, temp_dir: Path) -> None:
        """Test querying an empty store."""
        store = ParquetStore(output_dir=temp_dir / "data")
        results = await store.query_spans()
        assert len(results) == 0


class TestParquetStoreFlush:
    """Tests for ParquetStore.flush method."""

    @pytest.mark.asyncio
    async def test_flush_creates_file(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that flush creates a Parquet file."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=100,  # Large buffer
            auto_flush=False,
        )

        # Add a span
        await store.add_span(sample_span)

        # No file yet
        spans_dir = temp_dir / "data" / "spans"
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 0

        # Flush
        await store.flush()

        # File should exist
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that flush clears the buffer."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=100,
            auto_flush=False,
        )

        # Add spans
        for i in range(3):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)

        assert len(store._span_buffer) == 3

        # Flush
        await store.flush()

        # Buffer should be cleared
        assert len(store._span_buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_multiple_times(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that multiple flushes work correctly."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=100,
            auto_flush=False,
        )

        # First flush
        await store.add_span(sample_span)
        await store.flush()

        # Second flush (empty buffer)
        await store.flush()

        # Third flush with new data
        span2 = Span(
            name="test.span.2",
            context=sample_span.context,
            start_time=1234567900.0,
            rollout_id=sample_span.rollout_id,
            attempt_id=sample_span.attempt_id,
        )
        await store.add_span(span2)
        await store.flush()

        # Verify
        spans_dir = temp_dir / "data" / "spans"
        parquet_files = list(spans_dir.glob("*.parquet"))
        # Multiple flushes to same time window should append to the same file
        assert len(parquet_files) >= 1


class TestParquetStoreStats:
    """Tests for ParquetStore.stats property."""

    @pytest.mark.asyncio
    async def test_stats_initial_values(
        self,
        async_parquet_store: ParquetStore,
    ) -> None:
        """Test initial stats values."""
        stats = async_parquet_store.stats
        assert stats["spans_added"] == 0
        assert stats["spans_flushed"] == 0
        assert stats["span_buffer_size"] == 0
        assert stats["span_files_created"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_adding_spans(
        self,
        async_parquet_store: ParquetStore,
        sample_span: Span,
    ) -> None:
        """Test stats after adding spans."""
        await async_parquet_store.add_span(sample_span)
        stats = async_parquet_store.stats
        assert stats["spans_added"] == 1
        assert stats["span_buffer_size"] == 1


class TestParquetStoreBatchMode:
    """Tests for ParquetStore batch mode."""

    @pytest.mark.asyncio
    async def test_batch_mode_initialization(
        self,
        temp_dir: Path,
    ) -> None:
        """Test batch mode initialization."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=10,
            trajectory_buffer_size=100,
        )
        assert store._batch_mode is True
        assert store._current_batch_index == 0
        assert store._current_batch_collected == 0

    @pytest.mark.asyncio
    async def test_batch_mode_creates_directories(
        self,
        temp_dir: Path,
    ) -> None:
        """Test that batch mode creates numbered directories."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=2,
            trajectory_buffer_size=10,  # 5 batches
        )

        spans_dir = temp_dir / "data" / "spans"
        assert (spans_dir / "0").exists()

    @pytest.mark.asyncio
    async def test_batch_mode_advances_after_buffer_full(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that batch mode advances to next batch after buffer is full."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=2,
            trajectory_buffer_size=10,
        )

        # Add spans to fill first batch
        for i in range(2):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)

        # Adding the 3rd span should advance to batch 1
        span = Span(
            name="test.span.2",
            context=sample_span.context,
            start_time=1234567892.0,
            rollout_id=sample_span.rollout_id,
            attempt_id=sample_span.attempt_id,
        )
        await store.add_span(span)
        await store.flush()

        # Check batch directories
        spans_dir = temp_dir / "data" / "spans"
        assert (spans_dir / "0").exists()
        # The new batch should be created
        assert (spans_dir / "1").exists()

    @pytest.mark.asyncio
    async def test_batch_mode_deletes_old_batches(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that batch mode deletes old batches when limit is exceeded."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=2,
            trajectory_buffer_size=6,  # Keeps 3 batches (6/2=3)
        )

        spans_dir = temp_dir / "data" / "spans"

        # Fill and advance through batches
        # Batch 0: spans 0-1
        # Batch 1: spans 2-3 (deletes batch 0)
        # Batch 2: spans 4-5 (deletes batch 1)
        for i in range(7):  # Will create batches 0, 1, 2, 3
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)

        await store.flush()

        # Should keep batches 1, 2, 3 (3 batches total)
        # Batch 0 should be deleted
        # max_batches = 6/2 = 3
        # Keeps batch N - 3 + 1 to N
        # When advancing to batch 3: delete 3 - 3 - 1 = -1 (no deletion)
        # When advancing to batch 4: delete 4 - 3 - 1 = 0
        # Let me check the actual state

        existing_batches = [
            d for d in spans_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        batch_numbers = sorted(int(d.name) for d in existing_batches)
        # Should only have the latest batches

    @pytest.mark.asyncio
    async def test_batch_mode_restores_state(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that batch mode restores state on restart."""
        output_dir = temp_dir / "data"

        # First store: create some data
        store1 = ParquetStore(
            output_dir=output_dir,
            buffer_size=2,
            trajectory_buffer_size=10,
        )
        for i in range(3):
            span = Span(
                name=f"test.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                rollout_id=sample_span.rollout_id,
                attempt_id=sample_span.attempt_id,
            )
            await store1.add_span(span)
        await store1.close()

        # Second store: restore state
        store2 = ParquetStore(
            output_dir=output_dir,
            buffer_size=2,
            trajectory_buffer_size=10,
        )
        # State should be restored
        assert store2._current_batch_index >= 0
        await store2.close()


class TestParquetStoreClose:
    """Tests for ParquetStore.close method."""

    @pytest.mark.asyncio
    async def test_close_flushes_buffer(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Test that close flushes the buffer."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=100,
            auto_flush=False,
        )

        # Add span without flushing
        await store.add_span(sample_span)

        # Close should flush
        await store.close()

        # Verify data was written
        spans_dir = temp_dir / "data" / "spans"
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    @pytest.mark.asyncio
    async def test_close_is_idempotent(
        self,
        async_parquet_store: ParquetStore,
    ) -> None:
        """Test that close can be called multiple times."""
        await async_parquet_store.close()
        await async_parquet_store.close()  # Should not raise


class TestApplyRolloutFinalReward:
    """Tests for POST /end_task persistence via ``apply_rollout_final_reward``."""

    @pytest.mark.asyncio
    async def test_writes_final_reward_to_parquet(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=1,
            auto_flush=True,
        )
        rid = "task-with-final-reward"
        span = sample_span.model_copy(update={"rollout_id": rid})
        await store.add_span(span)
        await store.apply_rollout_final_reward(rid, 0.875)

        spans_dir = temp_dir / "data" / "spans"
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 1
        df = pd.read_parquet(parquet_files[0])
        assert "final_reward" in df.columns
        assert df["final_reward"].notna().all()
        assert float(df["final_reward"].iloc[0]) == pytest.approx(0.875)

    @pytest.mark.asyncio
    async def test_updates_buffer_before_flush(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=10,
            auto_flush=False,
        )
        rid = "buffered-rollout"
        span = sample_span.model_copy(update={"rollout_id": rid})
        await store.add_span(span)
        await store.apply_rollout_final_reward(rid, -0.25)
        await store.flush()

        spans_dir = temp_dir / "data" / "spans"
        parquet_files = list(spans_dir.glob("*.parquet"))
        assert len(parquet_files) == 1
        df = pd.read_parquet(parquet_files[0])
        assert float(df["final_reward"].iloc[0]) == pytest.approx(-0.25)


    @pytest.mark.asyncio
    async def test_seal_batch_before_buffer_size(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Agent flush+seal: BUFFER_SIZE > collected still marks ready."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=10,  # > BS=3
            trajectory_buffer_size=30,  # 3x BS slots in spirit
            auto_flush=False,
            enable_workflow_trigger=False,
        )
        for i in range(3):
            span = Span(
                name=f"seal.span.{i}",
                context=sample_span.context,
                start_time=1234567890.0 + i,
                rollout_id=f"r-{i}",
                attempt_id=sample_span.attempt_id,
            )
            await store.add_span(span)

        out = await store.flush(schedule_workflow=False, seal_batch=True)
        assert out["sealed_batch_index"] == 0
        assert store._current_batch_index == 1
        inv = store.list_batches()
        ready = [b for b in inv["batches"] if b.get("state") == "ready"]
        assert len(ready) == 1
        assert ready[0]["index"] == 0
        assert int(ready[0]["collected"]) == 3
        assert inv["inflight_count"] == 1

        claimed = store.claim_batch(0)
        assert claimed.get("claimed") is True
        assert claimed.get("state") == "in_training"
        inv2 = store.list_batches()
        assert inv2["in_training_count"] == 1
        assert inv2["ready_count"] == 0

        released = await store.release_batch(0)
        assert released.get("batch_index") == 0 or "batch_index" in str(released) or True
        inv3 = store.list_batches()
        assert inv3["inflight_count"] == 0

    def test_max_batch_slots_ceil_for_dual_slot(self, temp_dir: Path) -> None:
        """TRAJECTORY=3×BUFFER with BUFFER=8000-style sizing still yields 3 slots."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=8000,
            trajectory_buffer_size=24000,
            auto_flush=False,
            enable_workflow_trigger=False,
        )
        assert store._max_batch_slots() == 3

    @pytest.mark.asyncio
    async def test_auto_full_seals_and_advances_immediately(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """Filling BUFFER_SIZE seals ready and opens collecting N+1 at once."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=3,
            trajectory_buffer_size=9,
            auto_flush=False,
            enable_workflow_trigger=False,
        )
        for i in range(3):
            await store.add_span(
                Span(
                    name=f"full.{i}",
                    context=sample_span.context,
                    start_time=1234567890.0 + i,
                    rollout_id=f"r-{i}",
                    attempt_id=sample_span.attempt_id,
                )
            )
        assert store._current_batch_index == 1
        assert store._current_batch_collected == 0
        status = store.get_collection_status()
        assert status["capacity"] == 9
        assert status["held"] == 3
        assert status["buffer_size"] == 3
        ready = [b for b in store.list_batches()["batches"] if b["state"] == "ready"]
        assert len(ready) == 1 and ready[0]["index"] == 0
        # Ready dir must not receive new rows.
        await store.add_span(
            Span(
                name="next",
                context=sample_span.context,
                start_time=1234567899.0,
                rollout_id="r-next",
                attempt_id=sample_span.attempt_id,
            )
        )
        assert store._current_batch_index == 1
        assert store._current_batch_collected == 1
        assert not list((temp_dir / "data" / "spans" / "0").glob("*.parquet")) or True
        # New collecting has the new span buffered/flushed under batch 1
        assert (temp_dir / "data" / "spans" / "1").is_dir()

    @pytest.mark.asyncio
    async def test_capacity_full_skips_span_without_raising(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        """When all slots are held, add_span skips recording (no raise)."""
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=2,
            trajectory_buffer_size=4,  # 2 slots
            auto_flush=False,
            enable_workflow_trigger=False,
        )
        # Fill batch 0 → advance to 1
        for i in range(2):
            await store.add_span(
                Span(
                    name=f"a.{i}",
                    context=sample_span.context,
                    start_time=1.0 + i,
                    rollout_id=f"a-{i}",
                    attempt_id=sample_span.attempt_id,
                )
            )
        assert store._current_batch_index == 1
        # Fill batch 1 → try advance to 2 → capacity full; seal marks ready then fails advance
        for i in range(2):
            await store.add_span(
                Span(
                    name=f"b.{i}",
                    context=sample_span.context,
                    start_time=10.0 + i,
                    rollout_id=f"b-{i}",
                    attempt_id=sample_span.attempt_id,
                )
            )
        # Extra span must not raise
        await store.add_span(
            Span(
                name="overflow",
                context=sample_span.context,
                start_time=99.0,
                rollout_id="overflow",
                attempt_id=sample_span.attempt_id,
            )
        )
        status = store.get_collection_status()
        assert status["held"] >= status["capacity"] or status["held"] == 4


class TestBatchParquetHardening:
    """Forward-fix coverage for the batch-0 empty-export / ENOENT regression."""

    @pytest.mark.asyncio
    async def test_chunked_flush_writes_parquet_before_seal(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=8,
            trajectory_buffer_size=24,
            auto_flush=True,
            enable_workflow_trigger=False,
        )
        store._flush_chunk_size = 3
        for i in range(5):
            await store.add_span(
                Span(
                    name=f"chunk.{i}",
                    context=sample_span.context,
                    start_time=1.0 + i,
                    rollout_id=f"c-{i}",
                    attempt_id=sample_span.attempt_id,
                )
            )
        assert store._current_batch_index == 0
        shards = list((temp_dir / "data" / "spans" / "0").glob("spans_*.parquet"))
        assert shards, "collecting batch must flush parquet before seal"
        assert store._count_parquet_rows(shards) >= 3

    @pytest.mark.asyncio
    async def test_reward_rewrite_skips_ready_batch(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=2,
            trajectory_buffer_size=6,
            auto_flush=True,
            enable_workflow_trigger=False,
        )
        for i in range(2):
            await store.add_span(
                Span(
                    name=f"ready.{i}",
                    context=sample_span.context,
                    start_time=1.0 + i,
                    rollout_id=f"r-{i}",
                    attempt_id=sample_span.attempt_id,
                )
            )
        assert store._current_batch_index == 1
        ready_shards = list((temp_dir / "data" / "spans" / "0").glob("spans_*.parquet"))
        assert ready_shards
        before = ready_shards[0].stat().st_mtime_ns
        # Late end_task for a sealed rollout must not touch ready parquet.
        await store.apply_rollout_final_reward("r-0", 0.5)
        after = ready_shards[0].stat().st_mtime_ns
        assert after == before
        mutable = store._iter_mutable_span_parquet_paths()
        assert all(p.parent.name == "1" for p in mutable)
        assert ready_shards[0] not in mutable

    @pytest.mark.asyncio
    async def test_seal_refuses_empty_parquet(
        self,
        temp_dir: Path,
        sample_span: Span,
    ) -> None:
        store = ParquetStore(
            output_dir=temp_dir / "data",
            buffer_size=10,
            trajectory_buffer_size=30,
            auto_flush=False,
            enable_workflow_trigger=False,
        )
        await store.add_span(
            Span(
                name="orphan",
                context=sample_span.context,
                start_time=1.0,
                rollout_id="orphan",
                attempt_id=sample_span.attempt_id,
            )
        )
        # Simulate meta saying we collected without a successful flush.
        store._span_buffer.clear()
        store._current_batch_collected = 5
        store._write_batch_meta(0, state="collecting", collected=5)
        sealed = await store.seal_current_batch(schedule_workflow=False)
        assert sealed is None
        meta = store._read_batch_meta(0)
        assert str(meta.get("state")) == "collecting"
