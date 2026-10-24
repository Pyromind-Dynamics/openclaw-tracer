# Copyright (c) 2025 OpenClaw-Tracer
# Tests for release_spans: the consumption watermark for span shards + images.

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openclaw_tracer.storage.parquet_store import ParquetStore


def _write_shard(path, trace_ids) -> None:
    table = pa.table({"trace_id": pa.array(trace_ids, pa.string())})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _media_dir(store: ParquetStore, trace_id: str) -> "object":
    folder = store.output_dir / "spans" / "media" / "images" / trace_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "span1_0.jpg").write_bytes(b"img")
    return folder


@pytest.fixture()
def store(tmp_path):
    # trajectory_buffer_size=0 keeps the store in the flat (non-batch) layout the
    # deployment actually runs, where /admin/batches is empty and only span
    # release can bound what the next export sees.
    return ParquetStore(output_dir=tmp_path, buffer_size=1, trajectory_buffer_size=0)


@pytest.mark.asyncio
async def test_release_named_shards_and_prune_their_images(store):
    spans_dir = store.output_dir / "spans"
    old = spans_dir / "spans_20261023_070000.parquet"
    new = spans_dir / "spans_20261024_060000.parquet"
    _write_shard(old, ["trace-old-1", "trace-old-2"])
    _write_shard(new, ["trace-new-1"])
    old_media = _media_dir(store, "trace-old-1")
    kept_media = _media_dir(store, "trace-new-1")

    result = await store.release_spans([str(old.name)], keep_current=False)

    assert result["deleted_file_count"] == 1
    assert result["deleted_rows"] == 2
    assert not old.exists()
    assert new.exists()  # shard a later round produced is untouched
    assert not old_media.exists()  # images of the consumed trace go with it
    assert kept_media.exists()  # unconsumed trace keeps its images
    assert result["deleted_media_count"] == 1
    assert [p.split("/")[-1] for p in result["kept_files"]] == [new.name]


@pytest.mark.asyncio
async def test_current_window_is_spared(store):
    spans_dir = store.output_dir / "spans"
    live = store._current_window_path()
    _write_shard(live, ["trace-live"])
    _write_shard(spans_dir / "spans_20261023_070000.parquet", ["trace-old"])
    live_media = _media_dir(store, "trace-live")

    result = await store.release_spans(keep_current=True)

    assert live.exists()
    assert live_media.exists()
    assert [s["reason"] for s in result["skipped"]] == ["current_window"]
    assert result["deleted_file_count"] == 1


@pytest.mark.asyncio
async def test_untouched_trace_keeps_media_when_other_shard_still_holds_it(store):
    spans_dir = store.output_dir / "spans"
    first = spans_dir / "spans_20261023_070000.parquet"
    second = spans_dir / "spans_20261023_080000.parquet"
    # A trace can span two time windows; deleting only one must keep its images.
    _write_shard(first, ["trace-split"])
    _write_shard(second, ["trace-split"])
    media = _media_dir(store, "trace-split")

    result = await store.release_spans([str(first.name)], keep_current=False)

    assert result["deleted_file_count"] == 1
    assert result["deleted_media_count"] == 0
    assert media.exists()


@pytest.mark.asyncio
async def test_live_shard_is_sealed_before_delete(store):
    live = store._current_window_path()
    _write_shard(live, ["trace-live"])
    media = _media_dir(store, "trace-live")

    # keep_current=False means "the export consumed the live shard too".
    result = await store.release_spans([str(live)], keep_current=False)

    assert result["deleted_file_count"] == 1
    assert not live.exists()
    assert not media.exists()
    assert result["sealed_shard"] and "_g1" in result["sealed_shard"], result
    # Post-seal writes land in the new shard, never in the deleted one.
    assert store._shard_name() == result["sealed_shard"]


@pytest.mark.asyncio
async def test_host_style_path_resolves_to_the_mounted_shard(store):
    shard = store.output_dir / "spans" / "spans_20261023_073000.parquet"
    _write_shard(shard, ["trace-old"])

    # What the orchestrator reports: its own view of the mounted tree.
    host_style = "datas/openclaw-val-eval/spans/spans_20261023_073000.parquet"
    result = await store.release_spans([host_style], keep_current=False)

    assert result["deleted_file_count"] == 1
    assert not shard.exists()


@pytest.mark.asyncio
async def test_release_is_idempotent_for_unknown_shards(store):
    result = await store.release_spans(["spans/spans_19990101_000000.parquet"])

    assert result["deleted_file_count"] == 0
    assert result["skipped"] == [
        {
            "path": str(store.output_dir / "spans" / "spans_19990101_000000.parquet"),
            "reason": "missing",
        }
    ]


@pytest.mark.asyncio
async def test_path_outside_spans_is_refused(store, tmp_path):
    outside = tmp_path / "eval_summary.json"
    outside.write_text(json.dumps({"keep": "me"}), encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        await store.release_spans([str(outside)], keep_current=False)

    assert outside.exists()


@pytest.mark.asyncio
async def test_batch_mode_requires_explicit_files(tmp_path):
    store = ParquetStore(output_dir=tmp_path, buffer_size=1, trajectory_buffer_size=64)
    with pytest.raises(ValueError, match="explicit files"):
        await store.release_spans(None)
