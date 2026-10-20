# Copyright (c) 2025 OpenClaw-Tracer

"""Tests for data-URL image externalization before Parquet span export."""

from __future__ import annotations

from pathlib import Path

from openclaw_tracer.proxy.image_externalize import (
    externalize_inline_images_for_span,
    redact_data_url_images_in_json_body_for_log,
)

# 1x1 transparent PNG
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_data_url_image_written_as_jpeg_and_path_in_messages(tmp_path: Path) -> None:
    trace_id = "a" * 32
    span_id = "b" * 16
    data_url = f"data:image/png;base64,{_TINY_PNG_B64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    out_messages, out_system = externalize_inline_images_for_span(
        tmp_path, trace_id, span_id, messages, None
    )

    assert out_system is None
    rel = Path("spans") / "media" / "images" / trace_id / f"{span_id}_0.jpg"
    assert (tmp_path / rel).is_file()

    url = out_messages[0]["content"][1]["image_url"]["url"]
    assert url == rel.as_posix()
    assert not url.startswith("data:")


def test_https_image_url_unchanged(tmp_path: Path) -> None:
    trace_id = "c" * 32
    span_id = "d" * 16
    https_url = "https://example.com/picture.png"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": https_url}},
            ],
        }
    ]
    out_messages, _ = externalize_inline_images_for_span(
        tmp_path, trace_id, span_id, messages, None
    )
    assert out_messages[0]["content"][0]["image_url"]["url"] == https_url
    media_dir = tmp_path / "spans" / "media" / "images" / trace_id
    assert not media_dir.exists() or not any(media_dir.glob("*.jpg"))


def test_system_multimodal_list_gets_externalized(tmp_path: Path) -> None:
    trace_id = "e" * 32
    span_id = "f" * 16
    data_url = f"data:image/png;base64,{_TINY_PNG_B64}"
    system_parts = [
        {"type": "text", "text": "sys"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    out_messages, out_system = externalize_inline_images_for_span(
        tmp_path, trace_id, span_id, [], system_parts
    )
    assert out_messages == []
    rel = Path("spans") / "media" / "images" / trace_id / f"{span_id}_0.jpg"
    assert (tmp_path / rel).is_file()
    assert out_system[1]["image_url"]["url"] == rel.as_posix()


def test_custom_batch_media_prefix(tmp_path: Path) -> None:
    trace_id = "1" * 32
    span_id = "2" * 16
    data_url = f"data:image/png;base64,{_TINY_PNG_B64}"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": data_url}}],
        }
    ]
    prefix = "spans/3/media/images"
    out_messages, _ = externalize_inline_images_for_span(
        tmp_path, trace_id, span_id, messages, None, media_images_prefix=prefix
    )
    rel = Path("spans") / "3" / "media" / "images" / trace_id / f"{span_id}_0.jpg"
    assert (tmp_path / rel).is_file()
    assert out_messages[0]["content"][0]["image_url"]["url"] == rel.as_posix()


def test_redact_log_body_strips_base64_from_messages() -> None:
    data_url = f"data:image/png;base64,{_TINY_PNG_B64}"
    raw = (
        '{"model":"m","messages":[{"role":"user","content":['
        '{"type":"text","text":"hi"},'
        '{"type":"image_url","image_url":{"url":"'
        + data_url
        + '"}}]}]}'
    )
    out = redact_data_url_images_in_json_body_for_log(raw)
    assert _TINY_PNG_B64 not in out
    assert "data:image/png;base64" not in out
    assert "<inline_image_base64_redacted>" in out


def test_redact_log_body_https_unchanged() -> None:
    raw = '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://x/y"}}]}]}'
    out = redact_data_url_images_in_json_body_for_log(raw)
    assert "https://x/y" in out


def test_redact_log_body_regex_fallback_on_invalid_json() -> None:
    junk = 'not-json data:image/png;base64,AAAABBBBCCCC'
    out = redact_data_url_images_in_json_body_for_log(junk)
    assert "AAAABBBBCCCC" not in out
    assert "data:image/*;base64,<redacted>" in out
