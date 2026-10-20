# Copyright (c) 2025 OpenClaw-Tracer
# Async routing-change callbacks: HTTP webhook and/or shell script.

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from openclaw_tracer.routing_state import redact_model_list_for_callback

logger = logging.getLogger(__name__)


def _callback_url() -> str:
    return (os.getenv("ROUTING_CHANGE_CALLBACK_URL") or "").strip()


def _callback_script() -> str:
    return (os.getenv("ROUTING_CHANGE_CALLBACK_SCRIPT") or "").strip()


def _http_timeout() -> float:
    raw = os.getenv("ROUTING_CHANGE_CALLBACK_TIMEOUT", "5")
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 5.0


def _script_timeout() -> float:
    raw = os.getenv("ROUTING_CHANGE_CALLBACK_SCRIPT_TIMEOUT", "30")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def build_routing_event(
    *,
    reason: str,
    previous: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
    op: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event": "routing.updated",
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "previous": redact_model_list_for_callback(previous),
        "current": redact_model_list_for_callback(current),
        "op": op or {},
    }


def schedule_routing_change_callback(payload: Dict[str, Any]) -> None:
    """Fire-and-forget HTTP and/or shell callbacks. Never blocks the caller."""
    url = _callback_url()
    script = _callback_script()
    if not url and not script:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop; skipping routing change callback")
        return

    async def _run() -> None:
        if url:
            try:
                await _post_http(url, payload)
            except Exception:
                logger.exception("Routing HTTP callback failed for %s", url)
        if script:
            try:
                await _run_script(script, payload)
            except Exception:
                logger.exception("Routing script callback failed for %s", script)

    loop.create_task(_run())


async def _post_http(url: str, payload: Dict[str, Any]) -> None:
    timeout = _http_timeout()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        if response.status_code >= 400:
            logger.error(
                "Routing HTTP callback %s returned %s: %s",
                url,
                response.status_code,
                (response.text or "")[:500],
            )
        else:
            logger.info("Routing HTTP callback %s ok (%s)", url, response.status_code)


async def _run_script(script: str, payload: Dict[str, Any]) -> None:
    path = Path(os.path.expanduser(script))
    if not path.is_file():
        logger.warning("Routing callback script not found: %s", path)
        return

    fd, tmp_name = tempfile.mkstemp(prefix="routing_cb_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            str(path),
            tmp_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_script_timeout()
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("Routing callback script %s timed out", path)
            return
        if proc.returncode != 0:
            err_text = (stderr.decode(errors="replace") if stderr else "").strip()
            logger.error(
                "Routing callback script %s exited %s: %s",
                path,
                proc.returncode,
                err_text[-2000:] or "(no stderr)",
            )
        else:
            logger.info("Routing callback script %s exited 0", path)
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
