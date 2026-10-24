# Copyright (c) 2025 OpenClaw-Tracer
# Admin ops: flush + trajectory batch lifecycle (ADMIN_API_KEY).

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from openclaw_tracer.routing_admin import _require_admin
from openclaw_tracer.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


def register_admin_ops_routes(fastapi_app: Any, proxy: Any) -> None:
    """Register /admin/flush and /admin/batches* on the LiteLLM FastAPI app."""
    if getattr(fastapi_app.state, "openclaw_admin_ops_registered", False):
        return

    admin_key: Optional[str] = getattr(proxy, "admin_api_key", None)
    store = proxy.store

    @fastapi_app.post("/admin/flush")
    async def admin_flush(request: Request) -> JSONResponse:
        """Flush in-memory spans to Parquet (ADMIN_API_KEY).

        Optional ``trigger_workflow`` (JSON body or query, default false).
        """
        err = await _require_admin(request, admin_key)
        if err:
            return err

        trigger_workflow = False
        try:
            body = await request.json()
            if isinstance(body, dict) and "trigger_workflow" in body:
                raw = body.get("trigger_workflow")
                if isinstance(raw, bool):
                    trigger_workflow = raw
                elif isinstance(raw, str):
                    trigger_workflow = raw.strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                else:
                    trigger_workflow = bool(raw)
        except Exception:
            pass
        qp = request.query_params.get("trigger_workflow")
        if qp is not None:
            trigger_workflow = qp.strip().lower() in {"1", "true", "yes", "on"}

        if not hasattr(store, "flush"):
            return JSONResponse(
                {"error": "flush not supported for this storage backend"},
                status_code=501,
            )

        if trigger_workflow and isinstance(store, ParquetStore):
            await store.flush(schedule_workflow=True)
        else:
            await store.flush()

        status_payload: Dict[str, Any] = {
            "ok": True,
            "trigger_workflow": trigger_workflow,
        }
        if hasattr(store, "get_collection_status"):
            status_payload["status"] = store.get_collection_status()
        return JSONResponse(status_payload, status_code=200)

    @fastapi_app.get("/admin/batches")
    async def list_batches(request: Request) -> JSONResponse:
        """List trajectory batches and their hold/ready state."""
        err = await _require_admin(request, admin_key)
        if err:
            return err
        if not isinstance(store, ParquetStore):
            return JSONResponse(
                {"error": "batch listing requires ParquetStore"},
                status_code=501,
            )
        return JSONResponse({"ok": True, **store.list_batches()})

    @fastapi_app.post("/admin/batches/{batch_index}/release")
    async def release_batch(request: Request, batch_index: int) -> JSONResponse:
        """Mark training complete for a ready batch and delete its on-disk data.

        Only ``ready`` batches can be released. Fires DATA_DELETE_CALLBACK with
        reason ``training_complete``.
        """
        err = await _require_admin(request, admin_key)
        if err:
            return err
        if not isinstance(store, ParquetStore):
            return JSONResponse(
                {"error": "batch release requires ParquetStore"},
                status_code=501,
            )
        try:
            result = await store.release_batch(batch_index)
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc), "code": "invalid_batch"},
                status_code=400,
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                {"error": str(exc), "code": "batch_not_found"},
                status_code=404,
            )
        return JSONResponse({"ok": True, **result})

    @fastapi_app.post("/admin/spans/release")
    async def release_spans(request: Request) -> JSONResponse:
        """Delete consumed span shards (and their images) so they cannot be reread.

        Body (all optional)::

            {"files": ["spans/spans_20261023_073000.parquet"],  // omit = every shard
             "keep_current": true,                              // spare the live window
             "reason": "training_complete"}

        The caller names what its export actually consumed; leaving ``files`` out
        deletes all shards except the window the tracer is still appending to.
        """
        err = await _require_admin(request, admin_key)
        if err:
            return err
        if not isinstance(store, ParquetStore):
            return JSONResponse(
                {"error": "span release requires ParquetStore"},
                status_code=501,
            )

        files = None
        keep_current = True
        reason = "training_complete"
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - empty body is valid
            body = None
        if isinstance(body, dict):
            raw_files = body.get("files")
            if isinstance(raw_files, str):
                files = [raw_files]
            elif isinstance(raw_files, (list, tuple)):
                files = [str(f) for f in raw_files if str(f).strip()]
            if "keep_current" in body:
                keep_current = bool(body.get("keep_current"))
            if str(body.get("reason") or "").strip():
                reason = str(body["reason"]).strip()

        try:
            result = await store.release_spans(
                files=files, keep_current=keep_current, reason=reason
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc), "code": "invalid_span_path"},
                status_code=400,
            )
        return JSONResponse({"ok": True, **result})

    fastapi_app.state.openclaw_admin_ops_registered = True
    logger.info(
        "[AdminOps] Registered /admin/flush, /admin/batches and /admin/spans/release "
        "(enabled=%s)",
        bool(admin_key),
    )
