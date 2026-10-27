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

    def _bool_flag(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    @fastapi_app.post("/admin/flush")
    async def admin_flush(request: Request) -> JSONResponse:
        """Flush in-memory spans to Parquet (ADMIN_API_KEY).

        Body/query flags:
          - ``trigger_workflow`` (default false): fire WORKFLOW_TRIGGER_SCRIPT
          - ``seal_batch`` (default false): ops-only — seal collecting as ready
            (even if collected < BUFFER_SIZE), immediately advance to the next
            collecting dir, then optionally trigger. Same pipeline as auto-full.
            Agent feeders should not call this; they only write spans.
        """
        err = await _require_admin(request, admin_key)
        if err:
            return err

        trigger_workflow = False
        seal_batch = False
        try:
            body = await request.json()
            if isinstance(body, dict):
                if "trigger_workflow" in body:
                    trigger_workflow = _bool_flag(body.get("trigger_workflow"))
                if "seal_batch" in body:
                    seal_batch = _bool_flag(body.get("seal_batch"))
        except Exception:
            pass
        qp = request.query_params.get("trigger_workflow")
        if qp is not None:
            trigger_workflow = _bool_flag(qp)
        qp_seal = request.query_params.get("seal_batch")
        if qp_seal is not None:
            seal_batch = _bool_flag(qp_seal)

        if not hasattr(store, "flush"):
            return JSONResponse(
                {"error": "flush not supported for this storage backend"},
                status_code=501,
            )

        flush_result: Dict[str, Any] = {}
        try:
            if isinstance(store, ParquetStore):
                out = await store.flush(
                    schedule_workflow=trigger_workflow,
                    seal_batch=seal_batch,
                )
                if isinstance(out, dict):
                    flush_result = out
            else:
                await store.flush()
        except Exception as exc:  # noqa: BLE001
            # BatchCapacityFull and seal errors surface as 409/400.
            from openclaw_tracer.storage.parquet_store import BatchCapacityFull

            if isinstance(exc, BatchCapacityFull):
                return JSONResponse(
                    {"error": str(exc), "code": "batch_capacity_full"},
                    status_code=409,
                )
            if isinstance(exc, ValueError):
                return JSONResponse(
                    {"error": str(exc), "code": "invalid_flush"},
                    status_code=400,
                )
            raise

        status_payload: Dict[str, Any] = {
            "ok": True,
            "trigger_workflow": trigger_workflow,
            "seal_batch": seal_batch,
            **flush_result,
        }
        if hasattr(store, "get_collection_status"):
            status_payload["status"] = store.get_collection_status()
        if isinstance(store, ParquetStore):
            status_payload["batches"] = store.list_batches()
        return JSONResponse(status_payload, status_code=200)

    @fastapi_app.post("/admin/batches/{batch_index}/claim")
    async def claim_batch(request: Request, batch_index: int) -> JSONResponse:
        """Claim a ready batch for training (state → in_training)."""
        err = await _require_admin(request, admin_key)
        if err:
            return err
        if not isinstance(store, ParquetStore):
            return JSONResponse(
                {"error": "batch claim requires ParquetStore"},
                status_code=501,
            )
        try:
            result = store.claim_batch(batch_index)
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

    @fastapi_app.get("/admin/config")
    async def get_config(request: Request) -> JSONResponse:
        """Read runtime knobs: buffer_size, trajectory_buffer_size, workflow_trigger_script."""
        err = await _require_admin(request, admin_key)
        if err:
            return err
        if not isinstance(store, ParquetStore):
            return JSONResponse(
                {"error": "runtime config requires ParquetStore"},
                status_code=501,
            )
        return JSONResponse({"ok": True, **store.get_runtime_config()})

    @fastapi_app.patch("/admin/config")
    async def patch_config(request: Request) -> JSONResponse:
        """Hot-update runtime knobs (partial body).

        Accepted JSON fields (all optional)::

            {
              "buffer_size": 8000,
              "trajectory_buffer_size": 24000,
              "workflow_trigger_script": "/app/trigger/run.sh"
            }

        Aliases: ``BUFFER_SIZE``, ``TRAJECTORY_BUFFER_SIZE``, ``WORKFLOW_TRIGGER_SCRIPT``.
        """
        err = await _require_admin(request, admin_key)
        if err:
            return err
        if not isinstance(store, ParquetStore):
            return JSONResponse(
                {"error": "runtime config requires ParquetStore"},
                status_code=501,
            )
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = None
        if not isinstance(body, dict) or not body:
            return JSONResponse(
                {
                    "error": "JSON body required with at least one of: "
                    "buffer_size, trajectory_buffer_size, workflow_trigger_script",
                    "code": "invalid_config",
                },
                status_code=400,
            )

        kwargs: Dict[str, Any] = {}
        if "buffer_size" in body or "BUFFER_SIZE" in body:
            raw = body["buffer_size"] if "buffer_size" in body else body["BUFFER_SIZE"]
            try:
                kwargs["buffer_size"] = int(raw)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "buffer_size must be an integer", "code": "invalid_config"},
                    status_code=400,
                )
        if "trajectory_buffer_size" in body or "TRAJECTORY_BUFFER_SIZE" in body:
            raw = (
                body["trajectory_buffer_size"]
                if "trajectory_buffer_size" in body
                else body["TRAJECTORY_BUFFER_SIZE"]
            )
            try:
                kwargs["trajectory_buffer_size"] = int(raw)
            except (TypeError, ValueError):
                return JSONResponse(
                    {
                        "error": "trajectory_buffer_size must be an integer",
                        "code": "invalid_config",
                    },
                    status_code=400,
                )
        if "workflow_trigger_script" in body or "WORKFLOW_TRIGGER_SCRIPT" in body:
            raw = (
                body["workflow_trigger_script"]
                if "workflow_trigger_script" in body
                else body["WORKFLOW_TRIGGER_SCRIPT"]
            )
            if raw is None or not str(raw).strip():
                return JSONResponse(
                    {
                        "error": "workflow_trigger_script must be a non-empty path",
                        "code": "invalid_config",
                    },
                    status_code=400,
                )
            kwargs["workflow_trigger_script"] = str(raw).strip()

        if not kwargs:
            return JSONResponse(
                {
                    "error": "no recognized fields; use buffer_size, "
                    "trajectory_buffer_size, and/or workflow_trigger_script",
                    "code": "invalid_config",
                },
                status_code=400,
            )

        try:
            config = store.update_runtime_config(**kwargs)
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc), "code": "invalid_config"},
                status_code=400,
            )
        return JSONResponse({"ok": True, **config})

    fastapi_app.state.openclaw_admin_ops_registered = True
    logger.info(
        "[AdminOps] Registered /admin/flush, /admin/batches, /admin/spans/release, "
        "/admin/config (enabled=%s)",
        bool(admin_key),
    )
