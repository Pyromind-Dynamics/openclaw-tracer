# Copyright (c) 2025 OpenClaw-Tracer
# Admin API for routing CRUD (ADMIN_API_KEY).

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from openclaw_tracer.routing_state import (
    ensure_deployment_ids,
    find_deployment,
    get_deployment_id,
    shallow_merge_dict,
    validate_weights_for_model_names,
)

logger = logging.getLogger(__name__)


def _extract_admin_key(headers: Dict[str, str]) -> Optional[str]:
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    for key, value in headers.items():
        if key.lower() == "x-admin-api-key":
            return value.strip()
    return None


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": "Unauthorized",
                "type": "authentication_error",
                "code": "invalid_admin_api_key",
            }
        },
        status_code=401,
    )


def _admin_disabled() -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": "Admin API disabled: set ADMIN_API_KEY to enable",
                "type": "admin_disabled",
                "code": "admin_api_key_missing",
            }
        },
        status_code=503,
    )


async def _require_admin(request: Request, admin_api_key: Optional[str]) -> Optional[JSONResponse]:
    if not admin_api_key:
        return _admin_disabled()
    provided = _extract_admin_key(dict(request.headers))
    if provided is None or provided != admin_api_key:
        return _unauthorized()
    return None


def _parse_drain_flag(body: Dict[str, Any], default: bool) -> bool:
    if "drain" not in body:
        return default
    val = body.get("drain")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def register_routing_admin_routes(fastapi_app: Any, proxy: Any) -> None:
    """Register /admin/routing/* routes on the LiteLLM FastAPI app."""
    if getattr(fastapi_app.state, "openclaw_routing_admin_registered", False):
        return

    admin_key: Optional[str] = getattr(proxy, "admin_api_key", None)

    @fastapi_app.get("/admin/routing")
    async def get_routing(request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        return JSONResponse(
            {
                "ok": True,
                "updated_at": proxy.routing_updated_at,
                "model_list": proxy.model_list,
            }
        )

    @fastapi_app.get("/admin/routing/deployments/{deployment_id}")
    async def get_deployment(deployment_id: str, request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        _idx, entry = find_deployment(proxy.model_list, deployment_id)
        if entry is None:
            return JSONResponse(
                {"error": "deployment not found", "id": deployment_id},
                status_code=404,
            )
        return JSONResponse({"ok": True, "deployment": entry})

    @fastapi_app.get("/admin/routing/models/{model_name}")
    async def get_model_deployments(model_name: str, request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        entries = [e for e in proxy.model_list if e.get("model_name") == model_name]
        return JSONResponse(
            {
                "ok": True,
                "model_name": model_name,
                "deployments": entries,
                "updated_at": proxy.routing_updated_at,
            }
        )

    @fastapi_app.put("/admin/routing")
    async def put_routing(request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        model_list = body.get("model_list")
        if not isinstance(model_list, list):
            return JSONResponse({"error": "model_list must be a list"}, status_code=400)

        reason = body.get("reason") or "cutover"
        drain = _parse_drain_flag(body, default=True)
        try:
            result = await proxy.apply_model_list(
                model_list,
                reason=reason,
                drain=drain,
                op={"method": "PUT", "path": "/admin/routing"},
            )
        except RoutingApplyError as exc:
            return JSONResponse(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )
        return JSONResponse({"ok": True, **result})

    @fastapi_app.post("/admin/routing/deployments")
    async def create_deployment(request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        deployment = body.get("deployment")
        if not isinstance(deployment, dict) or not deployment.get("model_name"):
            return JSONResponse(
                {"error": "deployment with model_name is required"},
                status_code=400,
            )
        if not isinstance(deployment.get("litellm_params"), dict):
            return JSONResponse(
                {"error": "deployment.litellm_params must be an object"},
                status_code=400,
            )

        reason = body.get("reason") or "create"
        drain = _parse_drain_flag(body, default=False)
        new_list = ensure_deployment_ids(proxy.model_list + [deployment])
        # last entry is the new one (ensure ids on all)
        created = new_list[-1]
        try:
            result = await proxy.apply_model_list(
                new_list,
                reason=reason,
                drain=drain,
                op={
                    "method": "POST",
                    "path": "/admin/routing/deployments",
                    "id": get_deployment_id(created),
                },
            )
        except RoutingApplyError as exc:
            return JSONResponse(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )
        return JSONResponse(
            {
                "ok": True,
                "deployment": created,
                **{k: v for k, v in result.items() if k != "model_list"},
                "model_list": result["model_list"],
            }
        )

    @fastapi_app.put("/admin/routing/deployments/{deployment_id}")
    async def replace_deployment(deployment_id: str, request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        deployment = body.get("deployment")
        if not isinstance(deployment, dict):
            return JSONResponse({"error": "deployment object required"}, status_code=400)

        idx, existing = find_deployment(proxy.model_list, deployment_id)
        if existing is None:
            return JSONResponse(
                {"error": "deployment not found", "id": deployment_id},
                status_code=404,
            )

        replacement = deepcopy(deployment)
        info = replacement.setdefault("model_info", {})
        if not isinstance(info, dict):
            info = {}
            replacement["model_info"] = info
        info["id"] = deployment_id

        new_list = deepcopy(proxy.model_list)
        new_list[idx] = replacement
        reason = body.get("reason") or "update"
        drain = _parse_drain_flag(body, default=False)
        try:
            result = await proxy.apply_model_list(
                new_list,
                reason=reason,
                drain=drain,
                op={
                    "method": "PUT",
                    "path": f"/admin/routing/deployments/{deployment_id}",
                    "id": deployment_id,
                },
            )
        except RoutingApplyError as exc:
            return JSONResponse(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )
        return JSONResponse({"ok": True, **result})

    @fastapi_app.patch("/admin/routing/deployments/{deployment_id}")
    async def patch_deployment(deployment_id: str, request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        patch = body.get("deployment")
        if not isinstance(patch, dict):
            # allow top-level fields as patch
            patch = {
                k: v
                for k, v in body.items()
                if k in {"model_name", "litellm_params", "model_info"}
            }
        if not patch:
            return JSONResponse({"error": "empty patch"}, status_code=400)

        idx, existing = find_deployment(proxy.model_list, deployment_id)
        if existing is None:
            return JSONResponse(
                {"error": "deployment not found", "id": deployment_id},
                status_code=404,
            )

        merged = shallow_merge_dict(existing, patch)
        info = merged.setdefault("model_info", {})
        if not isinstance(info, dict):
            info = {}
            merged["model_info"] = info
        info["id"] = deployment_id

        new_list = deepcopy(proxy.model_list)
        new_list[idx] = merged
        reason = body.get("reason") or "update"
        drain = _parse_drain_flag(body, default=False)
        try:
            result = await proxy.apply_model_list(
                new_list,
                reason=reason,
                drain=drain,
                op={
                    "method": "PATCH",
                    "path": f"/admin/routing/deployments/{deployment_id}",
                    "id": deployment_id,
                },
            )
        except RoutingApplyError as exc:
            return JSONResponse(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )
        return JSONResponse({"ok": True, **result})

    @fastapi_app.delete("/admin/routing/deployments/{deployment_id}")
    async def delete_deployment(deployment_id: str, request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err

        drain = True
        reason = "delete"
        try:
            body = await request.json()
            if isinstance(body, dict):
                drain = _parse_drain_flag(body, default=True)
                reason = body.get("reason") or reason
        except Exception:
            # empty body is fine for DELETE
            pass

        # also allow ?drain=false
        qp = request.query_params.get("drain")
        if qp is not None:
            drain = qp.strip().lower() in {"1", "true", "yes", "on"}

        idx, existing = find_deployment(proxy.model_list, deployment_id)
        if existing is None:
            return JSONResponse(
                {"error": "deployment not found", "id": deployment_id},
                status_code=404,
            )

        new_list = [e for i, e in enumerate(proxy.model_list) if i != idx]
        try:
            result = await proxy.apply_model_list(
                new_list,
                reason=reason,
                drain=drain,
                op={
                    "method": "DELETE",
                    "path": f"/admin/routing/deployments/{deployment_id}",
                    "id": deployment_id,
                },
            )
        except RoutingApplyError as exc:
            return JSONResponse(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )
        return JSONResponse({"ok": True, **result})

    @fastapi_app.patch("/admin/routing/weights")
    async def patch_weights(request: Request) -> JSONResponse:
        err = await _require_admin(request, admin_key)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        model_name = body.get("model_name")
        weights = body.get("weights")
        if not model_name or not isinstance(weights, list):
            return JSONResponse(
                {"error": "model_name and weights[] are required"},
                status_code=400,
            )

        new_list = deepcopy(proxy.model_list)
        for item in weights:
            if not isinstance(item, dict):
                return JSONResponse({"error": "invalid weights entry"}, status_code=400)
            dep_id = item.get("id")
            weight = item.get("weight")
            if dep_id is None or weight is None:
                return JSONResponse(
                    {"error": "each weights entry needs id and weight"},
                    status_code=400,
                )
            try:
                weight_f = float(weight)
            except (TypeError, ValueError):
                return JSONResponse({"error": "weight must be a number"}, status_code=400)
            if weight_f < 0:
                return JSONResponse({"error": "weight must be >= 0"}, status_code=400)

            idx, entry = find_deployment(new_list, str(dep_id))
            if entry is None:
                return JSONResponse(
                    {"error": "deployment not found", "id": dep_id},
                    status_code=404,
                )
            if entry.get("model_name") != model_name:
                return JSONResponse(
                    {
                        "error": "deployment model_name mismatch",
                        "id": dep_id,
                        "expected": model_name,
                        "actual": entry.get("model_name"),
                    },
                    status_code=400,
                )
            params = new_list[idx].setdefault("litellm_params", {})
            params["weight"] = weight_f

        # Removing weight from a backend: if any weight became 0, drain by default
        removing = any(
            float((w.get("weight") if isinstance(w, dict) else 0) or 0) <= 0 for w in weights
        )
        drain = _parse_drain_flag(body, default=removing)
        reason = body.get("reason") or "migration"
        try:
            result = await proxy.apply_model_list(
                new_list,
                reason=reason,
                drain=drain,
                op={"method": "PATCH", "path": "/admin/routing/weights"},
            )
        except RoutingApplyError as exc:
            return JSONResponse(
                {"error": str(exc), "code": exc.code},
                status_code=exc.status_code,
            )
        return JSONResponse({"ok": True, **result})

    fastapi_app.state.openclaw_routing_admin_registered = True
    logger.info(
        "[RoutingAdmin] Registered /admin/routing routes (enabled=%s)",
        bool(admin_key),
    )


class RoutingApplyError(Exception):
    """Raised when apply_model_list fails validation or drain."""

    def __init__(self, message: str, *, code: str = "routing_error", status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
