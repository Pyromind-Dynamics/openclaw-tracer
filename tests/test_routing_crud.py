# Copyright (c) 2025 OpenClaw-Tracer
# Tests for routing CRUD, drain, and change callbacks.

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from openclaw_tracer.proxy.llm_proxy import AuthMiddleware, InflightTracker
from openclaw_tracer.routing_admin import RoutingApplyError, register_routing_admin_routes
from openclaw_tracer.routing_callback import (
    build_routing_event,
    schedule_routing_change_callback,
)
from openclaw_tracer.routing_state import (
    ensure_deployment_ids,
    find_deployment,
    get_deployment_id,
    load_routing_state,
    redact_model_list_for_callback,
    save_routing_state,
    validate_weights_for_model_names,
)


def _dep(
    model_name: str,
    api_base: str,
    weight: float = 100,
    api_key: str = "sk-secret-key-abcdef",
    dep_id: Optional[str] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "model_name": model_name,
        "litellm_params": {
            "model": f"openai/{model_name}",
            "api_base": api_base,
            "api_key": api_key,
            "weight": weight,
        },
    }
    if dep_id:
        entry["model_info"] = {"id": dep_id}
    return entry


class FakeProxy:
    """Minimal stand-in for LLMProxy admin routes + apply_model_list."""

    def __init__(
        self,
        model_list: List[Dict[str, Any]],
        *,
        admin_api_key: Optional[str] = "admin-secret",
        state_path: Optional[Path] = None,
    ) -> None:
        self.model_list = ensure_deployment_ids(model_list)
        self.admin_api_key = admin_api_key
        self.routing_updated_at = None
        self.routing_state_path = state_path or Path("/tmp/routing_state_test.json")
        self.inflight = InflightTracker()
        self._routing_lock = asyncio.Lock()
        self.reload_calls: List[Dict[str, Any]] = []
        self.callback_payloads: List[Dict[str, Any]] = []

    def _drain_timeout_seconds(self) -> float:
        return 0.2

    async def reload_models(
        self,
        model_list: List[Dict[str, Any]],
        *,
        reason: str = "reload",
        op: Optional[Dict[str, Any]] = None,
        persist: bool = True,
        schedule_callback: bool = True,
    ) -> Dict[str, Any]:
        previous = deepcopy(self.model_list)
        prepared = ensure_deployment_ids(model_list)
        self.model_list = prepared
        self.routing_updated_at = "2026-01-01T00:00:00Z"
        self.reload_calls.append({"reason": reason, "op": op, "model_list": prepared})
        if persist:
            save_routing_state(
                self.routing_state_path, prepared, updated_at=self.routing_updated_at
            )
        if schedule_callback:
            payload = build_routing_event(
                reason=reason, previous=previous, current=prepared, op=op
            )
            self.callback_payloads.append(payload)
            schedule_routing_change_callback(payload)
        return {
            "updated_at": self.routing_updated_at,
            "model_list": prepared,
            "deployment_count": len(prepared),
        }

    async def apply_model_list(
        self,
        model_list: List[Dict[str, Any]],
        *,
        reason: str = "manual",
        drain: bool = False,
        op: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prepared = ensure_deployment_ids(model_list)
        weight_err = validate_weights_for_model_names(prepared)
        if weight_err:
            raise RoutingApplyError(weight_err, code="invalid_weights", status_code=400)

        async with self._routing_lock:
            if drain:
                ok = await self.inflight.wait_zero(self._drain_timeout_seconds())
                if not ok:
                    raise RoutingApplyError(
                        f"drain timed out (inflight={self.inflight.count})",
                        code="drain_timeout",
                        status_code=409,
                    )
            return await self.reload_models(
                prepared, reason=reason, op=op, persist=True, schedule_callback=True
            )


def _admin_app(proxy: FakeProxy) -> FastAPI:
    app = FastAPI()
    register_routing_admin_routes(app, proxy)
    return app


def _admin_headers(key: str = "admin-secret") -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --- routing_state ---


def test_ensure_deployment_ids_stable() -> None:
    raw = [_dep("m1", "http://a"), _dep("m1", "http://b", dep_id="fixed-id")]
    first = ensure_deployment_ids(raw)
    second = ensure_deployment_ids(first)
    assert get_deployment_id(first[1]) == "fixed-id"
    assert get_deployment_id(second[0]) == get_deployment_id(first[0])
    assert get_deployment_id(first[0]) != get_deployment_id(first[1])


def test_redact_api_keys() -> None:
    lst = ensure_deployment_ids([_dep("m1", "http://a", api_key="sk-abcdefghijklmnop")])
    redacted = redact_model_list_for_callback(lst)
    key = redacted[0]["litellm_params"]["api_key"]
    assert "abcdefghijklmnop" not in key
    assert key.startswith("sk-a")
    assert lst[0]["litellm_params"]["api_key"] == "sk-abcdefghijklmnop"


def test_validate_weights_rejects_all_zero() -> None:
    lst = ensure_deployment_ids(
        [
            _dep("m1", "http://a", weight=0),
            _dep("m1", "http://b", weight=0),
        ]
    )
    err = validate_weights_for_model_names(lst)
    assert err is not None
    assert "weight > 0" in err


def test_save_and_load_routing_state(tmp_path: Path) -> None:
    path = tmp_path / "routing_state.json"
    lst = ensure_deployment_ids([_dep("m1", "http://a")])
    save_routing_state(path, lst, updated_at="t1")
    loaded = load_routing_state(path)
    assert loaded is not None
    assert loaded["updated_at"] == "t1"
    assert len(loaded["model_list"]) == 1
    assert get_deployment_id(loaded["model_list"][0]) == get_deployment_id(lst[0])


# --- auth ---


@pytest.mark.asyncio
async def test_proxy_auth_skips_admin_paths() -> None:
    auth = AuthMiddleware("proxy-key")
    ok, err = await auth.authenticate("GET", "/admin/routing", {})
    assert ok is True
    assert err is None
    ok2, err2 = await auth.authenticate(
        "POST", "/v1/chat/completions", {"authorization": "Bearer admin-secret"}
    )
    assert ok2 is False
    assert err2 is not None


# --- admin CRUD ---


def test_admin_requires_admin_key(tmp_path: Path) -> None:
    proxy = FakeProxy([_dep("m1", "http://a")], state_path=tmp_path / "rs.json")
    client = TestClient(_admin_app(proxy))

    r = client.get("/admin/routing")
    assert r.status_code == 401

    r2 = client.get("/admin/routing", headers={"Authorization": "Bearer proxy-key"})
    assert r2.status_code == 401

    r3 = client.get("/admin/routing", headers=_admin_headers())
    assert r3.status_code == 200
    assert r3.json()["ok"] is True


def test_admin_disabled_without_key(tmp_path: Path) -> None:
    proxy = FakeProxy([_dep("m1", "http://a")], admin_api_key=None, state_path=tmp_path / "rs.json")
    client = TestClient(_admin_app(proxy))
    r = client.get("/admin/routing", headers=_admin_headers("anything"))
    assert r.status_code == 503


def test_create_and_delete_keeps_logical_model_name(tmp_path: Path) -> None:
    proxy = FakeProxy(
        [_dep("pcb_avi_sft_merge_v10", "http://old", weight=100, dep_id="old-id")],
        state_path=tmp_path / "rs.json",
    )
    client = TestClient(_admin_app(proxy))

    create = client.post(
        "/admin/routing/deployments",
        headers=_admin_headers(),
        json={
            "reason": "ab_test",
            "deployment": _dep("pcb_avi_sft_merge_v10", "http://new", weight=50),
        },
    )
    assert create.status_code == 200, create.text
    body = create.json()
    new_id = body["deployment"]["model_info"]["id"]
    assert len(proxy.model_list) == 2
    assert {e["model_name"] for e in proxy.model_list} == {"pcb_avi_sft_merge_v10"}

    # migrate weights then delete old — client model name unchanged
    patch = client.patch(
        "/admin/routing/weights",
        headers=_admin_headers(),
        json={
            "reason": "migration",
            "model_name": "pcb_avi_sft_merge_v10",
            "weights": [
                {"id": "old-id", "weight": 0},
                {"id": new_id, "weight": 100},
            ],
            "drain": False,
        },
    )
    assert patch.status_code == 200, patch.text

    delete = client.request(
        "DELETE",
        "/admin/routing/deployments/old-id",
        headers={**_admin_headers(), "Content-Type": "application/json"},
        content=json.dumps({"drain": False, "reason": "cutover"}),
    )
    assert delete.status_code == 200, delete.text
    assert len(proxy.model_list) == 1
    assert proxy.model_list[0]["model_name"] == "pcb_avi_sft_merge_v10"
    assert get_deployment_id(proxy.model_list[0]) == new_id
    assert find_deployment(proxy.model_list, "old-id")[1] is None


def test_put_full_replace(tmp_path: Path) -> None:
    proxy = FakeProxy(
        [_dep("m1", "http://a", dep_id="a")],
        state_path=tmp_path / "rs.json",
    )
    client = TestClient(_admin_app(proxy))
    r = client.put(
        "/admin/routing",
        headers=_admin_headers(),
        json={
            "reason": "cutover",
            "drain": False,
            "model_list": [_dep("m1", "http://b", dep_id="b")],
        },
    )
    assert r.status_code == 200
    assert get_deployment_id(proxy.model_list[0]) == "b"
    assert proxy.model_list[0]["litellm_params"]["api_base"] == "http://b"


def test_invalid_weights_does_not_reload(tmp_path: Path) -> None:
    proxy = FakeProxy(
        [
            _dep("m1", "http://a", weight=50, dep_id="a"),
            _dep("m1", "http://b", weight=50, dep_id="b"),
        ],
        state_path=tmp_path / "rs.json",
    )
    client = TestClient(_admin_app(proxy))
    before = deepcopy(proxy.model_list)
    r = client.patch(
        "/admin/routing/weights",
        headers=_admin_headers(),
        json={
            "model_name": "m1",
            "weights": [
                {"id": "a", "weight": 0},
                {"id": "b", "weight": 0},
            ],
            "drain": False,
        },
    )
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_weights"
    assert proxy.model_list == before
    assert proxy.reload_calls == []


# --- drain ---


@pytest.mark.asyncio
async def test_drain_waits_then_applies(tmp_path: Path) -> None:
    proxy = FakeProxy(
        [_dep("m1", "http://a", dep_id="a")],
        state_path=tmp_path / "rs.json",
    )
    await proxy.inflight.enter()

    async def release() -> None:
        await asyncio.sleep(0.05)
        await proxy.inflight.exit()

    asyncio.create_task(release())
    result = await proxy.apply_model_list(
        [_dep("m1", "http://b", dep_id="b")],
        reason="cutover",
        drain=True,
    )
    assert result["deployment_count"] == 1
    assert get_deployment_id(proxy.model_list[0]) == "b"


@pytest.mark.asyncio
async def test_drain_timeout_does_not_apply(tmp_path: Path) -> None:
    proxy = FakeProxy(
        [_dep("m1", "http://a", dep_id="a")],
        state_path=tmp_path / "rs.json",
    )
    await proxy.inflight.enter()
    with pytest.raises(RoutingApplyError) as exc:
        await proxy.apply_model_list(
            [_dep("m1", "http://b", dep_id="b")],
            reason="cutover",
            drain=True,
        )
    assert exc.value.code == "drain_timeout"
    assert get_deployment_id(proxy.model_list[0]) == "a"
    assert proxy.reload_calls == []


def test_delete_drain_timeout_via_api(tmp_path: Path) -> None:
    proxy = FakeProxy(
        [
            _dep("m1", "http://a", weight=50, dep_id="a"),
            _dep("m1", "http://b", weight=50, dep_id="b"),
        ],
        state_path=tmp_path / "rs.json",
    )
    # Simulate in-flight work visible to TestClient's event loop
    proxy.inflight._count = 1
    proxy.inflight._zero.clear()
    client = TestClient(_admin_app(proxy))

    r = client.request(
        "DELETE",
        "/admin/routing/deployments/a",
        headers={**_admin_headers(), "Content-Type": "application/json"},
        content=json.dumps({"drain": True}),
    )
    assert r.status_code == 409
    assert r.json()["code"] == "drain_timeout"
    assert len(proxy.model_list) == 2


# --- callbacks ---


@pytest.mark.asyncio
async def test_http_and_shell_callbacks_scheduled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "shell_ok"
    script = tmp_path / "cb.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        'python3 -c "import json,sys; json.load(open(sys.argv[1])); '
        f"open({str(marker)!r},'w').write('ok')\" \"$1\"\n"
    )
    script.chmod(0o755)

    http_calls: List[Any] = []

    class FakeResponse:
        status_code = 200
        text = "ok"

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, json: Any = None) -> FakeResponse:
            http_calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setenv("ROUTING_CHANGE_CALLBACK_URL", "http://callback.test/hook")
    monkeypatch.setenv("ROUTING_CHANGE_CALLBACK_SCRIPT", str(script))
    monkeypatch.setenv("ROUTING_CHANGE_CALLBACK_TIMEOUT", "2")

    with patch("openclaw_tracer.routing_callback.httpx.AsyncClient", FakeClient):
        proxy = FakeProxy(
            [_dep("m1", "http://a", dep_id="a")],
            state_path=tmp_path / "rs.json",
        )
        await proxy.apply_model_list(
            [_dep("m1", "http://b", dep_id="b")],
            reason="cutover",
            drain=False,
            op={"method": "PUT", "path": "/admin/routing"},
        )
        await asyncio.sleep(0.8)

    assert http_calls, "HTTP callback should have been invoked"
    payload = http_calls[0]["json"]
    assert payload["event"] == "routing.updated"
    assert payload["reason"] == "cutover"
    assert "sk-secret" not in json.dumps(payload)
    assert marker.exists(), "shell callback should have run"
    assert proxy.callback_payloads


@pytest.mark.asyncio
async def test_callback_skipped_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTING_CHANGE_CALLBACK_URL", raising=False)
    monkeypatch.delenv("ROUTING_CHANGE_CALLBACK_SCRIPT", raising=False)
    with patch("openclaw_tracer.routing_callback.asyncio.get_running_loop") as loop:
        schedule_routing_change_callback({"event": "routing.updated"})
        loop.assert_not_called()


# --- InflightTracker ---


@pytest.mark.asyncio
async def test_inflight_tracker_zero() -> None:
    t = InflightTracker()
    assert t.count == 0
    assert await t.wait_zero(0.1) is True
    await t.enter()
    assert t.count == 1
    assert await t.wait_zero(0.05) is False
    await t.exit()
    assert t.count == 0
    assert await t.wait_zero(0.1) is True


def test_inference_path_detection() -> None:
    assert InflightTracker.is_inference_path("/v1/chat/completions")
    assert InflightTracker.is_inference_path("/v1/embeddings")
    assert not InflightTracker.is_inference_path("/admin/routing")
    assert not InflightTracker.is_inference_path("/health")


@pytest.mark.asyncio
async def test_llm_proxy_reload_models_mocked_load_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLMProxy.reload_models persists state after mocked LiteLLM load_config."""
    from openclaw_tracer.proxy.llm_proxy import LLMProxy
    from openclaw_tracer.storage.parquet_store import ParquetStore

    monkeypatch.delenv("ROUTING_CHANGE_CALLBACK_URL", raising=False)
    monkeypatch.delenv("ROUTING_CHANGE_CALLBACK_SCRIPT", raising=False)

    store = ParquetStore(
        output_dir=tmp_path / "data",
        buffer_size=10,
        time_window_minutes=9999,
    )
    state_path = tmp_path / "data" / "routing_state.json"
    proxy = LLMProxy(
        port=0,
        model_list=[_dep("m1", "http://a", dep_id="a")],
        store=store,
        proxy_api_key="proxy-key",
        admin_api_key="admin-key",
        routing_state_path=state_path,
    )

    mock_router = MagicMock()
    mock_load = AsyncMock(return_value=(mock_router, [{"model_name": "m1"}], None))
    with patch("litellm.proxy.proxy_server.proxy_config.load_config", mock_load):
        with patch("litellm.proxy.proxy_server.llm_router", mock_router, create=True):
            with patch("litellm.proxy.proxy_server.llm_model_list", [], create=True):
                result = await proxy.reload_models(
                    [_dep("m1", "http://b", dep_id="b")],
                    reason="cutover",
                    schedule_callback=False,
                )

    assert result["deployment_count"] == 1
    assert get_deployment_id(proxy.model_list[0]) == "b"
    assert state_path.is_file()
    loaded = load_routing_state(state_path)
    assert loaded is not None
    assert get_deployment_id(loaded["model_list"][0]) == "b"
    mock_load.assert_awaited()
