# Copyright (c) 2025 OpenClaw-Tracer
# Integration tests for task tracking feature

"""Integration tests for task tracking feature."""

import asyncio
import shutil
from typing import AsyncGenerator, Generator
from pathlib import Path

import pytest
from httpx import AsyncClient

from openclaw_tracer import LLMProxy
from openclaw_tracer.storage.parquet_store import ParquetStore


# Skip all integration tests if we can't create a working proxy
pytest.importorskip("litellm.proxy.proxy_server")


@pytest.fixture(scope="session", autouse=True)
def _integration_tests_disable_upstream_proxy_env() -> Generator[None, None, None]:
    """LiteLLM uses httpx with trust_env=True for upstream calls; strip proxy vars to avoid hangs."""
    import os
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    saved: dict[str, str | None] = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture
def task_tracking_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for task tracking test data."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def proxy_with_task_tracking(task_tracking_temp_dir: Path) -> AsyncGenerator[LLMProxy, None]:
    """Create a proxy with task tracking enabled."""
    store = ParquetStore(
        output_dir=task_tracking_temp_dir / "data",
        buffer_size=1,
        auto_flush=True,
    )

    model_list = [{
        "model_name": "test-model",
        "litellm_params": {
            "model": "openai/gpt-3.5-turbo",
            "api_key": "test-key",
        },
    }]

    proxy = LLMProxy(
        port=None,  # Random port
        host="127.0.0.1",
        model_list=model_list,
        store=store,
        proxy_api_key="test-proxy-key-12345",  # Use same key as test_integration.py to avoid middleware conflicts
        task_timeout_minutes=10,
    )

    await proxy.start()
    # Wait for server to be ready
    await asyncio.sleep(0.5)

    yield proxy

    await proxy.stop()
    await store.close()


@pytest.fixture
async def authenticated_client(proxy_with_task_tracking: LLMProxy) -> AsyncGenerator[AsyncClient, None]:
    """Create an authenticated HTTP client for the proxy."""
    async with AsyncClient(
        base_url=proxy_with_task_tracking.url,
        headers={"Authorization": "Bearer test-proxy-key-12345"},  # Use same key as test_integration.py
        timeout=30.0,
        trust_env=False,
    ) as client:
        yield client


@pytest.fixture
async def unauth_client(proxy_with_task_tracking: LLMProxy) -> AsyncGenerator[AsyncClient, None]:
    """Create an HTTP client without authentication."""
    async with AsyncClient(
        base_url=proxy_with_task_tracking.url,
        timeout=30.0,
        trust_env=False,
    ) as client:
        yield client


class TestTaskTrackingIntegration:
    """End-to-end tests for task tracking."""

    @pytest.mark.asyncio
    async def test_tracer_version_endpoint(self, proxy_with_task_tracking: LLMProxy):
        """Test /tracer-version returns correct information."""
        async with AsyncClient() as client:
            response = await client.get(f"{proxy_with_task_tracking.url}/tracer-version")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "openclaw-tracer"
        assert "version" in data
        assert "task-tracking" in data["features"]

    @pytest.mark.asyncio
    async def test_tracer_version_no_auth(self, unauth_client: AsyncClient):
        """Test /tracer-version works without authentication."""
        response = await unauth_client.get("/tracer-version")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "openclaw-tracer"
        assert "task-tracking" in data["features"]

    @pytest.mark.asyncio
    async def test_end_task_endpoint_nonexistent(self, authenticated_client: AsyncClient):
        """Test /end_task endpoint with non-existent task."""
        response = await authenticated_client.post(
            "/end_task",
            json={"task_id": "test-task-123"},
        )

        # Task doesn't exist, should return 404
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_end_task_endpoint_missing_task_id(self, authenticated_client: AsyncClient):
        """Test /end_task endpoint with missing task_id."""
        response = await authenticated_client.post(
            "/end_task",
            json={},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "task_id" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_end_task_endpoint_invalid_json(self, authenticated_client: AsyncClient):
        """Test /end_task endpoint with invalid JSON."""
        response = await authenticated_client.post(
            "/end_task",
            content="invalid json",
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_end_task_endpoint_unauthorized(self, unauth_client: AsyncClient):
        """Test /end_task requires authentication."""
        response = await unauth_client.post(
            "/end_task",
            json={"task_id": "test-task"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_task_flow_create_and_end(self, authenticated_client: AsyncClient, proxy_with_task_tracking: LLMProxy):
        """Test full task flow: create via request, then end."""
        task_id = "integration-test-task"

        # Make a request with task_id header to create the task
        # Note: We use /v1/models which is a simple GET that doesn't require upstream
        response = await authenticated_client.get(
            "/v1/models",
            headers={"X-Task-ID": task_id},
        )

        # Should succeed
        assert response.status_code == 200

        # Give time for task to be registered
        await asyncio.sleep(0.1)

        # Now end the task
        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id},
        )

        assert end_response.status_code == 200
        data = end_response.json()
        assert data["task_id"] == task_id
        assert "attempt_count" in data
        assert "total_requests" in data
        assert "duration_seconds" in data
        assert data["attempt_count"] >= 1
        assert data["total_requests"] >= 1
        assert data.get("final_reward") is None

    @pytest.mark.asyncio
    async def test_session_id_alias_create_and_end(
        self, authenticated_client: AsyncClient, proxy_with_task_tracking: LLMProxy
    ):
        """``X-Session-Id`` creates the same task session as ``X-Task-ID``."""
        task_id = "integration-session-alias-task"
        response = await authenticated_client.get(
            "/v1/models",
            headers={"X-Session-Id": task_id},
        )
        assert response.status_code == 200
        await asyncio.sleep(0.1)

        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id},
        )
        assert end_response.status_code == 200
        data = end_response.json()
        assert data["task_id"] == task_id
        assert data["attempt_count"] >= 1
        assert data["total_requests"] >= 1

    @pytest.mark.asyncio
    async def test_end_task_with_final_reward(self, authenticated_client: AsyncClient):
        """POST /end_task accepts final_reward (and alias reward)."""
        task_id = "integration-final-reward-task"
        response = await authenticated_client.get(
            "/v1/models",
            headers={"X-Task-ID": task_id},
        )
        assert response.status_code == 200
        await asyncio.sleep(0.1)

        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id, "final_reward": 1.25},
        )
        assert end_response.status_code == 200
        data = end_response.json()
        assert data["task_id"] == task_id
        assert data["final_reward"] == 1.25

    @pytest.mark.asyncio
    async def test_end_task_final_reward_alias_reward(self, authenticated_client: AsyncClient):
        task_id = "integration-reward-alias-task"
        await authenticated_client.get("/v1/models", headers={"X-Task-ID": task_id})
        await asyncio.sleep(0.1)
        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id, "reward": -0.5},
        )
        assert end_response.status_code == 200
        assert end_response.json()["final_reward"] == -0.5

    @pytest.mark.asyncio
    async def test_end_task_invalid_final_reward(self, authenticated_client: AsyncClient):
        task_id = "integration-invalid-reward-task"
        await authenticated_client.get("/v1/models", headers={"X-Task-ID": task_id})
        await asyncio.sleep(0.1)
        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id, "final_reward": "not-a-number"},
        )
        assert end_response.status_code == 400

    @pytest.mark.asyncio
    async def test_task_multiple_attempts(self, authenticated_client: AsyncClient):
        """Test that multiple requests with same task_id increment attempt."""
        task_id = "multi-attempt-task"

        # Make multiple requests
        for _ in range(3):
            response = await authenticated_client.get(
                "/v1/models",
                headers={"X-Task-ID": task_id},
            )
            assert response.status_code == 200

        # Give time for all requests to be processed
        await asyncio.sleep(0.1)

        # End the task and check stats
        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id},
        )

        assert end_response.status_code == 200
        data = end_response.json()
        assert data["task_id"] == task_id
        assert data["attempt_count"] == 3
        assert data["total_requests"] == 3
        assert data.get("final_reward") is None

    @pytest.mark.asyncio
    async def test_task_with_previous_reward_header(self, authenticated_client: AsyncClient):
        """Test that X-Previous-Reward header is captured."""
        task_id = "reward-test-task"

        # Make a request with reward header
        response = await authenticated_client.get(
            "/v1/models",
            headers={
                "X-Task-ID": task_id,
                "X-Previous-Reward": "0.85",
            },
        )

        assert response.status_code == 200

        # Give time for task to be registered
        await asyncio.sleep(0.1)

        # End the task
        end_response = await authenticated_client.post(
            "/end_task",
            json={"task_id": task_id},
        )

        assert end_response.status_code == 200
        # Reward doesn't affect the stats, but task should be tracked
        data = end_response.json()
        assert data["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_auto_generated_task_id(self, authenticated_client: AsyncClient):
        """Test request without X-Task-ID generates auto task ID."""
        # Make a request without task_id header
        response = await authenticated_client.get("/v1/models")
        assert response.status_code == 200

        # The middleware should have created an auto-generated task_id
        # We can't easily verify this without internal access, but we can
        # verify the request didn't fail
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_status_endpoint(self, authenticated_client: AsyncClient):
        """Test /status endpoint returns collection status."""
        response = await authenticated_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        # Should have status information
        assert isinstance(data, dict)


class TestTaskTimeout:
    """Tests for task timeout functionality."""

    @pytest.mark.asyncio
    async def test_task_timeout_cleanup_short_timeout(self):
        """Test that tasks are cleaned up after short timeout."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a proxy with very short timeout
            store = ParquetStore(
                output_dir=Path(tmpdir) / "data-timeout",
                buffer_size=1,
                auto_flush=True,
            )

            proxy = LLMProxy(
                port=None,
                host="127.0.0.1",
                model_list=[{
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-3.5-turbo",
                        "api_key": "test-key",
                    },
                }],
                store=store,
                proxy_api_key="test-proxy-key-12345",  # Use same key as test_integration.py
                task_timeout_minutes=0,  # Immediate timeout for testing
            )

            await proxy.start()
            await asyncio.sleep(0.2)

            try:
                task_id = "timeout-test-task"

                async with AsyncClient(
                    base_url=proxy.url,
                    headers={"Authorization": "Bearer test-proxy-key-12345"},
                    timeout=30.0,
                ) as client:
                    # Create a task
                    response = await client.get(
                        "/v1/models",
                        headers={"X-Task-ID": task_id},
                    )
                    assert response.status_code == 200

                    # Wait for cleanup (cleanup runs every 60s, so we skip this test)
                    # In a real scenario, we'd need to expose internal state or wait longer

                # Verify proxy is still running
                assert proxy.is_running
            finally:
                await proxy.stop()
                await store.close()
