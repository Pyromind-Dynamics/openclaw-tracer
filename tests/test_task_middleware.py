# Copyright (c) 2025 OpenClaw-Tracer
# Tests for TaskMiddleware

"""Tests for TaskMiddleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openclaw_tracer.middleware.task_middleware import TaskMiddleware
from openclaw_tracer.task_manager import TaskManager


@pytest.fixture
async def task_manager():
    """Create a TaskManager for testing."""
    mgr = TaskManager(timeout_minutes=1)
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest.fixture
def task_middleware(task_manager):
    """Create a TaskMiddleware for testing."""
    app = MagicMock()
    return TaskMiddleware(app, task_manager)


class TestTaskMiddleware:
    """Tests for TaskMiddleware."""

    @pytest.mark.asyncio
    async def test_extract_task_id_header(self, task_middleware, task_manager):
        """Test extracting X-Task-ID header."""
        # Mock request
        request = MagicMock()
        request.headers = {"x-task-id": "test-task-123"}
        request.state = {}

        # Mock call_next
        async def call_next(req):
            response = MagicMock()
            return response

        call_next_mock = AsyncMock(side_effect=call_next)

        # Process request
        await task_middleware.dispatch(request, call_next_mock)

        # Verify state was set
        assert request.state.get("task_id") == "test-task-123"
        assert request.state.get("attempt_id") == "at-000001"
        call_next_mock.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_extract_previous_reward_header(self, task_middleware, task_manager):
        """Test extracting X-Previous-Reward header."""
        request = MagicMock()
        request.headers = {"x-previous-reward": "0.85"}
        request.state = {}

        async def call_next(req):
            response = MagicMock()
            return response

        call_next_mock = AsyncMock(side_effect=call_next)

        await task_middleware.dispatch(request, call_next_mock)

        assert request.state.get("previous_reward") == 0.85

    @pytest.mark.asyncio
    async def test_extract_previous_reward_invalid(self, task_middleware, task_manager):
        """Test invalid reward value is handled gracefully."""
        request = MagicMock()
        request.headers = {"x-previous-reward": "invalid"}
        request.state = {}

        async def call_next(req):
            return MagicMock()

        call_next_mock = AsyncMock(side_effect=call_next)

        await task_middleware.dispatch(request, call_next_mock)

        # Should be None for invalid values
        assert request.state.get("previous_reward") is None

    @pytest.mark.asyncio
    async def test_case_insensitive_headers(self, task_middleware, task_manager):
        """Test headers are case-insensitive."""
        request = MagicMock()
        request.headers = {"X-Task-ID": "test-task", "X-Previous-Reward": "0.5"}
        request.state = {}

        async def call_next(req):
            return MagicMock()

        call_next_mock = AsyncMock(side_effect=call_next)

        await task_middleware.dispatch(request, call_next_mock)

        assert request.state.get("task_id") == "test-task"
        assert request.state.get("previous_reward") == 0.5

    @pytest.mark.asyncio
    async def test_no_headers_creates_auto_task(self, task_middleware, task_manager):
        """Test request without headers creates auto task."""
        request = MagicMock()
        request.headers = {}
        request.state = {}

        async def call_next(req):
            return MagicMock()

        call_next_mock = AsyncMock(side_effect=call_next)

        await task_middleware.dispatch(request, call_next_mock)

        # Should have auto-generated task_id and attempt_id
        assert request.state.get("task_id") is not None
        assert request.state.get("task_id").startswith("auto-")
        assert request.state.get("attempt_id") == "at-000001"
        assert request.state.get("previous_reward") is None

    @pytest.mark.asyncio
    async def test_same_task_increments_attempt(self, task_middleware, task_manager):
        """Test same task_id increments attempt on subsequent requests."""
        # First request
        request1 = MagicMock()
        request1.headers = {"x-task-id": "repeat-task"}
        request1.state = {}

        async def call_next1(req):
            return MagicMock()

        await task_middleware.dispatch(request1, AsyncMock(side_effect=call_next1))
        assert request1.state.get("attempt_id") == "at-000001"

        # Second request with same task_id
        request2 = MagicMock()
        request2.headers = {"x-task-id": "repeat-task"}
        request2.state = {}

        async def call_next2(req):
            return MagicMock()

        await task_middleware.dispatch(request2, AsyncMock(side_effect=call_next2))
        assert request2.state.get("attempt_id") == "at-000002"

    @pytest.mark.asyncio
    async def test_session_id_lowercase_alias(self, task_middleware, task_manager):
        """``x-session-id`` alone resolves the same as ``X-Task-ID``."""
        request = MagicMock()
        request.headers = {"x-session-id": "session-lower"}
        request.state = {}

        await task_middleware.dispatch(request, AsyncMock(return_value=MagicMock()))

        assert request.state.get("task_id") == "session-lower"
        assert request.state.get("attempt_id") == "at-000001"

    @pytest.mark.asyncio
    async def test_session_id_mixed_case_alias(self, task_middleware, task_manager):
        """``X-Session-Id`` alone resolves the same as ``X-Task-ID``."""
        request = MagicMock()
        request.headers = {"X-Session-Id": "session-mixed"}
        request.state = {}

        await task_middleware.dispatch(request, AsyncMock(return_value=MagicMock()))

        assert request.state.get("task_id") == "session-mixed"
        assert request.state.get("attempt_id") == "at-000001"

    @pytest.mark.asyncio
    async def test_task_id_wins_over_session_id(self, task_middleware, task_manager):
        """When both headers are set, ``X-Task-ID`` is canonical."""
        request = MagicMock()
        request.headers = {
            "X-Task-ID": "from-task",
            "X-Session-Id": "from-session",
        }
        request.state = {}

        await task_middleware.dispatch(request, AsyncMock(return_value=MagicMock()))

        assert request.state.get("task_id") == "from-task"
