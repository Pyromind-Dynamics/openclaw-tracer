# Copyright (c) 2025 OpenClaw-Tracer
# LiteLLM proxy server for data collection

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import app, save_worker_config
try:
    from portpicker import pick_unused_port
except ImportError:
    # Fallback to simple port selection
    def pick_unused_port():
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

from openclaw_tracer.proxy.http_proxy_env import active_upstream_proxy_summary, apply_upstream_proxy_env
from openclaw_tracer.proxy.image_externalize import (
    externalize_inline_images_for_span,
    redact_data_url_images_in_json_body_for_log,
)
from openclaw_tracer.storage.base import StorageBackend
from openclaw_tracer.storage.parquet_store import ParquetStore
from openclaw_tracer.task_manager import TaskManager
from openclaw_tracer.types.core import (
    Attributes,
    Resource,
    Span,
    SpanContext,
    SpanKind,
    StatusCode,
)

logger = logging.getLogger(__name__)

# Context variables for passing task information through LiteLLM callbacks
task_context: contextvars.ContextVar[dict] = contextvars.ContextVar("task_context")

# Setup diagnostic logger for response debugging
diagnostic_logger = logging.getLogger("diagnostic")
diagnostic_logger.setLevel(logging.INFO)
if not diagnostic_logger.handlers:
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Add file handler for diagnostic logs
    diagnostic_handler = logging.FileHandler(log_dir / "diagnostic.log")
    diagnostic_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    diagnostic_handler.setFormatter(formatter)
    diagnostic_logger.addHandler(diagnostic_handler)


class HTTPAccessLogger:
    """Middleware to log all HTTP requests and responses to a file.

    Logs include:
    - Timestamp
    - Request method, path, headers, body
    - Response status, headers, body
    - Processing duration
    """

    def __init__(self, log_file: str | None = None):
        """Initialize the HTTP access logger.

        Args:
            log_file: Path to the log file. If None, logs to stdout.
        """
        self.log_file = Path(log_file) if log_file else None
        self._lock = asyncio.Lock()
        self._request_contexts: Dict[str, Dict[str, Any]] = {}

        # Create log file directory if needed
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def log_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: str | None = None,
    ) -> str:
        """Log an incoming HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            headers: Request headers
            body: Request body (if any)

        Returns:
            Request ID for correlating with response
        """
        request_id = uuid4().hex[:16]
        timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "type": "request",
            "request_id": request_id,
            "timestamp": timestamp,
            "method": method,
            "path": path,
            "headers": dict(headers) if headers else {},
            "body": body,
        }

        # Store context for response correlation
        self._request_contexts[request_id] = {
            "start_time": timestamp,
            "method": method,
            "path": path,
        }

        await self._write_log(log_entry)
        return request_id

    async def log_response(
        self,
        request_id: str,
        status_code: int,
        headers: Dict[str, str],
        body: str | None = None,
        error: str | None = None,
    ) -> None:
        """Log an HTTP response.

        Args:
            request_id: Correlation ID from request
            status_code: HTTP status code
            headers: Response headers
            body: Response body (if any)
            error: Error message (if request failed)
        """
        timestamp = datetime.utcnow().isoformat()
        context = self._request_contexts.get(request_id, {})

        log_entry = {
            "type": "response",
            "request_id": request_id,
            "timestamp": timestamp,
            "status_code": status_code,
            "headers": dict(headers) if headers else {},
            "body": body,
            "error": error,
            "request_method": context.get("method"),
            "request_path": context.get("path"),
            "duration_ms": self._calculate_duration(
                str(context.get("start_time")), timestamp
            ) if context.get("start_time") else None,
        }

        await self._write_log(log_entry)

        # Clean up context
        self._request_contexts.pop(request_id, None)

    async def log_error(
        self,
        method: str,
        path: str,
        error: str,
        headers: Dict[str, str] | None = None,
    ) -> None:
        """Log an error without a corresponding request.

        Args:
            method: HTTP method
            path: Request path
            error: Error message
            headers: Request headers (if available)
        """
        timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "type": "error",
            "timestamp": timestamp,
            "method": method,
            "path": path,
            "error": error,
            "headers": dict(headers) if headers else {},
        }

        await self._write_log(log_entry)

    async def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to file or stdout.

        Args:
            entry: Log entry dictionary
        """
        log_line = json.dumps(entry, ensure_ascii=False) + "\n"

        async with self._lock:
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line)
            else:
                print(log_line, end="")

    @staticmethod
    def _calculate_duration(start: str, end: str) -> float | None:
        """Calculate duration between two ISO timestamps.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            Duration in milliseconds
        """
        try:
            from datetime import timezone
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
            return (end_dt - start_dt).total_seconds() * 1000
        except Exception:
            return None


class AuthMiddleware:
    """Middleware to authenticate requests using API Key.

    Supports two authentication methods:
    1. Authorization: Bearer <API_KEY>
    2. X-API-Key: <API_KEY>

    Authentication failures return 401 Unauthorized with JSON error response.
    """

    # Paths that don't require PROXY_API_KEY authentication
    PUBLIC_PATHS = {"/health", "/status", "/v1/models", "/tracer-version"}

    def __init__(self, api_key: str):
        """Initialize the authentication middleware.

        Args:
            api_key: The API key to validate against.
        """
        if not api_key:
            raise ValueError("PROXY_API_KEY must be set and non-empty")
        self.api_key = api_key

    @staticmethod
    def _is_admin_path(path: str) -> bool:
        return path == "/admin" or path.startswith("/admin/")

    def _extract_api_key(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract API key from request headers.

        Args:
            headers: Request headers dictionary

        Returns:
            The extracted API key, or None if not found.
        """
        # Try Authorization: Bearer <token> header
        auth_header = headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()  # Remove "Bearer " prefix

        # Try X-API-Key header (case-insensitive)
        for key, value in headers.items():
            if key.lower() == "x-api-key":
                return value.strip()

        return None

    def _create_unauthorized_response(self) -> Dict[str, Any]:
        """Create the standard 401 Unauthorized error response.

        Returns:
            Dictionary with error details.
        """
        return {
            "error": {
                "message": "Unauthorized",
                "type": "authentication_error",
                "code": "invalid_api_key"
            }
        }

    async def authenticate(
        self,
        method: str,
        path: str,
        headers: Dict[str, str]
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate a request.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers

        Returns:
            Tuple of (is_authenticated, error_response)
            - If authenticated: (True, None)
            - If not authenticated: (False, error_response_dict)
        """
        # Skip PROXY auth for public paths and admin routes (admin uses ADMIN_API_KEY)
        if path in self.PUBLIC_PATHS or self._is_admin_path(path):
            return True, None

        # Extract API key from headers
        provided_key = self._extract_api_key(headers)

        if provided_key is None:
            # No API key provided
            return False, self._create_unauthorized_response()

        if provided_key != self.api_key:
            # Invalid API key
            logger.warning(f"[Auth] Invalid API key attempt from path: {path}")
            return False, self._create_unauthorized_response()

        # Authentication successful
        return True, None


class InflightTracker:
    """Count in-flight inference requests for routing drain."""

    INFERENCE_PREFIXES = (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/chat/completions",
        "/completions",
    )

    def __init__(self) -> None:
        self._count = 0
        self._lock = asyncio.Lock()
        self._zero = asyncio.Event()
        self._zero.set()

    @property
    def count(self) -> int:
        return self._count

    @classmethod
    def is_inference_path(cls, path: str) -> bool:
        return any(path == p or path.startswith(p + "/") for p in cls.INFERENCE_PREFIXES) or path in {
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
        }

    async def enter(self) -> None:
        async with self._lock:
            self._count += 1
            self._zero.clear()

    async def exit(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)
            if self._count == 0:
                self._zero.set()

    async def wait_zero(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._zero.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


def _to_timestamp(value: Union[datetime, float, int, None]) -> Optional[float]:
    """Convert datetime to Unix timestamp.

    Args:
        value: datetime object, timestamp (float/int), or None

    Returns:
        Unix timestamp as float, or None if input is None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


def _get_pre_call_data(args: Any, kwargs: Any) -> Dict[str, Any]:
    """Extract LiteLLM request payload from hook args.

    The LiteLLM logger hooks receive `(*args, **kwargs)` whose third positional
    argument or `data=` kwarg contains the request payload.

    Args:
        args: Positional arguments from the hook.
        kwargs: Keyword arguments from the hook.

    Returns:
        The request payload dict, or empty dict if not found.
    """
    try:
        if kwargs.get("data"):
            data = kwargs["data"]
        elif len(args) >= 3:
            data = args[2]
        else:
            # Return empty dict if we can't find data
            return {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # Return empty dict on any error
        return {}


def _extract_litellm_request(kwargs: Any) -> tuple[Dict[str, Any], List[Any]]:
    """Extract request payload dict and messages list from LiteLLM logger kwargs."""
    data: Dict[str, Any] = {}
    messages: List[Any] = []
    if not isinstance(kwargs, dict):
        return data, messages
    if "messages" in kwargs:
        messages = kwargs["messages"]
        data = kwargs
        logger.debug(f"Found messages directly in kwargs, count: {len(messages)}")
    elif "litellm_params" in kwargs:
        litellm_params = kwargs["litellm_params"]
        messages = litellm_params.get("messages", [])
        data = litellm_params if isinstance(litellm_params, dict) else {}
        logger.debug(f"Found messages in litellm_params, count: {len(messages)}")
    elif "data" in kwargs and isinstance(kwargs["data"], dict):
        messages = kwargs["data"].get("messages", [])
        data = kwargs["data"]
        logger.debug(f"Found messages in data, count: {len(messages)}")
    else:
        data = _get_pre_call_data(kwargs, kwargs)
        messages = data.get("messages", [])
        logger.debug(f"Found messages via _get_pre_call_data, count: {len(messages)}")
    return data, messages


def _split_system_from_messages(messages: List[Any]) -> tuple[Any, List[Any]]:
    """Strip leading system message for separate span attribute storage."""
    messages_for_storage = list(messages)
    system_message: Any = None
    if (
        messages_for_storage
        and isinstance(messages_for_storage[0], dict)
        and messages_for_storage[0].get("role") == "system"
    ):
        system_message = messages_for_storage[0].get("content")
        messages_for_storage = messages_for_storage[1:]
    return system_message, messages_for_storage


class RequestSanitizer(CustomLogger):
    """LiteLLM callback to sanitize requests before sending to upstream API.

    Removes parameters that are not supported by certain backends (e.g., vLLM, Gemini OpenAI compatibility).

    Can be configured to only sanitize specific models by name prefix.
    """

    # Parameters to remove from requests
    SANITIZED_PARAMS = {"tool_choice", "tools", "tool_use", "parallel_tool_calls"}

    callback_name: str

    def __init__(self, sanitize_models: Optional[Set[str]] = None):
        """Initialize the request sanitizer.

        Args:
            sanitize_models: Set of model name prefixes that should have their requests sanitized.
                           If None, all models are sanitized.
                           Example: {"gemini", "gpt"} will sanitize "gemini-pro" and "gpt-4" but not "claude-3".
        """
        super().__init__()
        self.sanitize_models = sanitize_models

    def _should_sanitize(self, model: str) -> bool:
        """Check if a model should have its requests sanitized.

        Args:
            model: The model name/identifier.

        Returns:
            True if the model should be sanitized, False otherwise.
        """
        if self.sanitize_models is None:
            # Sanitize all models
            return True
        # Check if model name starts with any of the prefixes
        return any(model.startswith(prefix) for prefix in self.sanitize_models)

    async def async_log_pre_api_call(self, kwargs, **_):
        """Called before the API request is made.

        Removes unsupported parameters from the request for configured models.

        Args:
            kwargs: The request kwargs that will be sent to the API
        """
        # Debug: log the structure of kwargs
        logger.debug(f"[RequestSanitizer] Called with kwargs keys: {kwargs.keys() if isinstance(kwargs, dict) else 'not a dict'}")
        logger.debug(f"[RequestSanitizer] sanitize_models: {self.sanitize_models}")

        # Check if we should sanitize this model
        # Try multiple ways to get the model name
        model = kwargs.get("model", "")
        logger.debug(f"[RequestSanitizer] Initial model from kwargs['model']: '{model}'")

        # If model is empty, try other locations
        if not model and isinstance(kwargs, dict):
            litellm_params = kwargs.get("litellm_params", {})
            model = litellm_params.get("model", "")
            logger.debug(f"[RequestSanitizer] Model from litellm_params: '{model}'")

        should_sanitize = self._should_sanitize(model)
        logger.debug(f"[RequestSanitizer] Should sanitize '{model}': {should_sanitize}")

        if not should_sanitize:
            return kwargs

        # Sanitize the data payload
        if isinstance(kwargs, dict):
            # Check for data in kwargs
            data = kwargs.get("data")
            logger.debug(f"[RequestSanitizer] data in kwargs: {data is not None}, type: {type(data)}")

            if isinstance(data, dict):
                logger.debug(f"[RequestSanitizer] data keys: {data.keys()}")
                logger.debug(f"[RequestSanitizer] data.get('tools'): {data.get('tools')}")

                modified = False
                original_data = data.copy()
                for param in self.SANITIZED_PARAMS:
                    if param in data:
                        del data[param]
                        modified = True
                        logger.info(f"[RequestSanitizer] Removed '{param}' from request")

                if modified:
                    logger.info(
                        f"Sanitized request for '{model}': removed {self.SANITIZED_PARAMS & set(original_data.keys())} from request"
                    )

        return kwargs


class SpanLogger(CustomLogger):
    """LiteLLM logger hook to capture LLM calls as spans.

    This class integrates with LiteLLM's callback system to capture:
    - Request data (prompts, messages, parameters)
    - Response data (completions, token IDs, usage)
    - Timing information
    - Error information

    All captured data is forwarded to a StorageBackend for persistence.
    """

    callback_name: str

    def __init__(self, store: StorageBackend):
        """Initialize the span logger.

        Args:
            store: Storage backend for persisting captured data.
        """
        super().__init__()
        self.store = store
        self._current_spans: Dict[str, Dict[str, Any]] = {}
        self._media_output_dir: Optional[Path] = None
        output_dir = getattr(store, "output_dir", None)
        if output_dir is not None:
            self._media_output_dir = Path(output_dir)

    async def _maybe_externalize_images(
        self,
        trace_id: str,
        span_id: str,
        messages_for_storage: List[Any],
        system_message: Any,
    ) -> tuple[List[Any], Any]:
        """Replace data-URL images with on-disk JPEG paths relative to the store output dir."""
        if self._media_output_dir is None:
            return messages_for_storage, system_message
        media_prefix = "spans/media/images"
        prepare = getattr(self.store, "prepare_media_images_prefix", None)
        if callable(prepare):
            media_prefix = prepare()
        return await asyncio.to_thread(
            externalize_inline_images_for_span,
            self._media_output_dir,
            trace_id,
            span_id,
            messages_for_storage,
            system_message,
            media_prefix,
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called when an LLM request succeeds.

        Captures complete conversation data including:
        - System prompt (separately identified)
        - Full message history (multi-turn conversations)
        - Tool calls and tool results (separately identified)
        - Response content
        - Usage statistics
        """
        try:
            # Get task context from contextvars
            ctx = task_context.get() or {}
            if ctx.get("no_trace"):
                logger.debug("Skipping span capture due to X-Task-No-Trace")
                return
            task_id = ctx.get("task_id")
            attempt_id = ctx.get("attempt_id")
            previous_reward = ctx.get("previous_reward")
            # Debug: log what we received
            logger.debug(f"async_log_success_event kwargs type: {type(kwargs)}")
            logger.debug(f"async_log_success_event kwargs keys: {kwargs.keys() if isinstance(kwargs, dict) else 'not a dict'}")
            logger.debug(f"async_log_success_event response_obj type: {type(response_obj)}")

            data, messages = _extract_litellm_request(kwargs)
            logger.debug(f"Extracted messages: {len(messages)} messages")

            system_message, messages_for_storage = _split_system_from_messages(messages)
            if system_message is not None:
                sys_dbg = (
                    len(system_message)
                    if isinstance(system_message, str)
                    else "multimodal"
                )
                logger.debug(f"Extracted system prompt: {sys_dbg}")

            # Extract tool results from conversation (separately identified)
            tool_results = None
            tool_messages = [m for m in messages if m.get("role") == "tool"]
            if tool_messages:
                tool_results = [
                    {
                        "tool_call_id": m.get("tool_call_id"),
                        "content": m.get("content"),
                    }
                    for m in tool_messages
                ]
                logger.debug(f"Found {len(tool_results)} tool results")

            # Basic info
            model = kwargs.get("model", data.get("model", "unknown"))
            span_id = uuid4().hex[:16]
            trace_id = uuid4().hex[:32]

            messages_for_storage, system_message = await self._maybe_externalize_images(
                trace_id, span_id, messages_for_storage, system_message
            )

            # Build enhanced attributes - only add non-None values
            attributes: Attributes = {
                "llm.model": model,
                # Full messages array (without system, already saved separately)
                "llm.request.messages": json.dumps(messages_for_storage, ensure_ascii=False),
            }

            # System prompt (separately identified) - only add if present
            if system_message is not None:
                attributes["llm.request.system"] = system_message

            # Request parameters - only add if present
            temperature = data.get("temperature")
            if temperature is not None:
                attributes["llm.request.temperature"] = temperature

            max_tokens = data.get("max_tokens")
            if max_tokens is not None:
                attributes["llm.request.max_tokens"] = max_tokens

            top_p = data.get("top_p")
            if top_p is not None:
                attributes["llm.request.top_p"] = top_p

            # Save tools definition if present
            if data.get("tools"):
                attributes["llm.request.tools"] = json.dumps(data["tools"], ensure_ascii=False)

            # Process response - handle both object and dict formats
            content = None
            if response_obj:
                diagnostic_logger.info(f"[RESPONSE_DEBUG] Response object type: {type(response_obj)}")
                diagnostic_logger.info(f"[RESPONSE_DEBUG] Response object dir: {[x for x in dir(response_obj) if not x.startswith('_')]}")

                # Try to get the dict representation for objects
                if hasattr(response_obj, '__dict__'):
                    diagnostic_logger.info(f"[RESPONSE_DEBUG] Response __dict__: {response_obj.__dict__}")
                elif hasattr(response_obj, 'model_dump'):
                    diagnostic_logger.info(f"[RESPONSE_DEBUG] Response model_dump: {response_obj.model_dump()}")
                elif isinstance(response_obj, dict):
                    diagnostic_logger.info(f"[RESPONSE_DEBUG] Response dict keys: {list(response_obj.keys())}")
                    diagnostic_logger.info(f"[RESPONSE_DEBUG] Response dict: {response_obj}")

                # Handle dict-style response (some providers return this)
                if isinstance(response_obj, dict):
                    logger.debug(f"Response is dict, keys: {list(response_obj.keys())}")

                    # Try OpenAI-style format first (with choices)
                    choices = response_obj.get("choices", [])
                    if choices:
                        choice = choices[0]
                        message = choice.get("message", {})
                        content = message.get("content")
                        if content:
                            attributes["llm.response.content"] = content
                            logger.debug(f"Got content from dict (OpenAI-style): {len(content)} chars")

                        # Check for tool_calls
                        tool_calls = message.get("tool_calls")
                        if tool_calls:
                            attributes["llm.response.tool_calls"] = json.dumps(tool_calls, ensure_ascii=False)
                            logger.debug(f"Found tool_calls in dict response")

                    # Try Anthropic-style format (direct content field)
                    elif "content" in response_obj:
                        content_array = response_obj.get("content", [])
                        if content_array:
                            # Anthropic returns content as an array of blocks
                            # Extract text from text blocks
                            text_parts = []
                            tool_use_parts = []

                            for block in content_array:
                                if isinstance(block, dict):
                                    block_type = block.get("type")
                                    if block_type == "text":
                                        text_parts.append(block.get("text", ""))
                                    elif block_type == "tool_use":
                                        tool_use_parts.append({
                                            "id": block.get("id"),
                                            "name": block.get("name"),
                                            "input": block.get("input"),
                                        })

                            if text_parts:
                                attributes["llm.response.content"] = "".join(text_parts)
                                logger.debug(f"Got content from dict (Anthropic-style): {len(attributes['llm.response.content'])} chars")

                            if tool_use_parts:
                                attributes["llm.response.tool_calls"] = json.dumps(tool_use_parts, ensure_ascii=False)
                                logger.debug(f"Found tool_use blocks in Anthropic response")

                    # Usage from dict
                    usage = response_obj.get("usage")
                    if usage:
                        attributes["llm.usage.prompt_tokens"] = usage.get("prompt_tokens", 0)
                        attributes["llm.usage.completion_tokens"] = usage.get("completion_tokens", 0)
                        attributes["llm.usage.total_tokens"] = usage.get("total_tokens", 0)
                        logger.debug(f"Got usage from dict: {usage}")

                # Handle object-style response
                else:
                    # Check for Claude Responses API format first (has 'output' field, no 'choices')
                    if hasattr(response_obj, 'output') and getattr(response_obj, 'output'):
                        output = getattr(response_obj, 'output')
                        diagnostic_logger.info(f"[RESPONSE_DEBUG] Detected Responses API format, output type: {type(output)}")

                        # Responses API output is a list of items (reasoning, messages, etc.)
                        if isinstance(output, list):
                            text_parts = []
                            reasoning_parts = []
                            tool_use_parts = []

                            for item in output:
                                # Get item type
                                item_type = getattr(item, 'type', None) or (item.__class__.__name__ if hasattr(item, '__class__') else '')

                                diagnostic_logger.info(f"[RESPONSE_DEBUG] Output item type: {item_type}")

                                # Handle reasoning items
                                if item_type and 'reasoning' in item_type.lower():
                                    # Extract reasoning content
                                    content = getattr(item, 'content', [])
                                    if content:
                                        for c in content:
                                            if hasattr(c, 'text'):
                                                reasoning_parts.append(c.text)

                                # Handle message items
                                elif item_type and 'message' in item_type.lower():
                                    # Extract message content
                                    content_list = getattr(item, 'content', [])
                                    if content_list:
                                        for c in content_list:
                                            c_type = getattr(c, 'type', None)
                                            if c_type == 'output_text':
                                                text = getattr(c, 'text', None)
                                                if text:
                                                    text_parts.append(text)
                                            elif c_type == 'tool_use':
                                                # Tool use in response
                                                tool_use_parts.append({
                                                    'id': getattr(c, 'id', None),
                                                    'name': getattr(c, 'name', None),
                                                    'input': getattr(c, 'input', None),
                                                })

                            # Save reasoning if present
                            if reasoning_parts:
                                attributes['llm.response.reasoning'] = ''.join(reasoning_parts)

                            # Save text content
                            if text_parts:
                                attributes["llm.response.content"] = ''.join(text_parts)
                                diagnostic_logger.info(f"[RESPONSE_DEBUG] SUCCESS: Got content from Responses API ({len(attributes['llm.response.content'])} chars)")
                            else:
                                diagnostic_logger.warning(f"[RESPONSE_DEBUG] FAILED: No text content found in Responses API output")

                            # Save tool calls if present
                            if tool_use_parts:
                                attributes["llm.response.tool_calls"] = json.dumps(tool_use_parts, ensure_ascii=False)

                        # Handle usage from Responses API
                        usage = getattr(response_obj, 'usage', None)
                        if usage:
                            attributes["llm.usage.prompt_tokens"] = getattr(usage, 'prompt_tokens', 0)
                            attributes["llm.usage.completion_tokens"] = getattr(usage, 'completion_tokens', 0)
                            attributes["llm.usage.total_tokens"] = getattr(usage, 'total_tokens', 0)

                    # Then check for OpenAI format (has 'choices')
                    elif hasattr(response_obj, 'choices'):
                        choices = getattr(response_obj, "choices", [])
                        diagnostic_logger.info(f"[RESPONSE_DEBUG] OpenAI format, Choices count: {len(choices)}, type: {type(choices)}")

                        if choices:
                            choice = choices[0]
                            diagnostic_logger.info(f"[RESPONSE_DEBUG] Choice type: {type(choice)}")

                            message = getattr(choice, "message", {})
                            diagnostic_logger.info(f"[RESPONSE_DEBUG] Message type: {type(message)}")

                            content = getattr(message, "content", None)
                            diagnostic_logger.info(f"[RESPONSE_DEBUG] Content from message.content: {repr(content)[:200] if content else 'None'}")

                            if content:
                                attributes["llm.response.content"] = content
                                diagnostic_logger.info(f"[RESPONSE_DEBUG] SUCCESS: Got content ({len(content)} chars)")
                            else:
                                diagnostic_logger.warning(f"[RESPONSE_DEBUG] FAILED: Could not extract content from message.content")

                            # Extract reasoning_content (thinking process) if present
                            reasoning = getattr(message, "reasoning_content", None)
                            if reasoning:
                                attributes["llm.response.reasoning"] = reasoning
                                diagnostic_logger.info(f"[RESPONSE_DEBUG] Got reasoning_content ({len(reasoning)} chars)")
                            else:
                                # Also check provider_specific_fields for reasoning
                                provider_fields = getattr(choice, "provider_specific_fields", {})
                                if provider_fields and isinstance(provider_fields, dict):
                                    reasoning = provider_fields.get("reasoning")
                                    if reasoning:
                                        attributes["llm.response.reasoning"] = reasoning
                                        diagnostic_logger.info(f"[RESPONSE_DEBUG] Got reasoning from provider_fields ({len(reasoning)} chars)")

                            # Extract tool_calls (separately identified)
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                tool_calls = [
                                    {
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        }
                                    }
                                    for tc in message.tool_calls
                                ]
                                attributes["llm.response.tool_calls"] = json.dumps(
                                    tool_calls, ensure_ascii=False
                                )

                            # Provider-specific fields (like token_ids from vLLM)
                            provider_fields = getattr(choice, "provider_specific_fields", {})
                            if provider_fields:
                                attributes["llm.provider.token_ids"] = str(
                                    provider_fields.get("token_ids", [])
                                )

                        # Add usage information
                        usage = getattr(response_obj, "usage", None)
                        if usage:
                            attributes["llm.usage.prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                            attributes["llm.usage.completion_tokens"] = getattr(
                                usage, "completion_tokens", 0
                            )
                            attributes["llm.usage.total_tokens"] = getattr(usage, "total_tokens", 0)

                    # Check for prompt_token_ids (from vLLM)
                    if hasattr(response_obj, "prompt_token_ids"):
                        attributes["llm.prompt_token_ids"] = str(response_obj.prompt_token_ids)

            # Save tool results if present
            if tool_results:
                attributes["llm.conversation.tool_results"] = json.dumps(
                    tool_results, ensure_ascii=False
                )

            # Convert datetime to timestamp if needed
            start_ts = _to_timestamp(start_time)
            end_ts = _to_timestamp(end_time)

            # Create the span
            span = Span(
                name="llm.completion",
                context=SpanContext(trace_id=trace_id, span_id=span_id),
                start_time=start_ts,
                end_time=end_ts,
                kind=SpanKind.CLIENT,
                status="OK",
                attributes=attributes,
                previous_reward=previous_reward,
                resource=Resource(
                    attributes={
                        "service.name": "openclaw-tracer",
                        "llm.provider": kwargs.get("provider", "litellm"),
                    }
                ),
            )

            # Override rollout_id and attempt_id from task context
            if task_id:
                span.rollout_id = task_id
            if attempt_id:
                span.attempt_id = attempt_id

            # Store the span
            await self.store.add_span(span)

        except Exception as e:
            logger.error(f"Error in async_log_success_event: {e}", exc_info=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called when an LLM request fails.

        Still captures complete conversation data for debugging.
        """
        try:
            # Get task context from contextvars
            ctx = task_context.get() or {}
            if ctx.get("no_trace"):
                logger.debug("Skipping failure span capture due to X-Task-No-Trace")
                return
            task_id = ctx.get("task_id")
            attempt_id = ctx.get("attempt_id")
            previous_reward = ctx.get("previous_reward")
            data, messages = _extract_litellm_request(kwargs)
            system_message, messages_for_storage = _split_system_from_messages(messages)

            model = kwargs.get("model", data.get("model", "unknown"))
            span_id = uuid4().hex[:16]
            trace_id = uuid4().hex[:32]

            messages_for_storage, system_message = await self._maybe_externalize_images(
                trace_id, span_id, messages_for_storage, system_message
            )

            attributes: Attributes = {
                "llm.model": model,
                # Full messages array
                "llm.request.messages": json.dumps(messages_for_storage, ensure_ascii=False),
                # Error information
                "llm.error": str(response_obj) if response_obj else "Unknown error",
            }

            # System prompt (separately identified) - only add if present
            if system_message is not None:
                attributes["llm.request.system"] = system_message

            # Convert datetime to timestamp if needed
            start_ts = _to_timestamp(start_time)
            end_ts = _to_timestamp(end_time)

            span = Span(
                name="llm.completion",
                context=SpanContext(trace_id=trace_id, span_id=span_id),
                start_time=start_ts,
                end_time=end_ts,
                kind=SpanKind.CLIENT,
                status="ERROR",
                attributes=attributes,
                previous_reward=previous_reward,
                resource=Resource(
                    attributes={
                        "service.name": "openclaw-tracer",
                        "llm.provider": kwargs.get("provider", "litellm"),
                    }
                ),
            )

            # Override rollout_id and attempt_id from task context
            if task_id:
                span.rollout_id = task_id
            if attempt_id:
                span.attempt_id = attempt_id

            await self.store.add_span(span)

        except Exception as e:
            logger.error(f"Error in async_log_failure_event: {e}", exc_info=True)


class LLMProxy:
    """LiteLLM proxy server for data collection.

    This class wraps LiteLLM's proxy server with custom logging to capture
    all LLM requests and responses for training data collection.

    LiteLLM 使用全局 app，重复 start 只保证不崩，不保证更新中间件配置

    Usage:
        ```python
        store = ParquetStore(output_dir="data")
        proxy = LLMProxy(
            port=43886,
            model_list=[...],
            store=store,
            log_file="logs/http.jsonl"
        )
        await proxy.start()
        ```
    """

    auth_middleware: Optional[AuthMiddleware]

    def __init__(
        self,
        port: Optional[int] = None,
        model_list: Optional[List[Dict[str, Any]]] = None,
        store: Optional[StorageBackend] = None,
        host: str = "0.0.0.0",
        num_workers: int = 1,
        log_file: Optional[str] = None,
        proxy_api_key: Optional[str] = None,
        task_timeout_minutes: int = 10,
        admin_api_key: Optional[str] = None,
        routing_state_path: Optional[str | Path] = None,
    ):
        """Initialize the LLM proxy.

        Args:
            port: Port to listen on. If None, a random port is picked.
            model_list: List of model configurations for LiteLLM.
            store: Storage backend for collected data.
            host: Host to bind to.
            num_workers: Number of worker processes.
            log_file: Path to HTTP access log file (JSONL format).
            proxy_api_key: API key for proxy authentication. Required.
            task_timeout_minutes: Minutes of inactivity before a task is cleaned up.
            admin_api_key: Separate key for /admin/routing APIs (optional).
            routing_state_path: Optional path for persisted routing state.
        """
        from openclaw_tracer.routing_state import (
            ensure_deployment_ids,
            load_routing_state,
            resolve_routing_state_path,
        )

        # Pick random port if not specified
        if port is None:
            self.port = pick_unused_port()
        else:
            self.port = port
        self.host = host
        self.num_workers = num_workers

        # Storage backend
        self.store = store or ParquetStore()

        output_dir = getattr(self.store, "output_dir", None) or "./data"
        self.routing_state_path = resolve_routing_state_path(
            output_dir=output_dir, explicit=routing_state_path
        )

        # Model list: prefer persisted routing state when present
        initial_list = model_list or []
        saved = load_routing_state(self.routing_state_path)
        if saved and isinstance(saved.get("model_list"), list):
            initial_list = saved["model_list"]
            self.routing_updated_at = saved.get("updated_at")
            logger.info(
                "[Routing] Restored %s deployment(s) from %s",
                len(initial_list),
                self.routing_state_path,
            )
        else:
            self.routing_updated_at = None

        self.model_list = ensure_deployment_ids(initial_list)
        self._routing_lock = asyncio.Lock()
        self.inflight = InflightTracker()
        self.admin_api_key = (admin_api_key or os.getenv("ADMIN_API_KEY") or "").strip() or None

        # Span logger
        self.span_logger = SpanLogger(self.store)

        # Request sanitizer (removes unsupported params like tool_choice)
        # Configure which models need tool parameters removed
        # Gemini via OpenAI compatibility mode requires this due to thought_signature requirement
        sanitize_models = self._get_sanitize_models()
        self.request_sanitizer = RequestSanitizer(sanitize_models=sanitize_models)

        # HTTP access logger
        self.http_logger = HTTPAccessLogger(log_file)

        # Authentication middleware
        if proxy_api_key:
            self.auth_middleware = AuthMiddleware(proxy_api_key)
            logger.info("[Auth] Authentication enabled")
        else:
            self.auth_middleware = None
            logger.warning("[Auth] No PROXY_API_KEY provided, running without authentication")

        if self.admin_api_key:
            logger.info("[Auth] Admin API enabled (ADMIN_API_KEY set)")
        else:
            logger.warning("[Auth] ADMIN_API_KEY not set; /admin/routing will return 503")

        # Task Manager for session management
        self.task_manager = TaskManager(timeout_minutes=task_timeout_minutes)
        logger.info(f"[TaskManager] Initialized with timeout={task_timeout_minutes}min")

        # Server state
        self._app: Optional[Any] = None
        self._server: Optional[Any] = None
        self._is_running = False

        # Register the logger with LiteLLM
        self._litellm_callback = self.span_logger
        self._litellm_callback.callback_name = "span_logger"

    def _get_sanitize_models(self) -> Optional[Set[str]]:
        """Identify which models need their tool parameters sanitized.

        Models using OpenAI compatibility mode with different tool calling semantics
        (like Gemini) need tool parameters removed.

        Returns:
            Set of model name prefixes that should be sanitized, or None for all models.
        """
        sanitize_models = set()

        logger.info(f"[RequestSanitizer] Checking {len(self.model_list)} models for sanitization...")

        for model in self.model_list:
            model_name = model.get("model_name", "")
            litellm_params = model.get("litellm_params", {})
            litellm_model = litellm_params.get("model", "")

            logger.info(f"[RequestSanitizer] Checking model: {model_name}, litellm_model: {litellm_model}")

            # Detect models using OpenAI compatibility mode with Gemini
            # These models have the thought_signature requirement issue
            if "openai/" in litellm_model and ("gemini" in litellm_model.lower() or "gemini" in model_name.lower()):
                # Extract the model name prefix (e.g., "gemini-3-flash-preview")
                sanitize_models.add(model_name)
                logger.info(f"[RequestSanitizer] ✓ Marked '{model_name}' for tool parameter sanitization (Gemini via OpenAI compat)")

            # Check for explicit drop_params flag
            if litellm_params.get("drop_params"):
                # Models with drop_params might also need sanitization
                # Add model_name to be safe
                if "gemini" in model_name.lower() or "gemini" in litellm_model.lower():
                    if model_name not in sanitize_models:
                        sanitize_models.add(model_name)
                        logger.info(f"[RequestSanitizer] ✓ Marked '{model_name}' for tool parameter sanitization (drop_params + Gemini)")

        logger.info(f"[RequestSanitizer] Final sanitize_models set: {sanitize_models}")
        return sanitize_models if sanitize_models else None

    async def start(self) -> None:
        """Start the proxy server."""
        if self._is_running:
            logger.warning("Proxy server is already running")
            return

        logger.info(f"Starting LLM proxy on {self.host}:{self.port}")

        # Start TaskManager
        await self.task_manager.start()

        if self.http_logger.log_file:
            logger.info(f"HTTP access log: {self.http_logger.log_file}")

        # Load configuration into LiteLLM proxy (hot-reloadable)
        await self._load_router(self.model_list, persist=True, reason="startup")

        # Add the custom loggers
        from litellm import logging_callback_manager

        # Add span logger for data collection
        logging_callback_manager.add_litellm_callback(self._litellm_callback)

        # Add request sanitizer to remove unsupported parameters (e.g., tools for Gemini)
        self.request_sanitizer.callback_name = "request_sanitizer"
        logging_callback_manager.add_litellm_callback(self.request_sanitizer)

        logger.info(f"Request sanitizer enabled for models: {self.request_sanitizer.sanitize_models}")

        # Add HTTP logging middleware (includes TaskMiddleware)
        self._setup_http_middleware(app)

        # Register /status endpoint for collection progress
        self._register_status_route(app)

        # Admin routing CRUD + flush / batch release
        from openclaw_tracer.admin_ops import register_admin_ops_routes
        from openclaw_tracer.routing_admin import register_routing_admin_routes

        register_routing_admin_routes(app, self)
        register_admin_ops_routes(app, self)

        self._app = app

        # Start the server
        import uvicorn

        server_config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            workers=self.num_workers,
            log_level="info",
        )
        self._server = uvicorn.Server(server_config)

        # Run server in background task
        self._server_task = asyncio.create_task(self._server.serve())
        self._is_running = True

        # Start periodic flush for the storage backend
        if hasattr(self.store, "start_periodic_flush"):
            self.store.start_periodic_flush()

        logger.info(f"LLM proxy started on http://{self.host}:{self.port}")
        logger.info(f"Models available: {[m.get('model_name', m.get('model')) for m in self.model_list]}")

    def _drain_timeout_seconds(self) -> float:
        raw = os.getenv("ROUTING_DRAIN_TIMEOUT_SECONDS", "120")
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 120.0

    async def _load_router(
        self,
        model_list: List[Dict[str, Any]],
        *,
        persist: bool = True,
        reason: str = "reload",
        previous: Optional[List[Dict[str, Any]]] = None,
        op: Optional[Dict[str, Any]] = None,
        schedule_callback: bool = False,
    ) -> Dict[str, Any]:
        """Load model_list into LiteLLM router; optionally persist and callback."""
        from datetime import timezone

        from litellm.proxy.proxy_server import proxy_config
        from litellm.proxy import proxy_server

        from openclaw_tracer.routing_state import (
            ensure_deployment_ids,
            model_list_to_yaml,
            save_routing_state,
        )

        prepared = ensure_deployment_ids(model_list)
        yaml_content = model_list_to_yaml(prepared)
        config_file: Optional[str] = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
                fh.write(yaml_content)
                config_file = fh.name

            new_router, model_list_result, _ = await proxy_config.load_config(
                None, config_file
            )
            proxy_server.llm_router = new_router
            proxy_server.llm_model_list = model_list_result
        finally:
            if config_file:
                try:
                    os.unlink(config_file)
                except OSError:
                    pass

        prev = previous if previous is not None else deepcopy(self.model_list)
        self.model_list = prepared
        updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.routing_updated_at = updated_at

        if persist:
            save_routing_state(self.routing_state_path, prepared, updated_at=updated_at)

        logger.info(
            "[Routing] Loaded %s deployment(s) reason=%s path=%s",
            len(prepared),
            reason,
            self.routing_state_path if persist else "(no persist)",
        )

        if schedule_callback:
            from openclaw_tracer.routing_callback import (
                build_routing_event,
                schedule_routing_change_callback,
            )

            payload = build_routing_event(
                reason=reason,
                previous=prev,
                current=prepared,
                op=op,
            )
            schedule_routing_change_callback(payload)

        return {
            "updated_at": updated_at,
            "model_list": prepared,
            "deployment_count": len(prepared),
        }

    async def reload_models(
        self,
        model_list: List[Dict[str, Any]],
        *,
        reason: str = "reload",
        op: Optional[Dict[str, Any]] = None,
        persist: bool = True,
        schedule_callback: bool = True,
    ) -> Dict[str, Any]:
        """Replace in-memory LiteLLM router with a new model_list."""
        previous = deepcopy(self.model_list)
        return await self._load_router(
            model_list,
            persist=persist,
            reason=reason,
            previous=previous,
            op=op,
            schedule_callback=schedule_callback,
        )

    async def apply_model_list(
        self,
        model_list: List[Dict[str, Any]],
        *,
        reason: str = "manual",
        drain: bool = False,
        op: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate, optionally drain, then reload + persist + callback.

        Raises:
            RoutingApplyError: validation or drain timeout (router unchanged).
        """
        from openclaw_tracer.routing_admin import RoutingApplyError
        from openclaw_tracer.routing_state import (
            ensure_deployment_ids,
            validate_weights_for_model_names,
        )

        prepared = ensure_deployment_ids(model_list)
        weight_err = validate_weights_for_model_names(prepared)
        if weight_err:
            raise RoutingApplyError(weight_err, code="invalid_weights", status_code=400)

        async with self._routing_lock:
            if drain:
                timeout = self._drain_timeout_seconds()
                logger.info(
                    "[Routing] Draining inflight=%s timeout=%ss before reason=%s",
                    self.inflight.count,
                    timeout,
                    reason,
                )
                ok = await self.inflight.wait_zero(timeout)
                if not ok:
                    raise RoutingApplyError(
                        f"drain timed out after {timeout}s "
                        f"(inflight={self.inflight.count})",
                        code="drain_timeout",
                        status_code=409,
                    )

            return await self.reload_models(
                prepared,
                reason=reason,
                op=op,
                persist=True,
                schedule_callback=True,
            )

    def _setup_http_middleware(self, fastapi_app: Any) -> None:
        """Set up HTTP logging and authentication middleware for the FastAPI app.

        Args:
            fastapi_app: The FastAPI application instance
        """
        if getattr(fastapi_app.state, "openclaw_http_middleware_installed", False):
            logger.debug("HTTP middleware already installed on LiteLLM app; skipping add_middleware")
            return

        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request as StarletteRequest
        from starlette.responses import Response as StarletteResponse, JSONResponse

        # Import and add TaskMiddleware FIRST (before auth/logging to extract headers)
        from openclaw_tracer.middleware.task_middleware import TaskMiddleware
        fastapi_app.add_middleware(TaskMiddleware, task_manager=self.task_manager)
        logger.info("[TaskMiddleware] Added to middleware chain")

        inflight_tracker = self.inflight

        class HTTPLogMiddleware(BaseHTTPMiddleware):
            def __init__(
                self,
                app,
                http_logger: HTTPAccessLogger,
                auth_middleware: Optional[AuthMiddleware] = None,
                inflight: Optional[InflightTracker] = None,
            ):
                super().__init__(app)
                self.http_logger = http_logger
                self.auth_middleware = auth_middleware
                self.inflight = inflight

            async def dispatch(
                self,
                request: StarletteRequest,
                call_next: Callable[[StarletteRequest], Awaitable[StarletteResponse]],
            ) -> StarletteResponse:
                # Extract request info
                method = request.method
                path = str(request.url.path)
                query = str(request.url.query) if request.url.query else None

                # Get headers
                headers = dict(request.headers)

                # Authenticate request if auth middleware is enabled
                if self.auth_middleware:
                    is_authenticated, error_response = await self.auth_middleware.authenticate(
                        method=method,
                        path=path,
                        headers=headers
                    )
                    if not is_authenticated:
                        # Return 401 Unauthorized
                        return JSONResponse(
                            status_code=401,
                            content=error_response
                        )

                # Get body (for POST/PUT/PATCH)
                body = None
                if method in ("POST", "PUT", "PATCH"):
                    try:
                        body_bytes = await request.body()
                        if body_bytes:
                            body = body_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        pass

                log_body = (
                    redact_data_url_images_in_json_body_for_log(body)
                    if body and "data:image" in body
                    else body
                )

                # Log request
                request_id = await self.http_logger.log_request(
                    method=method,
                    path=f"{path}?{query}" if query else path,
                    headers=headers,
                    body=log_body,
                )

                track = bool(
                    self.inflight
                    and method in ("POST", "PUT", "PATCH")
                    and InflightTracker.is_inference_path(path)
                )
                if track and self.inflight is not None:
                    await self.inflight.enter()

                # Process request
                try:
                    response = await call_next(request)

                    # Get response body (if possible)
                    response_body = None
                    try:
                        response_body_bytes = response.body
                        if response_body_bytes:
                            # Convert bytes/memoryview to str
                            if isinstance(response_body_bytes, memoryview):
                                response_body = response_body_bytes.tobytes().decode("utf-8", errors="replace")
                            else:
                                response_body = response_body_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        pass

                    # Log response
                    await self.http_logger.log_response(
                        request_id=request_id,
                        status_code=response.status_code,
                        headers=dict(response.headers) if hasattr(response, "headers") else {},
                        body=response_body,
                    )

                    return response

                except Exception as e:
                    # Log error
                    await self.http_logger.log_response(
                        request_id=request_id,
                        status_code=500,
                        headers={},
                        body=None,
                        error=str(e),
                    )
                    raise
                finally:
                    if track and self.inflight is not None:
                        await self.inflight.exit()

        # Add middleware to the app
        fastapi_app.add_middleware(
            HTTPLogMiddleware,
            http_logger=self.http_logger,
            auth_middleware=self.auth_middleware,
            inflight=inflight_tracker,
        )
        fastapi_app.state.openclaw_http_middleware_installed = True

    def _register_status_route(self, fastapi_app: Any) -> None:
        """Register the /status endpoint for collection progress.

        Args:
            fastapi_app: The FastAPI application instance
        """
        if getattr(fastapi_app.state, "openclaw_status_route_registered", False):
            return

        store = self.store

        @fastapi_app.get("/status")
        async def collection_status():
            if hasattr(store, "get_collection_status"):
                return store.get_collection_status()
            return {"error": "Status not available for this storage backend"}

        @fastapi_app.get("/tracer-version")
        async def tracer_version():
            """Return tracer version information (no auth required)."""
            from openclaw_tracer import __version__
            return {
                "name": "openclaw-tracer",
                "version": __version__,
                "features": [
                    "task-tracking",
                    "reward-tracking",
                    "no-trace-header",
                    "flush-api",
                ],
            }

        @fastapi_app.post("/end_task")
        async def end_task(request: Request) -> JSONResponse:
            """End a task and return its statistics."""
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    {"error": "Invalid JSON body"},
                    status_code=400
                )

            task_id = body.get("task_id")
            if not task_id:
                return JSONResponse(
                    {"error": "task_id is required"},
                    status_code=400
                )

            final_reward_raw = body.get("final_reward")
            if final_reward_raw is None:
                final_reward_raw = body.get("reward")

            final_reward = None
            if final_reward_raw is not None:
                try:
                    final_reward = float(final_reward_raw)
                except (TypeError, ValueError):
                    return JSONResponse(
                        {
                            "error": "final_reward (or reward) must be a number",
                            "task_id": task_id,
                        },
                        status_code=400,
                    )

            stats = await self.task_manager.end_task(task_id, final_reward=final_reward)
            if stats is None:
                return JSONResponse(
                    {"error": "Task not found", "task_id": task_id},
                    status_code=404
                )

            if isinstance(self.store, ParquetStore):
                await self.store.apply_rollout_final_reward(stats.task_id, stats.final_reward)

            return JSONResponse(
                {
                    "task_id": stats.task_id,
                    "attempt_count": stats.attempt_count,
                    "total_requests": stats.total_requests,
                    "duration_seconds": stats.duration_seconds,
                    "final_reward": stats.final_reward,
                },
                status_code=200
            )

        # NOTE: POST /flush moved to POST /admin/flush (ADMIN_API_KEY); see admin_ops.py

        fastapi_app.state.openclaw_status_route_registered = True

    async def stop(self) -> None:
        """Stop the proxy server."""
        if not self._is_running:
            return

        logger.info("Stopping LLM proxy")

        if self._server:
            self._server.should_exit = True

        if self._server_task:
            await self._server_task

        # Stop TaskManager
        await self.task_manager.stop()

        # Flush storage
        await self.store.flush()

        self._is_running = False
        logger.info("LLM proxy stopped")

    async def wait(self) -> None:
        """Wait until the server is stopped."""
        if self._server_task:
            await self._server_task

    @property
    def is_running(self) -> bool:
        """Check if the proxy is running."""
        return self._is_running

    @property
    def url(self) -> str:
        """Get the base URL of the proxy."""
        return f"http://{self.host}:{self.port}"

    @property
    def v1_url(self) -> str:
        """Get the v1 API endpoint URL."""
        return f"{self.url}/v1"

    @property
    def stats(self) -> Dict[str, Any]:
        """Get proxy statistics."""
        return {
            "host": self.host,
            "port": self.port,
            "is_running": self._is_running,
            "url": self.url,
            "model_count": len(self.model_list),
            "storage": self.store.stats,
            "log_file": str(self.http_logger.log_file) if self.http_logger.log_file else None,
        }


# Convenience function for quick startup


async def run_proxy(
    model_list: List[Dict[str, Any]],
    port: Optional[int] = None,
    output_dir: str = "data",
    log_file: Optional[str] = None,
    proxy_api_key: Optional[str] = None,
    task_timeout_minutes: int = 10,
) -> LLMProxy:
    """Run the LLM proxy server.

    Args:
        model_list: List of model configurations.
        port: Port to listen on.
        output_dir: Directory for Parquet output.
        log_file: Path to HTTP access log file (JSONL format).
        proxy_api_key: API key for proxy authentication.
        task_timeout_minutes: Minutes of inactivity before a task is cleaned up.

    Returns:
        The running LLMProxy instance.

    Example:
        ```python
        proxy = await run_proxy(
            model_list=[{
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4"},
            }],
            port=43886,
            log_file="logs/http.jsonl",
            proxy_api_key="your-api-key"
        )
        ```
    """
    store = ParquetStore(output_dir=output_dir)
    proxy = LLMProxy(
        port=port,
        model_list=model_list,
        store=store,
        log_file=log_file,
        proxy_api_key=proxy_api_key,
        task_timeout_minutes=task_timeout_minutes,
    )
    await proxy.start()
    return proxy
