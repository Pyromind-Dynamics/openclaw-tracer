# Copyright (c) 2025 OpenClaw-Tracer
# Routing state: deployment ids, YAML for LiteLLM, and on-disk persistence.

from __future__ import annotations

import json
import logging
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


def resolve_routing_state_path(
    output_dir: Optional[str | Path] = None,
    explicit: Optional[str | Path] = None,
) -> Path:
    """Resolve routing state file path from env or defaults."""
    if explicit is not None:
        return Path(os.path.expanduser(str(explicit)))
    env = os.getenv("ROUTING_STATE_PATH")
    if env and env.strip():
        return Path(os.path.expanduser(env.strip()))
    base = Path(output_dir) if output_dir else Path("./data")
    return base / "routing_state.json"


def ensure_deployment_ids(model_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a deep-copied model_list with model_info.id on every entry."""
    result: List[Dict[str, Any]] = []
    for entry in model_list:
        item = deepcopy(entry)
        info = item.setdefault("model_info", {})
        if not isinstance(info, dict):
            info = {}
            item["model_info"] = info
        if not info.get("id"):
            info["id"] = str(uuid4())
        result.append(item)
    return result


def get_deployment_id(entry: Dict[str, Any]) -> Optional[str]:
    info = entry.get("model_info") or {}
    if isinstance(info, dict):
        dep_id = info.get("id")
        return str(dep_id) if dep_id else None
    return None


def find_deployment(
    model_list: List[Dict[str, Any]], deployment_id: str
) -> Tuple[int, Optional[Dict[str, Any]]]:
    for idx, entry in enumerate(model_list):
        if get_deployment_id(entry) == deployment_id:
            return idx, entry
    return -1, None


def model_list_to_yaml(model_list: List[Dict[str, Any]]) -> str:
    """Serialize model_list to LiteLLM-compatible YAML (model_list only)."""
    lines = ["model_list:"]
    if not model_list:
        lines.append("  []")
        return "\n".join(lines) + "\n"

    for model in model_list:
        model_name = model.get("model_name", "")
        lines.append(f"  - model_name: {model_name}")
        litellm_params = model.get("litellm_params") or {}
        lines.append("    litellm_params:")
        for key, value in litellm_params.items():
            if isinstance(value, str):
                escaped = value.replace('"', '\\"')
                lines.append(f'      {key}: "{escaped}"')
            elif isinstance(value, bool):
                lines.append(f"      {key}: {str(value).lower()}")
            else:
                lines.append(f"      {key}: {value}")
        info = model.get("model_info")
        if isinstance(info, dict) and info:
            lines.append("    model_info:")
            for key, value in info.items():
                if isinstance(value, str):
                    escaped = value.replace('"', '\\"')
                    lines.append(f'      {key}: "{escaped}"')
                else:
                    lines.append(f"      {key}: {value}")
    return "\n".join(lines) + "\n"


def redact_model_list_for_callback(model_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep-copy model_list with api_key values redacted."""
    redacted = deepcopy(model_list)
    for entry in redacted:
        params = entry.get("litellm_params") or {}
        if not isinstance(params, dict):
            continue
        for key in ("api_key", "api_key_value"):
            if key in params and params[key]:
                raw = str(params[key])
                if len(raw) <= 8:
                    params[key] = "***"
                else:
                    params[key] = f"{raw[:4]}...{raw[-4:]}"
    return redacted


def load_routing_state(path: Path) -> Optional[Dict[str, Any]]:
    """Load routing state JSON if present and valid."""
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        model_list = data.get("model_list")
        if not isinstance(model_list, list):
            return None
        return data
    except Exception:
        logger.exception("Failed to load routing state from %s", path)
        return None


def save_routing_state(
    path: Path,
    model_list: List[Dict[str, Any]],
    updated_at: Optional[str] = None,
) -> None:
    """Atomically write routing state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": updated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_list": model_list,
    }
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".routing_state_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def shallow_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge patch into a deep copy of base (one level for nested dicts)."""
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = deepcopy(value)
    return result


def validate_weights_for_model_names(model_list: List[Dict[str, Any]]) -> Optional[str]:
    """Ensure each model_name that still has entries has at least one weight > 0."""
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for entry in model_list:
        name = entry.get("model_name") or ""
        by_name.setdefault(name, []).append(entry)

    for name, entries in by_name.items():
        weights = []
        for entry in entries:
            params = entry.get("litellm_params") or {}
            w = params.get("weight", 1)
            try:
                weights.append(float(w))
            except (TypeError, ValueError):
                return f"invalid weight for model_name={name!r}"
            if weights[-1] < 0:
                return f"negative weight for model_name={name!r}"
        if entries and all(w <= 0 for w in weights):
            return (
                f"model_name={name!r} has no deployment with weight > 0; "
                "add a backend or raise a weight before removing others"
            )
    return None
