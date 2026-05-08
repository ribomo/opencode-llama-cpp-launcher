from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from opencode_llama_cpp_launcher.services.errors import LauncherError


def wait_for_health(
    root_url: str,
    timeout_seconds: float = 120.0,
    interval_seconds: float = 0.5,
    is_server_running: Callable[[], bool] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if is_server_running is not None and not is_server_running():
            raise LauncherError("llama-server exited before becoming ready.")

        try:
            with urllib.request.urlopen(f"{root_url}/health", timeout=2.0) as response:
                if response.status == 200:
                    return
        except OSError as exc:
            last_error = exc

        if is_server_running is not None and not is_server_running():
            raise LauncherError("llama-server exited before becoming ready.")

        time.sleep(interval_seconds)

    suffix = f" Last error: {last_error}" if last_error else ""
    raise LauncherError(f"llama-server did not become ready at {root_url}.{suffix}")


def fetch_first_model_id(base_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LauncherError(f"Could not query llama-server models: {exc}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LauncherError("llama-server returned invalid JSON from /v1/models.") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise LauncherError("No models returned from llama-server /v1/models.")

    first_model = data[0]
    model_id = first_model.get("id") if isinstance(first_model, dict) else None
    if not isinstance(model_id, str) or not model_id:
        raise LauncherError("First llama-server model entry did not include an id.")

    return model_id
