from __future__ import annotations

import json

from opencode_llama_cpp_launcher.services.opencode_config import (
    build_opencode_config,
    dumps_opencode_config,
)


def test_builds_llama_cpp_provider_config() -> None:
    config = build_opencode_config(
        "http://127.0.0.1:8080/v1",
        "qwen3-coder:a3b",
    )

    assert config["model"] == "llama.cpp/qwen3-coder:a3b"
    assert config["small_model"] == "llama.cpp/qwen3-coder:a3b"
    provider = config["provider"]["llama.cpp"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert provider["models"]["qwen3-coder:a3b"]["name"] == "qwen3-coder:a3b (local)"


def test_dumps_valid_json() -> None:
    config = build_opencode_config("http://127.0.0.1:8080/v1", "model")

    assert json.loads(dumps_opencode_config(config)) == config
