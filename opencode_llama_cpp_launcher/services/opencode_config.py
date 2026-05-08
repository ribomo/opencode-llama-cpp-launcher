from __future__ import annotations

import json


PROVIDER_ID = "llama.cpp"
PROVIDER_PACKAGE = "@ai-sdk/openai-compatible"


def build_opencode_config(base_url: str, model_id: str) -> dict:
    selected_model = f"{PROVIDER_ID}/{model_id}"
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": selected_model,
        "small_model": selected_model,
        "provider": {
            PROVIDER_ID: {
                "npm": PROVIDER_PACKAGE,
                "name": "llama-server (local)",
                "options": {
                    "baseURL": base_url,
                },
                "models": {
                    model_id: {
                        "name": f"{model_id} (local)",
                    },
                },
            },
        },
    }


def dumps_opencode_config(config: dict) -> str:
    return json.dumps(config, separators=(",", ":"))
