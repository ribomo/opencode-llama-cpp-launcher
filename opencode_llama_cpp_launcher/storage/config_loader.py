from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from opencode_llama_cpp_launcher.models.launch_config import (
    DEFAULT_CTX_SIZE,
    DEFAULT_PORT,
    MAX_PORT,
)
from opencode_llama_cpp_launcher.services.errors import LauncherError


DEFAULT_CONFIG_NAMES = (".opencode-llama.yaml", ".opencode-llama.yml")

CONFIG_TEMPLATE = """# .opencode-llama.yaml
model: /absolute/path/to/model.gguf
llama_server: /optional/path/to/llama-server
port: 8080
ctx_size: 8192
"""


@dataclass(frozen=True)
class FileConfig:
    path: Path | None
    model: Path | None = None
    llama_server_path: Path | None = None
    port: int = DEFAULT_PORT
    ctx_size: int = DEFAULT_CTX_SIZE


def find_config_path(project: Path, explicit_config: Path | None) -> Path | None:
    if explicit_config is not None:
        return explicit_config.expanduser()

    for config_name in DEFAULT_CONFIG_NAMES:
        candidate = project / config_name
        if candidate.exists():
            return candidate

    return None


def load_file_config(project: Path, explicit_config: Path | None = None) -> FileConfig:
    config_path = find_config_path(project, explicit_config)
    if config_path is None:
        return FileConfig(path=None)

    if not config_path.exists():
        raise LauncherError(f"Config file does not exist: {config_path}")

    raw_config = _load_yaml_mapping(config_path)
    return FileConfig(
        path=config_path,
        model=_path_value(raw_config.get("model"), config_path.parent, "model"),
        llama_server_path=_path_value(
            raw_config.get("llama_server"),
            config_path.parent,
            "llama_server",
        ),
        port=_int_value(raw_config.get("port"), DEFAULT_PORT, "port"),
        ctx_size=_int_value(raw_config.get("ctx_size"), DEFAULT_CTX_SIZE, "ctx_size"),
    )


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LauncherError(f"Could not parse YAML config {config_path}: {exc}") from exc
    except OSError as exc:
        raise LauncherError(f"Could not read config {config_path}: {exc}") from exc

    if raw_config is None:
        return {}

    if not isinstance(raw_config, dict):
        raise LauncherError(f"Config file must contain a YAML mapping: {config_path}")

    return raw_config


def _path_value(value: Any, base_dir: Path, key: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LauncherError(f"Config value `{key}` must be a string path.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _int_value(value: Any, default: int, key: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise LauncherError(f"Config value `{key}` must be an integer.")
    if value <= 0:
        raise LauncherError(f"Config value `{key}` must be greater than zero.")
    if key == "port" and value > MAX_PORT:
        raise LauncherError(f"Config value `{key}` must be at most {MAX_PORT}.")
    return value
