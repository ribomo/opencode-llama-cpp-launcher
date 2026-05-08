from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_PORT = 65535
DEFAULT_CTX_SIZE = 8192


@dataclass(frozen=True)
class LaunchConfig:
    project: Path
    model_path: Path
    llama_server_path: Path | None = None
    port: int = DEFAULT_PORT
    ctx_size: int = DEFAULT_CTX_SIZE
    host: str = DEFAULT_HOST
    dry_run: bool = False


@dataclass(frozen=True)
class LlamaServerConfig:
    model_path: Path
    command: list[str]
    selected_port: int
    root_url: str
    base_url: str


@dataclass(frozen=True)
class OpenCodeConfig:
    command: list[str]
    config: dict


@dataclass(frozen=True)
class BuiltLaunchConfig:
    project: Path
    llama_server: LlamaServerConfig
    opencode: OpenCodeConfig
