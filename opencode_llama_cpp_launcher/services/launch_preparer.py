from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from opencode_llama_cpp_launcher.models.launch_config import (
    LaunchConfig,
    BuiltLaunchConfig,
    LlamaServerConfig,
    OpenCodeConfig,
)
from opencode_llama_cpp_launcher.services.binaries import require_binary
from opencode_llama_cpp_launcher.services.errors import LauncherError
from opencode_llama_cpp_launcher.services.opencode_config import (
    build_opencode_config as build_opencode_provider_config,
)
from opencode_llama_cpp_launcher.services.ports import select_port


@dataclass(frozen=True)
class ServerUrls:
    root: str
    base: str


class LaunchPreparer:
    """Turn the launch request into commands, URLs, and provider config."""

    def prepare(self, config: LaunchConfig) -> BuiltLaunchConfig:
        self._validate_launch_targets(config)

        llama_server_binary = self._llama_server_binary(config)
        opencode_binary = require_binary("opencode")
        selected_port = select_port(config.host, config.port)
        urls = self._server_urls(config.host, selected_port)

        return BuiltLaunchConfig(
            project=config.project,
            llama_server=self._build_llama_server_config(
                config=config,
                llama_server_binary=llama_server_binary,
                selected_port=selected_port,
                urls=urls,
            ),
            opencode=self._build_opencode_config(
                opencode_binary=opencode_binary,
                base_url=urls.base,
                model_id=config.model_path.stem,
            ),
        )

    def _build_llama_server_config(
        self,
        *,
        config: LaunchConfig,
        llama_server_binary: str,
        selected_port: int,
        urls: ServerUrls,
    ) -> LlamaServerConfig:
        return LlamaServerConfig(
            model_path=config.model_path,
            command=self._llama_server_command(
                llama_server_binary=llama_server_binary,
                config=config,
                selected_port=selected_port,
            ),
            selected_port=selected_port,
            root_url=urls.root,
            base_url=urls.base,
        )

    def _build_opencode_config(
        self,
        *,
        opencode_binary: str,
        base_url: str,
        model_id: str,
    ) -> OpenCodeConfig:
        return OpenCodeConfig(
            command=[opencode_binary],
            config=build_opencode_provider_config(base_url, model_id),
        )

    def _llama_server_binary(self, config: LaunchConfig) -> str:
        if config.llama_server_path is None:
            return require_binary("llama-server")

        # A configured binary path is intentional, so fail clearly instead of
        # silently falling back to a different llama-server on PATH.
        return self._checked_binary_path("llama-server", config.llama_server_path)

    def _checked_binary_path(self, name: str, path: Path) -> str:
        if not path.exists() or not path.is_file():
            raise LauncherError(f"`{name}` binary does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise LauncherError(f"`{name}` binary is not executable: {path}")

        return str(path)

    def _validate_launch_targets(self, config: LaunchConfig) -> None:
        if not config.project.exists() or not config.project.is_dir():
            raise LauncherError(f"Project directory does not exist: {config.project}")
        if not config.model_path.exists() or not config.model_path.is_file():
            raise LauncherError(
                f"Selected GGUF file does not exist: {config.model_path}"
            )

    def _llama_server_command(
        self,
        llama_server_binary: str,
        config: LaunchConfig,
        selected_port: int,
    ) -> list[str]:
        return [
            llama_server_binary,
            "-m",
            str(config.model_path),
            "--host",
            config.host,
            "--port",
            str(selected_port),
            "-c",
            str(config.ctx_size),
        ]

    def _server_urls(self, host: str, port: int) -> ServerUrls:
        root_url = f"http://{host}:{port}"
        # llama-server's OpenAI-compatible routes live under /v1.
        return ServerUrls(root=root_url, base=f"{root_url}/v1")
