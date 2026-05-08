from __future__ import annotations

from pathlib import Path

from opencode_llama_cpp_launcher.models.launch_config import LaunchConfig, MAX_PORT
from opencode_llama_cpp_launcher.services.errors import LauncherError
from opencode_llama_cpp_launcher.storage.config_loader import (
    CONFIG_TEMPLATE,
    FileConfig,
    load_file_config,
)


class LaunchConfigLoader:
    """Merge CLI options and YAML defaults into the launch request."""

    def load(
        self,
        *,
        model: Path | None,
        llama_server: Path | None,
        project: Path,
        config: Path | None,
        port: int | None,
        ctx_size: int | None,
        dry_run: bool,
    ) -> LaunchConfig:
        project_path = self._absolute_path(project)
        file_config = load_file_config(project_path, config)

        # CLI flags should always win over values remembered in the project file.
        model_path = self._model_path_from_cli_or_config(model, file_config)
        llama_server_path = self._llama_server_path_from_cli_or_config(
            llama_server,
            file_config,
        )
        selected_port = self._port_from_cli_or_config(port, file_config)
        selected_ctx_size = self._ctx_size_from_cli_or_config(ctx_size, file_config)

        if model_path is None:
            self._raise_missing_model()

        self._validate_port(selected_port)
        self._validate_ctx_size(selected_ctx_size)

        return LaunchConfig(
            project=project_path,
            model_path=model_path,
            llama_server_path=llama_server_path,
            port=selected_port,
            ctx_size=selected_ctx_size,
            dry_run=dry_run,
        )

    def _model_path_from_cli_or_config(
        self,
        model: Path | None,
        file_config: FileConfig,
    ) -> Path | None:
        if model is not None:
            return self._absolute_path(model)
        if file_config.model is not None:
            return self._absolute_path(file_config.model)
        return None

    def _llama_server_path_from_cli_or_config(
        self,
        llama_server: Path | None,
        file_config: FileConfig,
    ) -> Path | None:
        if llama_server is not None:
            return self._absolute_path(llama_server)
        if file_config.llama_server_path is not None:
            return self._absolute_path(file_config.llama_server_path)
        return None

    def _port_from_cli_or_config(self, port: int | None, file_config: FileConfig) -> int:
        return port if port is not None else file_config.port

    def _ctx_size_from_cli_or_config(
        self,
        ctx_size: int | None,
        file_config: FileConfig,
    ) -> int:
        return ctx_size if ctx_size is not None else file_config.ctx_size

    def _validate_port(self, port: int) -> None:
        if not 0 < port <= MAX_PORT:
            raise LauncherError(f"Port must be between 1 and {MAX_PORT}.")

    def _validate_ctx_size(self, ctx_size: int) -> None:
        if ctx_size <= 0:
            raise LauncherError("Context size must be greater than zero.")

    def _raise_missing_model(self) -> None:
        raise LauncherError(
            "No GGUF model path was provided.\n\n"
            "Pass one with `--model /absolute/path/to/model.gguf` or create:\n\n"
            f"{CONFIG_TEMPLATE}"
        )

    def _absolute_path(self, path: Path) -> Path:
        # Normalize paths once at the boundary so downstream services can compare
        # and execute them without repeating expanduser/relative-path handling.
        return path.expanduser().resolve()
