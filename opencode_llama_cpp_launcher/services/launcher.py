from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from collections.abc import Callable
from typing import BinaryIO

from opencode_llama_cpp_launcher.models.launch_config import (
    BuiltLaunchConfig,
    LaunchConfig,
)
from opencode_llama_cpp_launcher.models.launch_status import LaunchStatus
from opencode_llama_cpp_launcher.services.errors import LauncherError
from opencode_llama_cpp_launcher.services.health import fetch_first_model_id, wait_for_health
from opencode_llama_cpp_launcher.services.launch_preparer import LaunchPreparer
from opencode_llama_cpp_launcher.services.opencode_config import (
    build_opencode_config,
    dumps_opencode_config,
)


StatusReporter = Callable[[LaunchStatus], None]
LLAMA_SERVER_LOG_TAIL_BYTES = 4000


def default_reporter(status: LaunchStatus) -> None:
    print(status.value, flush=True)


class Launcher:
    def __init__(
        self,
        report_status: StatusReporter = default_reporter,
        launch_preparer: LaunchPreparer | None = None,
    ) -> None:
        self._report_status = report_status
        self._launch_preparer = launch_preparer or LaunchPreparer()

    def run(self, config: LaunchConfig) -> int:
        built_config = self._build_config(config)

        if config.dry_run:
            self._print_dry_run(built_config)
            return 0

        with tempfile.TemporaryFile("w+b") as llama_server_log:
            llama_process = self._start_llama_server(built_config, llama_server_log)
            try:
                try:
                    self._wait_for_llama_server(
                        built_config.llama_server.root_url,
                        llama_process,
                    )
                except LauncherError as exc:
                    self._raise_with_llama_server_log(exc, llama_server_log)

                # The real model id is only available after llama-server starts.
                opencode_config = self._detect_opencode_config(
                    built_config.llama_server.base_url,
                )
                opencode_process = self._start_opencode(
                    built_config,
                    opencode_config,
                )
                return int(opencode_process.wait())
            finally:
                # OpenCode owns the interactive session; this launcher owns cleanup
                # for the llama-server process it started.
                self._terminate_process(llama_process)

    def _build_config(self, config: LaunchConfig) -> BuiltLaunchConfig:
        self._report_status(LaunchStatus.CHECKING_PREREQUISITES)
        return self._launch_preparer.prepare(config)

    def _start_llama_server(
        self,
        config: BuiltLaunchConfig,
        log_file: BinaryIO,
    ) -> subprocess.Popen:
        self._report_status(LaunchStatus.STARTING_LLAMA_SERVER)
        # Keep llama-server isolated from OpenCode's terminal UI and make the
        # whole llama.cpp process group easy to stop when OpenCode exits.
        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if hasattr(os, "setsid"):
            popen_kwargs["start_new_session"] = True

        return subprocess.Popen(config.llama_server.command, **popen_kwargs)

    def _wait_for_llama_server(
        self,
        root_url: str,
        process: subprocess.Popen,
    ) -> None:
        self._report_status(LaunchStatus.WAITING_FOR_LLAMA_SERVER)
        wait_for_health(
            root_url,
            is_server_running=lambda: process.poll() is None,
        )

    def _detect_opencode_config(self, base_url: str) -> dict:
        self._report_status(LaunchStatus.DETECTING_MODEL)
        model_id = fetch_first_model_id(base_url)
        return build_opencode_config(base_url, model_id)

    def _start_opencode(
        self,
        built_config: BuiltLaunchConfig,
        opencode_config: dict,
    ) -> subprocess.Popen:
        self._report_status(LaunchStatus.STARTING_OPENCODE)
        return subprocess.Popen(
            built_config.opencode.command,
            cwd=built_config.project,
            env=self._opencode_env(opencode_config),
        )

    def _opencode_env(self, opencode_config: dict) -> dict[str, str]:
        env = os.environ.copy()
        env["OPENCODE_CONFIG_CONTENT"] = dumps_opencode_config(opencode_config)
        return env

    def _print_dry_run(self, config: BuiltLaunchConfig) -> None:
        print(f"Project: {config.project}")
        print(f"Model: {config.llama_server.model_path}")
        print(f"Selected port: {config.llama_server.selected_port}")
        print("llama-server command:")
        print(" ".join(config.llama_server.command))
        print("opencode command:")
        print(" ".join(config.opencode.command))
        print("OPENCODE_CONFIG_CONTENT:")
        print(dumps_opencode_config(config.opencode.config))

    def _raise_with_llama_server_log(
        self,
        exc: LauncherError,
        log_file: BinaryIO,
    ) -> None:
        log_excerpt = self._llama_server_log_excerpt(log_file)
        if not log_excerpt:
            raise exc

        raise LauncherError(f"{exc}\n\nllama-server output:\n{log_excerpt}") from exc

    def _llama_server_log_excerpt(self, log_file: BinaryIO) -> str:
        log_file.flush()
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        if size == 0:
            return ""

        offset = max(0, size - LLAMA_SERVER_LOG_TAIL_BYTES)
        log_file.seek(offset)
        excerpt = log_file.read().decode("utf-8", errors="replace")
        if offset > 0:
            return f"[last {LLAMA_SERVER_LOG_TAIL_BYTES} bytes]\n{excerpt}"
        return excerpt

    def _terminate_process(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        self._terminate_process_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            process.wait()

    def _terminate_process_group(self, process: subprocess.Popen) -> None:
        if not hasattr(os, "killpg"):
            process.terminate()
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()

    def _kill_process_group(self, process: subprocess.Popen) -> None:
        if not hasattr(os, "killpg"):
            process.kill()
            return

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
