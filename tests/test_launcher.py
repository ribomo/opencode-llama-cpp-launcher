from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from opencode_llama_cpp_launcher.models.launch_config import LaunchConfig
from opencode_llama_cpp_launcher.services.errors import LauncherError
from opencode_llama_cpp_launcher.services.launch_preparer import LaunchPreparer
from opencode_llama_cpp_launcher.services.launcher import Launcher


def test_builder_builds_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.require_binary",
        lambda name: f"/bin/{name}",
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.select_port",
        lambda host, port: 9001,
    )

    built_config = LaunchPreparer().prepare(
        LaunchConfig(project=tmp_path, model_path=model, port=8080, ctx_size=4096)
    )

    assert built_config.llama_server.model_path == model
    assert built_config.llama_server.command == [
        "/bin/llama-server",
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "9001",
        "-c",
        "4096",
    ]
    assert built_config.opencode.command == ["/bin/opencode"]
    assert built_config.llama_server.selected_port == 9001


def test_builder_uses_configured_llama_server_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    llama_server = tmp_path / "llama-server"
    model.write_text("", encoding="utf-8")
    llama_server.write_text("", encoding="utf-8")
    llama_server.chmod(0o755)

    binary_lookups: list[str] = []

    def fake_require_binary(name: str) -> str:
        binary_lookups.append(name)
        return f"/bin/{name}"

    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.require_binary",
        fake_require_binary,
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.select_port",
        lambda host, port: 9001,
    )

    built_config = LaunchPreparer().prepare(
        LaunchConfig(
            project=tmp_path,
            model_path=model,
            llama_server_path=llama_server,
        )
    )

    assert built_config.llama_server.command[0] == str(llama_server)
    assert binary_lookups == ["opencode"]


def test_builder_rejects_missing_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.require_binary",
        lambda name: f"/bin/{name}",
    )

    with pytest.raises(LauncherError):
        LaunchPreparer().prepare(
            LaunchConfig(project=tmp_path, model_path=tmp_path / "missing.gguf")
        )


def test_run_sets_opencode_config_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("", encoding="utf-8")
    calls: list[dict] = []
    processes_by_pid = {}
    killpg_calls: list[tuple[int, int]] = []

    class FakeProcess:
        def __init__(
            self,
            command,
            stdin=None,
            stdout=None,
            stderr=None,
            start_new_session=None,
            cwd=None,
            env=None,
        ):
            self.command = command
            self.pid = len(calls) + 100
            self.cwd = cwd
            self.env = env
            self.terminated = False
            if command[0] == "/bin/llama-server" and stdout is not None:
                stdout.write(b"startup details stay hidden on success\n")
                stdout.flush()
            processes_by_pid[self.pid] = self
            calls.append(
                {
                    "command": command,
                    "stdin": stdin,
                    "stdout": stdout,
                    "stderr": stderr,
                    "start_new_session": start_new_session,
                    "cwd": cwd,
                    "env": env,
                    "process": self,
                }
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    def fake_killpg(pid, sig):
        killpg_calls.append((pid, sig))
        processes_by_pid[pid].terminated = True

    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.require_binary",
        lambda name: f"/bin/{name}",
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.select_port",
        lambda host, port: 9001,
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launcher.wait_for_health",
        lambda root_url, **kwargs: None,
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launcher.fetch_first_model_id",
        lambda base_url: "real-model",
    )
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launcher.os.killpg",
        fake_killpg,
    )

    exit_code = Launcher(report_status=lambda status: None).run(
        LaunchConfig(project=tmp_path, model_path=model)
    )

    assert exit_code == 0
    assert calls[0]["command"][0] == "/bin/llama-server"
    assert calls[0]["stdin"] == subprocess.DEVNULL
    assert calls[0]["stdout"] != subprocess.DEVNULL
    assert calls[0]["stdout"].closed is True
    assert calls[0]["stderr"] == subprocess.STDOUT
    assert calls[0]["start_new_session"] is True
    assert calls[1]["command"] == ["/bin/opencode"]
    assert calls[1]["cwd"] == tmp_path
    assert "OPENCODE_CONFIG_CONTENT" in calls[1]["env"]
    assert "real-model" in calls[1]["env"]["OPENCODE_CONFIG_CONTENT"]
    assert killpg_calls == [(calls[0]["process"].pid, signal.SIGTERM)]
    assert calls[0]["process"].terminated is True
    captured = capsys.readouterr()
    assert "startup details stay hidden on success" not in captured.out
    assert "startup details stay hidden on success" not in captured.err


def test_run_fails_when_llama_server_exits_before_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("", encoding="utf-8")

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.terminated = False
            stdout = kwargs.get("stdout")
            if stdout is not None:
                stdout.write(b"failed to load model on GPU 0\n")
                stdout.flush()

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    def fake_wait_for_health(root_url, *, is_server_running):
        if not is_server_running():
            raise LauncherError("llama-server exited before becoming ready.")

    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.require_binary",
        lambda name: f"/bin/{name}",
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launch_preparer.select_port",
        lambda host, port: 9001,
    )
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.services.launcher.wait_for_health",
        fake_wait_for_health,
    )
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)

    with pytest.raises(LauncherError) as exc_info:
        Launcher(report_status=lambda status: None).run(
            LaunchConfig(project=tmp_path, model_path=model)
        )

    message = str(exc_info.value)
    assert "llama-server exited before becoming ready." in message
    assert "llama-server output:" in message
    assert "failed to load model on GPU 0" in message
