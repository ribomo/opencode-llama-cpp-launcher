from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from opencode_llama_cpp_launcher.cli import entrypoint
from opencode_llama_cpp_launcher.cli.entrypoint import app


def test_cli_missing_model_prints_template(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--project", str(tmp_path)])

    assert result.exit_code == 1
    assert "No GGUF model path" in result.stderr
    assert ".opencode-llama.yaml" in result.stderr


def test_cli_dry_run_smoke(monkeypatch, tmp_path: Path) -> None:
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

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--project",
            str(tmp_path),
            "--model",
            str(model),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "llama-server command:" in result.stdout
    assert "OPENCODE_CONFIG_CONTENT:" in result.stdout


def test_doctor_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "opencode_llama_cpp_launcher.cli.entrypoint.require_binary",
        lambda name: f"/bin/{name}",
    )

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "OK llama-server" in result.stdout
    assert "OK opencode" in result.stdout


def test_cli_maps_signal_return_code_to_shell_exit_code(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("", encoding="utf-8")

    class FakeLauncher:
        def run(self, config):
            return -2

    monkeypatch.setattr(entrypoint, "Launcher", FakeLauncher)

    result = CliRunner().invoke(
        app,
        [
            "--project",
            str(tmp_path),
            "--model",
            str(model),
        ],
    )

    assert result.exit_code == 130


def test_cli_preserves_non_negative_launcher_return_code(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("", encoding="utf-8")

    class FakeLauncher:
        def run(self, config):
            return 7

    monkeypatch.setattr(entrypoint, "Launcher", FakeLauncher)

    result = CliRunner().invoke(
        app,
        [
            "--project",
            str(tmp_path),
            "--model",
            str(model),
        ],
    )

    assert result.exit_code == 7
