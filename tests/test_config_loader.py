from __future__ import annotations

from pathlib import Path

import pytest

from opencode_llama_cpp_launcher.models.launch_config import DEFAULT_CTX_SIZE, DEFAULT_PORT
from opencode_llama_cpp_launcher.services.errors import LauncherError
from opencode_llama_cpp_launcher.services.launch_config_loader import LaunchConfigLoader
from opencode_llama_cpp_launcher.storage.config_loader import (
    CONFIG_TEMPLATE,
    XDG_CONFIG_HOME_ENV,
    load_file_config,
)


@pytest.fixture(autouse=True)
def isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(XDG_CONFIG_HOME_ENV, str(tmp_path / "xdg-config"))


def _user_config_path(tmp_path: Path) -> Path:
    return tmp_path / "xdg-config" / "opencode-llama.yaml"


def _dotted_user_config_path(tmp_path: Path) -> Path:
    return tmp_path / "xdg-config" / ".opencode-llama.yaml"


def test_loads_default_yaml_config(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    llama_server = tmp_path / "llama-server"
    model.write_text("", encoding="utf-8")
    llama_server.write_text("", encoding="utf-8")
    (tmp_path / "opencode-llama.yaml").write_text(
        f"model: {model}\nllama_server: {llama_server}\nport: 9001\nctx_size: 4096\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.path == tmp_path / "opencode-llama.yaml"
    assert config.model == model
    assert config.llama_server_path == llama_server
    assert config.port == 9001
    assert config.ctx_size == 4096


def test_loads_yml_config_when_yaml_missing(tmp_path: Path) -> None:
    (tmp_path / "opencode-llama.yml").write_text(
        "model: model.gguf\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.path == tmp_path / "opencode-llama.yml"
    assert config.model == tmp_path / "model.gguf"
    assert config.llama_server_path is None
    assert config.port == DEFAULT_PORT
    assert config.ctx_size == DEFAULT_CTX_SIZE


def test_loads_dotted_project_config_when_preferred_project_config_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / ".opencode-llama.yaml").write_text(
        "model: dotted-project.gguf\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.path == tmp_path / ".opencode-llama.yaml"
    assert config.model == tmp_path / "dotted-project.gguf"


def test_preferred_project_config_wins_over_dotted_project_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "opencode-llama.yaml").write_text(
        "model: preferred-project.gguf\n",
        encoding="utf-8",
    )
    (tmp_path / ".opencode-llama.yaml").write_text(
        "model: dotted-project.gguf\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.path == tmp_path / "opencode-llama.yaml"
    assert config.model == tmp_path / "preferred-project.gguf"


def test_loads_user_config_when_project_config_missing(tmp_path: Path) -> None:
    user_config = _user_config_path(tmp_path)
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        "model: model.gguf\nllama_server: llama-server\nport: 9001\nctx_size: 4096\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.path == user_config
    assert config.model == user_config.parent / "model.gguf"
    assert config.llama_server_path == user_config.parent / "llama-server"
    assert config.port == 9001
    assert config.ctx_size == 4096


def test_loads_dotted_user_config_when_preferred_user_config_missing(
    tmp_path: Path,
) -> None:
    user_config = _dotted_user_config_path(tmp_path)
    user_config.parent.mkdir(parents=True)
    user_config.write_text("model: dotted-user.gguf\n", encoding="utf-8")

    config = load_file_config(tmp_path)

    assert config.path == user_config
    assert config.model == user_config.parent / "dotted-user.gguf"


def test_preferred_user_config_wins_over_dotted_user_config(tmp_path: Path) -> None:
    user_config = _user_config_path(tmp_path)
    dotted_user_config = _dotted_user_config_path(tmp_path)
    user_config.parent.mkdir(parents=True)
    user_config.write_text("model: preferred-user.gguf\n", encoding="utf-8")
    dotted_user_config.write_text("model: dotted-user.gguf\n", encoding="utf-8")

    config = load_file_config(tmp_path)

    assert config.path == user_config
    assert config.model == user_config.parent / "preferred-user.gguf"


def test_project_config_wins_over_user_config(tmp_path: Path) -> None:
    user_config = _user_config_path(tmp_path)
    user_config.parent.mkdir(parents=True)
    user_config.write_text("model: user.gguf\n", encoding="utf-8")
    (tmp_path / "opencode-llama.yaml").write_text(
        "model: project.gguf\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.path == tmp_path / "opencode-llama.yaml"
    assert config.model == tmp_path / "project.gguf"


def test_explicit_config_path_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("model: explicit.gguf\n", encoding="utf-8")
    (tmp_path / "opencode-llama.yaml").write_text(
        "model: default.gguf\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path, explicit)

    assert config.path == explicit
    assert config.model == tmp_path / "explicit.gguf"


def test_cli_flags_override_yaml_config(tmp_path: Path) -> None:
    cli_model = tmp_path / "cli.gguf"
    cli_llama_server = tmp_path / "cli-llama-server"
    cli_model.write_text("", encoding="utf-8")
    cli_llama_server.write_text("", encoding="utf-8")
    (tmp_path / "opencode-llama.yaml").write_text(
        "model: config.gguf\nllama_server: config-llama-server\nport: 9001\nctx_size: 4096\n",
        encoding="utf-8",
    )

    config = LaunchConfigLoader().load(
        model=cli_model,
        llama_server=cli_llama_server,
        project=tmp_path,
        config=None,
        port=9002,
        ctx_size=2048,
        dry_run=True,
    )

    assert config.model_path == cli_model
    assert config.llama_server_path == cli_llama_server
    assert config.project == tmp_path
    assert config.port == 9002
    assert config.ctx_size == 2048
    assert config.dry_run is True


def test_missing_model_error_includes_yaml_template(tmp_path: Path) -> None:
    with pytest.raises(LauncherError) as exc_info:
        LaunchConfigLoader().load(
            model=None,
            llama_server=None,
            project=tmp_path,
            config=None,
            port=None,
            ctx_size=None,
            dry_run=False,
        )

    assert "No GGUF model path" in str(exc_info.value)
    assert CONFIG_TEMPLATE.strip() in str(exc_info.value)


def test_cli_rejects_port_above_tcp_range(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("", encoding="utf-8")

    with pytest.raises(LauncherError, match="Port must be between 1 and 65535"):
        LaunchConfigLoader().load(
            model=model,
            llama_server=None,
            project=tmp_path,
            config=None,
            port=70000,
            ctx_size=None,
            dry_run=False,
        )


def test_yaml_rejects_port_above_tcp_range(tmp_path: Path) -> None:
    (tmp_path / "opencode-llama.yaml").write_text(
        "model: model.gguf\nport: 70000\n",
        encoding="utf-8",
    )

    with pytest.raises(LauncherError, match="must be at most 65535"):
        load_file_config(tmp_path)
