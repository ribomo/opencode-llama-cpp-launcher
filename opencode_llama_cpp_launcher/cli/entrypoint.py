from __future__ import annotations

from pathlib import Path

import typer

from opencode_llama_cpp_launcher.services.binaries import require_binary
from opencode_llama_cpp_launcher.services.errors import LauncherError
from opencode_llama_cpp_launcher.services.launch_config_loader import LaunchConfigLoader
from opencode_llama_cpp_launcher.services.launcher import Launcher


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=False,
    help="Launch OpenCode against a local llama.cpp GGUF model.",
)


@app.callback()
def launch_callback(
    ctx: typer.Context,
    model: Path | None = typer.Option(
        None,
        "--model",
        "-m",
        help="GGUF model path. Overrides YAML config.",
    ),
    llama_server: Path | None = typer.Option(
        None,
        "--llama-server",
        help="llama-server binary path. Overrides YAML config and PATH lookup.",
    ),
    project: Path = typer.Option(
        Path("."),
        "--project",
        "-p",
        help="Project directory where OpenCode should run.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-f",
        help="Path to opencode-llama.yaml.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Preferred llama-server port.",
    ),
    ctx_size: int | None = typer.Option(
        None,
        "--ctx-size",
        help="llama-server context size.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print resolved commands and generated config without launching.",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    try:
        launch_config = LaunchConfigLoader().load(
            model=model,
            llama_server=llama_server,
            project=project,
            config=config,
            port=port,
            ctx_size=ctx_size,
            dry_run=dry_run,
        )
        raise typer.Exit(_shell_exit_code(Launcher().run(launch_config)))
    except LauncherError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from exc


@app.command()
def doctor() -> None:
    """Check whether required external binaries are available."""
    failed = False

    for binary_name in ("llama-server", "opencode"):
        try:
            binary_path = require_binary(binary_name)
            typer.echo(f"OK {binary_name}: {binary_path}")
        except LauncherError as exc:
            failed = True
            typer.echo(f"Missing {binary_name}: {exc}", err=True)

    if failed:
        raise typer.Exit(1)


def _print_error(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)


def _shell_exit_code(return_code: int) -> int:
    if return_code >= 0:
        return return_code
    return 128 + abs(return_code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
