# OpenCode llama.cpp Launcher

Launch [OpenCode](https://opencode.ai/) with a local model served by
[llama.cpp](https://github.com/ggml-org/llama.cpp). The launcher starts
`llama-server`, wires OpenCode to it, and cleans up when your session ends.

![OpenCode llama.cpp Launcher demo](https://raw.githubusercontent.com/ribomo/opencode-llama-cpp-launcher/main/docs/opencode-llama-demo.gif)

## Requirements

- [OpenCode](https://opencode.ai/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`
- A local GGUF model, such as Qwen, DeepSeek, or Gemma

The launcher finds `llama-server` on `PATH`, or you can set `llama_server` in
your config.

Install OpenCode using its
[GitHub installation instructions](https://github.com/anomalyco/opencode#installation).
Install llama.cpp using its
[installation guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

## Install

For most users, install with `pipx`:

```bash
pipx install opencode-llama-cpp-launcher
```

Or install with `pip`:

```bash
python -m pip install opencode-llama-cpp-launcher
```

Check that the required external binaries are available:

```bash
opencode-llama doctor
```

## Configure

Create `opencode-llama.yaml` in the project where you want OpenCode to run, or
create `~/.config/opencode-llama.yaml` for a user-wide default:

```yaml
model: /absolute/path/to/model.gguf
ctx_size: 8192

# Optional
port: 8080
llama_server: /optional/path/to/llama-server
```

Config lookup order:

1. The path passed with `--config`
2. `opencode-llama.yaml` or `opencode-llama.yml` in the project directory
3. `~/.config/opencode-llama.yaml`

## Usage

Run with an explicit config file:

```bash
opencode-llama --config opencode-llama.yaml
```

Or pass the model directly:

```bash
opencode-llama --model /absolute/path/to/model.gguf
```

Useful options:

```bash
opencode-llama --help
opencode-llama --dry-run
opencode-llama --config opencode-llama.yaml
opencode-llama --port 9001
opencode-llama --ctx-size 8192
opencode-llama --llama-server /absolute/path/to/llama-server
```

If `llama-server` fails before becoming healthy, the launcher includes a bounded
tail of the server's startup output in the error message. Successful runs stay
quiet.

## Development

Install dependencies from this repository:

```bash
uv sync --dev
```

Run the test suite:

```bash
uv run pytest
```

Before publishing, check for local files:

```bash
git status --short --ignored
```

Do not commit local launcher configs, virtual environments, caches, build
artifacts, or model paths.

## License

MIT
