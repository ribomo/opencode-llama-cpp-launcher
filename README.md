# OpenCode llama.cpp Launcher

A one command solution for launching [OpenCode](https://opencode.ai/) with any
local LLM that `llama-server` can serve, including models like Qwen, DeepSeek,
and Gemma. This launcher starts `llama-server`, waits for it to become ready,
wires the OpenAI compatible provider config into OpenCode, and cleans up when
the local agentic coding session ends.

## Requirements

- Python 3.12+
- OpenCode
- llama.cpp's `llama-server`
- A local model supported by `llama-server`, for example Qwen, DeepSeek, or
  Gemma

The launcher finds `llama-server` on `PATH`, or you can set `llama_server` in
your config.

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
llama_server: /optional/path/to/llama-server
port: 8080
ctx_size: 8192
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
