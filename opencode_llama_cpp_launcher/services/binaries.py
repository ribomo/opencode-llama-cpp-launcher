from __future__ import annotations

import shutil

from opencode_llama_cpp_launcher.services.errors import LauncherError


def require_binary(name: str) -> str:
    binary_path = shutil.which(name)
    if binary_path is None:
        raise LauncherError(f"`{name}` was not found in PATH.")
    return binary_path
