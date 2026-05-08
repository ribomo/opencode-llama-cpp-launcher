from __future__ import annotations

from enum import StrEnum


class LaunchStatus(StrEnum):
    IDLE = "Idle"
    CHECKING_PREREQUISITES = "Checking prerequisites"
    STARTING_LLAMA_SERVER = "Starting llama-server"
    WAITING_FOR_LLAMA_SERVER = "Waiting for llama-server"
    DETECTING_MODEL = "Detecting model"
    STARTING_OPENCODE = "Starting OpenCode"
    READY = "Ready"
    ERROR = "Error"
