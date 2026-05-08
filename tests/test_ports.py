from __future__ import annotations

import socket

from opencode_llama_cpp_launcher.services.ports import select_port


def test_select_port_returns_preferred_when_available() -> None:
    port = select_port("127.0.0.1", 50801)

    assert port == 50801


def test_select_port_falls_back_when_busy() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        busy_port = int(sock.getsockname()[1])

        selected_port = select_port("127.0.0.1", busy_port)

    assert selected_port != busy_port
    assert isinstance(selected_port, int)
