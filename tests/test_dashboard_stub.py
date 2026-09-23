"""Dashboard placeholder HTTP server."""

from __future__ import annotations

import importlib.util
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

_stub_path = Path(__file__).resolve().parents[1] / "dashboard" / "stub_server.py"
_spec = importlib.util.spec_from_file_location("stub_server", _stub_path)
assert _spec and _spec.loader
_stub_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stub_module)
_StubHandler = _stub_module._StubHandler


def test_stub_server_returns_placeholder_text():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
        body = resp.read().decode()
    assert resp.status == 200
    assert "placeholder" in body.lower()
    thread.join(timeout=2)
