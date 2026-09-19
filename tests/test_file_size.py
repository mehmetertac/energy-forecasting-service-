"""File-size guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_no_file_exceeds_1000_lines():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_file_size.py"
    spec = importlib.util.spec_from_file_location("check_file_size", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
