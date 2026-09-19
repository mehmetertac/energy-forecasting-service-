"""Run pytest with the project virtualenv when available."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_python() -> Path:
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def main() -> int:
    python = resolve_python()
    return subprocess.call([str(python), "-m", "pytest", "-q", "tests"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
