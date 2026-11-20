#!/usr/bin/env sh
# One command for Linux/macOS: creates .venv, installs requirements, runs main.py.
# Usage:  ./run.sh [data_folder] [--open] [other main.py options]
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

venv_python() {
    if [ -x "$ROOT/.venv/bin/python" ]; then echo "$ROOT/.venv/bin/python"
    elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then echo "$ROOT/.venv/Scripts/python.exe"  # Git Bash
    fi
}

VPY="$(venv_python)"
if [ -z "$VPY" ]; then
    PY="${PYTHON:-}"
    if [ -z "$PY" ]; then
        if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
    fi
    echo "[setup] Creating virtual environment .venv ..."
    "$PY" -m venv "$ROOT/.venv"
    VPY="$(venv_python)"
fi

"$VPY" -m pip install -q --disable-pip-version-check -r "$ROOT/requirements.txt"
exec "$VPY" "$ROOT/main.py" "$@"
