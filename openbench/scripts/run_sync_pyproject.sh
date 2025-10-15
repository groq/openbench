#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${REPO_ROOT}/.venv/bin/python"

if [[ -x "${VENV_PY}" ]]; then
  exec "${VENV_PY}" scripts/sync_pyproject.py
fi

if command -v python >/dev/null 2>&1; then
  exec python scripts/sync_pyproject.py
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/sync_pyproject.py
fi

echo "No suitable Python interpreter found (.venv/bin/python, python, python3)" >&2
exit 1
