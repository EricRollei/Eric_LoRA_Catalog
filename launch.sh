#!/usr/bin/env sh
# The card editor: http://127.0.0.1:8765 (opens in your browser). Ctrl+C to stop.
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
exec "$PY" -X utf8 "$HERE/run.py" serve "$@"
