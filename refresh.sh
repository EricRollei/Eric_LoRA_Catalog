#!/usr/bin/env sh
# Brings the catalog up to date after adding LoRAs (every step skips what is done). Log: <data folder>/logs/refresh.log
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
exec "$PY" -X utf8 "$HERE/run.py" refresh "$@"
