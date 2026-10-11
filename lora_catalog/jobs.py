"""
Which LoRA Catalog runs are going right now, on any OS: each run.py command (except the editor) leaves a small lock
file in DATA_DIR/locks while it runs. A lock whose process has gone (a crash, a killed window) is ignored and cleaned
up. Used by the refresh (no two card writers at once), the editor's file moves (no moving under a running job) and
the ComfyUI panel (what is running).

    with job("cards"):          # in __main__
        ...
    running()                   -> [{"command": "cards", "pid": 123, "started": "...", "args": "..."}]

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, Iterator, List

from .config import DATA_DIR

LOCK_DIR = os.path.join(DATA_DIR, "locks")
# Commands that write cards or move rows by path: two of these at once fight over the GPU or the catalog.
WRITERS = {"cards", "genre-mood", "sliders", "refresh", "index", "facets"}


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259                                  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running(exclude_self: bool = True) -> List[Dict[str, Any]]:
    """Runs going now (stale locks are removed)."""
    found = []
    if not os.path.isdir(LOCK_DIR):
        return found
    for name in os.listdir(LOCK_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(LOCK_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
        except (OSError, ValueError):
            continue
        pid = int(info.get("pid") or 0)
        if exclude_self and pid == os.getpid():
            continue
        if _alive(pid):
            found.append(info)
        else:
            with contextlib.suppress(OSError):
                os.remove(path)
    return sorted(found, key=lambda info: info.get("started", ""))


def busy_writers() -> List[Dict[str, Any]]:
    return [info for info in running() if info.get("command") in WRITERS]


@contextlib.contextmanager
def job(command: str) -> Iterator[None]:
    os.makedirs(LOCK_DIR, exist_ok=True)
    path = os.path.join(LOCK_DIR, f"{command}-{os.getpid()}.json")
    info = {"command": command, "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": " ".join(sys.argv[1:])[:300]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)
