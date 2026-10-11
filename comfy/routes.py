"""
The LoRA Catalog panel's routes on ComfyUI's own server (web/lora_catalog_panel.js calls them).

    GET  /lora_catalog/status            counts, problems, running jobs, the refresh step, editor up or not
    POST /lora_catalog/job               {"command": "refresh" | "cards" | "genre-mood" | ..., "options": {...}}
    POST /lora_catalog/job/stop          {"pid": 123}  (only a pid that one of our lock files names)
    GET  /lora_catalog/log?name=refresh  the last lines of a job's log
    POST /lora_catalog/editor            start the card editor if it is not running; returns its URL
    GET  /lora_catalog/settings          the settings the panel edits (the Civitai key only as set / not set)
    POST /lora_catalog/settings          save them (settings.json)
    POST /lora_catalog/llm_test          one small text call and one small image call

Jobs run as their own process (python run.py <command>), started detached: ComfyUI's queue never waits on them, and
restarting ComfyUI does not stop a card pass. Commands and their options come from fixed lists here; nothing the
browser sends reaches a command line unchecked.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List

from ..lora_catalog import config, jobs
from ..lora_catalog.refresh import read_status

EDITOR_PORT = 8765
LOG_DIR = os.path.join(config.DATA_DIR, "logs")
RUN_PY = os.path.join(config.TOOL_DIR, "run.py")

# command -> allowed options: name -> (flag, kind). kind: bool | int | text
COMMANDS: Dict[str, Dict[str, tuple]] = {
    "refresh": {"skip_internet": ("--skip-internet", "bool"), "workers": ("--workers", "int"),
                "steps": ("--steps", "text")},
    "index": {"force": ("--force", "bool")},
    "cards": {"force": ("--force", "bool"), "workers": ("--workers", "int"), "only": ("--only", "text")},
    "genre-mood": {"force": ("--force", "bool"), "only": ("--only", "text")},
    "community": {"force": ("--force", "bool")},
    "images": {},
    "facets": {},
    "review-all": {},
    "export-sidecars": {"write": ("--write", "bool"), "overwrite_ours": ("--overwrite-ours", "bool")},
    "unload": {},
}
# settings the panel may change (paths and endpoints; nothing else in settings.json is touched)
EDITABLE = {"library_roots", "llm_base_url", "llm_backend", "card_model", "genre_model", "civitai_host",
            "civitai_api_key", "training_roots"}


def _args(command: str, options: Dict[str, Any]) -> List[str]:
    allowed = COMMANDS[command]
    args = [command]
    for key, value in (options or {}).items():
        if key not in allowed:
            continue
        flag, kind = allowed[key]
        if kind == "bool":
            if value is True:
                args.append(flag)
        elif kind == "int":
            args += [flag, str(max(1, min(16, int(value))))]
        elif kind == "text" and str(value).strip():
            text = str(value).strip()
            if len(text) > 200 or text.startswith("-"):
                raise ValueError(f"bad value for {key}")
            args += [flag, text]
    return args


def _start(args: List[str], log_name: str) -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, f"{log_name}.log"), "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  run.py {' '.join(args)}  (from the ComfyUI panel) =====\n")
    log.flush()
    kwargs: Dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                              "cwd": config.TOOL_DIR, "env": dict(os.environ, PYTHONIOENCODING="utf-8")}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                                   | 0x00000008)                                     # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen([sys.executable, "-X", "utf8", RUN_PY] + args, **kwargs)
    log.close()
    return process.pid


def _stop(pid: int) -> str:
    ours = {int(info.get("pid") or 0): info.get("command") for info in jobs.running()}
    if pid not in ours:
        return "not one of the LoRA Catalog's runs"
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            os.kill(pid, signal.SIGTERM)
    return f"stopped {ours[pid]} (pid {pid}); what it finished is saved"


def _counts(settings: config.Settings) -> Dict[str, Any]:
    path = settings.catalog_path
    if not os.path.isfile(path):
        return {"catalog": "not built yet"}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        out: Dict[str, Any] = {}
        for key, sql in (("loras", "select count(*) from loras"),
                         ("cards", "select count(*) from cards"),
                         ("uncarded", "select count(*) from loras l where not exists "
                                      "(select 1 from cards c where c.path = l.path)"),
                         ("no_genre", "select count(*) from cards where genre is null")):
            try:
                out[key] = connection.execute(sql).fetchone()[0]
            except sqlite3.Error:
                pass
        connection.close()
        return out
    except sqlite3.Error as error:
        return {"catalog": f"unreadable: {error}"}


def _editor_up(port: int = EDITOR_PORT) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def _tail(name: str, lines: int) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_")
    path = os.path.join(LOG_DIR, f"{safe}.log")
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 200_000))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _public_settings(settings: config.Settings) -> Dict[str, Any]:
    return {"library_roots": settings.library_roots, "training_roots": settings.training_roots,
            "llm_base_url": settings.llm_base_url, "llm_backend": settings.llm_backend,
            "card_model": settings.card_model, "genre_model": settings.genre_model,
            "civitai_host": settings.civitai_host, "civitai_key_set": bool(settings.civitai_key()),
            "data_dir": config.DATA_DIR, "catalog_path": settings.catalog_path,
            "settings_file": config.SETTINGS_FILE, "comfy_dir": config.COMFY_DIR}


def register() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return False
    if getattr(PromptServer, "instance", None) is None:
        return False
    routes = PromptServer.instance.routes

    async def off_loop(function, *args):
        return await asyncio.get_running_loop().run_in_executor(None, function, *args)

    @routes.get("/lora_catalog/status")
    async def status(_request):
        def build():
            settings = config.load_settings()
            return {"problems": settings.problems(), "counts": _counts(settings), "running": jobs.running(),
                    "refresh": read_status(), "editor": _editor_up(), "editor_url": f"http://127.0.0.1:{EDITOR_PORT}/"}
        return web.json_response(await off_loop(build))

    @routes.post("/lora_catalog/job")
    async def start_job(request):
        body = await request.json()
        command = str(body.get("command") or "")
        if command not in COMMANDS:
            return web.json_response({"error": f"unknown command {command}"}, status=400)
        try:
            args = _args(command, body.get("options") or {})
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        if command in jobs.WRITERS:
            busy = jobs.busy_writers()
            if busy:
                names = ", ".join(f"{b['command']} (pid {b['pid']})" for b in busy)
                return web.json_response({"error": f"already running: {names}"}, status=409)
        pid = await off_loop(_start, args, command)
        return web.json_response({"pid": pid, "log": command})

    @routes.post("/lora_catalog/job/stop")
    async def stop_job(request):
        body = await request.json()
        return web.json_response({"result": await off_loop(_stop, int(body.get("pid") or 0))})

    @routes.get("/lora_catalog/log")
    async def log(request):
        name = request.query.get("name", "refresh")
        lines = max(10, min(2000, int(request.query.get("lines", "200"))))
        return web.json_response({"name": name, "text": await off_loop(_tail, name, lines)})

    @routes.post("/lora_catalog/editor")
    async def editor(_request):
        url = f"http://127.0.0.1:{EDITOR_PORT}/"
        if not await off_loop(_editor_up):
            await off_loop(_start, ["serve", "--no-browser", "--port", str(EDITOR_PORT)], "editor")
            for _ in range(20):
                await asyncio.sleep(0.5)
                if await off_loop(_editor_up):
                    break
        return web.json_response({"url": url, "up": await off_loop(_editor_up)})

    @routes.get("/lora_catalog/settings")
    async def get_settings(_request):
        return web.json_response(await off_loop(lambda: _public_settings(config.load_settings())))

    @routes.post("/lora_catalog/settings")
    async def save(request):
        body = await request.json()
        values = {k: v for k, v in body.items() if k in EDITABLE}
        if "civitai_api_key" in values and not str(values["civitai_api_key"]).strip():
            values.pop("civitai_api_key")                  # blank = keep the key that is set
        for key in ("library_roots", "training_roots"):
            if key in values:
                values[key] = [str(v).strip() for v in values[key] if str(v).strip()]
        path = await off_loop(config.save_settings, values)
        settings = await off_loop(config.load_settings)
        return web.json_response({"saved": path, "problems": settings.problems(),
                                  "settings": _public_settings(settings)})

    @routes.post("/lora_catalog/llm_test")
    async def llm_test(_request):
        from ..lora_catalog.llm import test
        return web.json_response(await off_loop(lambda: test(config.load_settings())))

    return True
