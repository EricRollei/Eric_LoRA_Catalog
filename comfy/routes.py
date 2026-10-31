"""
The LoRA Catalog panel's routes on ComfyUI's own server (web/lora_catalog_panel.js calls them).

    GET  /lora_catalog/status            counts, problems, running jobs, the refresh step, editor up or not
    POST /lora_catalog/job               {"command": "refresh" | "cards" | "genre-mood" | ..., "options": {...}}
    POST /lora_catalog/job/stop          {"pid": 123}  (only a pid that one of our lock files names)
    GET  /lora_catalog/log?name=refresh  the last lines of a job's log
    POST /lora_catalog/editor            start the card editor if it is not running (restart it when it runs older code
                                         than is on disk); returns its URL
    POST /lora_catalog/editor/stop       stop the card editor          POST /lora_catalog/editor/start  start it
    GET  /lora_catalog/settings          the settings the panel edits (the Civitai key only as set / not set)
    POST /lora_catalog/settings          save them (settings.json)
    POST /lora_catalog/llm_test          one small text call and one small image call

Jobs run as their own process (python run.py <command>), started detached: ComfyUI's queue never waits on them, and
restarting ComfyUI does not stop a card pass. The card editor is different (2026-10-17): started from here it gets
ComfyUI's pid and stops by itself when ComfyUI closes, and the panel shows it with a Stop / Start switch - an editor
left running invisibly kept serving old code after updates. Commands and their options come from fixed lists here; nothing the
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
        # Its own hidden console (CREATE_NO_WINDOW): anything the job starts inherits it, so no windows pop up, and it
        # is not ComfyUI's console, so closing ComfyUI does not stop the job. (DETACHED_PROCESS, used before, left the
        # job with no console at all, and Windows opened a visible one for every step it started.)
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
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


# -- the card editor switch (2026-10-17) ---------------------------------------------------------------------------
def _editor_health(port: int = EDITOR_PORT) -> Dict[str, Any]:
    """{pid, started, code, parent} from a running editor; {} when none answers or it is older than /api/health."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, dict) and data.get("ok") else {}
    except Exception:
        return {}


def _disk_code() -> float:
    folder = os.path.join(config.TOOL_DIR, "lora_catalog")
    stamps = [os.path.getmtime(os.path.join(folder, n)) for n in os.listdir(folder) if n.endswith(".py")]
    return max(stamps) if stamps else 0.0


def _editor_state() -> Dict[str, Any]:
    """up / pid / since / stale (runs older code than is on disk) / tied (stops with this ComfyUI) / legacy (an editor
    from before 2026-10-17: no /api/health, no Stop - it is restarted when you open it)."""
    up = _editor_up()
    if not up:
        return {"up": False}
    health = _editor_health()
    if not health:
        return {"up": True, "legacy": True, "stale": True, "pid": _pid_on_port(EDITOR_PORT)}
    return {"up": True, "pid": health.get("pid"), "since": time.strftime("%Y-%m-%d %H:%M", time.localtime(health.get("started") or 0)),
            "stale": float(health.get("code") or 0) + 1 < _disk_code(), "tied": health.get("parent") == os.getpid()}


def _pid_on_port(port: int) -> int:
    """The pid listening on 127.0.0.1:port (netstat), 0 when none or not found."""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"] if os.name == "nt" else ["netstat", "-ltnp"],
                             capture_output=True, text=True, timeout=10, creationflags=flags).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}") and "LISTEN" in line.upper():
                return int(parts[-1].split("/")[0]) if parts[-1].split("/")[0].isdigit() else 0
    except Exception:
        return 0
    return 0


def _is_editor_process(pid: int) -> bool:
    """Only ever end a process that is this tool's card editor (its command line runs run.py serve)."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
                                 capture_output=True, text=True, timeout=15,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        else:
            out = open(f"/proc/{int(pid)}/cmdline", "rb").read().decode("utf-8", "replace").replace("\0", " ")
    except Exception:
        return False
    return "run.py" in out and " serve" in out


def _editor_stop() -> str:
    state = _editor_state()
    if not state.get("up"):
        return "the card editor is not running"
    pid = int(state.get("pid") or 0)
    if not state.get("legacy"):
        try:                                    # ask it to stop: it answers, then closes its port
            request = urllib.request.Request(f"http://127.0.0.1:{EDITOR_PORT}/api/shutdown", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=3).read()
        except Exception:
            pass
        for _ in range(20):
            time.sleep(0.25)
            if not _editor_up():
                return f"card editor stopped (pid {pid})"
    if _is_editor_process(pid):                 # an old editor without Stop, or one that did not close
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.25)
            if not _editor_up():
                return f"card editor stopped (pid {pid})"
    return f"could not stop what answers on port {EDITOR_PORT} (pid {pid or '?'}) - it is not this tool's editor"


def _editor_start() -> bool:
    _start(["serve", "--no-browser", "--port", str(EDITOR_PORT), "--parent-pid", str(os.getpid())], "editor")
    for _ in range(40):
        time.sleep(0.5)
        if _editor_up():
            return True
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
            state = _editor_state()
            return {"problems": settings.problems(), "counts": _counts(settings), "running": jobs.running(),
                    "refresh": read_status(), "editor": bool(state.get("up")), "editor_state": state,
                    "editor_url": f"http://127.0.0.1:{EDITOR_PORT}/"}
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
        state = await off_loop(_editor_state)
        note = ""
        if state.get("up") and state.get("stale"):     # older code than is on disk: a fresh one (edits are saved as you go)
            note = await off_loop(_editor_stop) + ", restarted with the current code"
            state = await off_loop(_editor_state)
        if not state.get("up"):
            await off_loop(_editor_start)
        return web.json_response({"url": url, "up": await off_loop(_editor_up), "note": note})

    @routes.post("/lora_catalog/editor/stop")
    async def editor_stop(_request):
        return web.json_response({"result": await off_loop(_editor_stop)})

    @routes.post("/lora_catalog/editor/start")
    async def editor_start(_request):
        if await off_loop(_editor_up):
            return web.json_response({"result": "already running", "up": True})
        up = await off_loop(_editor_start)
        return web.json_response({"result": "card editor started" if up else "did not start - see the editor log", "up": up})

    @routes.get("/lora_catalog/settings")
    async def get_settings(_request):
        return web.json_response(await off_loop(lambda: _public_settings(config.load_settings())))

    @routes.post("/lora_catalog/settings")
    async def save(request):
        # Guard (2026-10-15): after a hot reload (LG_HotReload) a GET for this path was seen landing here and failing on
        # the empty body (JSONDecodeError, shown as a 500 under Settings). A GET, or a request without a body, just
        # gets the settings, the same answer the GET route gives.
        if request.method != "POST" or not request.can_read_body:
            return web.json_response(await off_loop(lambda: _public_settings(config.load_settings())))
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
