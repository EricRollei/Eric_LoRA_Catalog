"""
run.py refresh: bring the catalog up to date after adding LoRAs, on any OS. Every step skips what is already done, so
it is safe to run any time. The same steps the old refresh_new_loras.bat ran:

     1 index        new or changed LoRA files (headers, sidecars, LoRA Manager data)
     2 hash         sha256 of files that have none yet (what Civitai and the duplicate check match on)
     3 identify     Civitai version of new files that came without one, by sha256              (internet)
     4 crawl        Civitai records for LoRAs that have a version id but no local record      (internet)
     5 creators     who made each LoRA (cache, LoRA Manager mirror, then the Civitai API)      (internet)
     6 images       Civitai example images next to LoRAs that have none                       (internet)
     7 community    Civitai audience data for LoRAs not collected in the last 30 days         (internet)
     8 cards        the LLM writes a card for every LoRA that has none                        (LLM, vision)
     9 genre-mood   genre and mood for cards that have none yet                               (LLM, text)
    10 review-all   the HTML review pages
    11 facets       card_facets, the table the LoRA Suggester reads

Internet steps carry on when they fail; any other failed step stops the run. Between 8 and 9 the card model is
unloaded when the text model is a different one (one LLM at a time on the GPU).

Each step runs as its own process (python run.py <step>) with its output appended to DATA_DIR/logs/refresh.log, and
DATA_DIR/logs/refresh_status.json says where the run is (the ComfyUI panel reads it).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

from .config import DATA_DIR, TOOL_DIR, Settings
from .jobs import busy_writers
from .llm import unload

LOG_DIR = os.path.join(DATA_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "refresh.log")
STATUS_PATH = os.path.join(LOG_DIR, "refresh_status.json")
INTERNET = {"identify", "crawl", "creators", "images", "community"}
STEP_NAMES = ["index", "hash", "identify", "crawl", "creators", "images", "community", "cards", "genre-mood",
              "review-all", "facets"]


def plan(workers: int = 3, model: str = "", skip_internet: bool = False, only_steps: str = "") -> List[Tuple[str, List[str]]]:
    steps: List[Tuple[str, List[str]]] = []
    wanted = {s.strip() for s in only_steps.split(",") if s.strip()}
    for name in STEP_NAMES:
        if wanted and name not in wanted:
            continue
        if skip_internet and name in INTERNET:
            continue
        args = [name]
        if name == "cards":
            args += ["--workers", str(max(1, workers))] + (["--model", model] if model else [])
        steps.append((name, args))
    return steps


def _status(**values: Any) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    values["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temp = STATUS_PATH + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(values, handle)
    os.replace(temp, STATUS_PATH)


def read_status() -> Dict[str, Any]:
    try:
        with open(STATUS_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _log(line: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def run(settings: Settings, workers: int = 3, model: str = "", skip_internet: bool = False,
        only_steps: str = "") -> int:
    """0 when every step finished (internet steps may have failed and been skipped), 1 when a step stopped the run."""
    others = busy_writers()
    if others:
        names = ", ".join(f"{o['command']} (since {o['started']})" for o in others)
        print(f"[refresh] another run is going: {names}. Run this again when it has finished.")
        return 1
    steps = plan(workers, model, skip_internet, only_steps)
    card_model = model or settings.card_model
    text_model = settings.genre_model
    _log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  refresh: {', '.join(n for n, _a in steps)} =====")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run_py = os.path.join(TOOL_DIR, "run.py")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for number, (name, args) in enumerate(steps, 1):
        label = f"{number}/{len(steps)} {name}"
        _status(state="running", step=number, of=len(steps), name=name, started=started, pid=os.getpid())
        _log(f"--- {label}  {time.strftime('%H:%M:%S')} ---")
        if name == "genre-mood" and text_model and text_model != card_model and "cards" in dict(steps):
            _log(f"    {unload(settings, card_model)}")
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            # no window: started from the ComfyUI panel the refresh has no console, and Windows would otherwise open
            # a new console window for every step (2026-10-11)
            code = subprocess.call([sys.executable, "-X", "utf8", run_py] + args, stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=TOOL_DIR, env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if code == 0:
            _log(f"[{label}] ok")
            continue
        if name in INTERNET:
            _log(f"[{label}] failed (needs the internet or a Civitai answer); carrying on without it")
            continue
        _log(f"[{label}] FAILED - the steps before it are saved; fix the problem and run the refresh again")
        _status(state="failed", step=number, of=len(steps), name=name, started=started, log=LOG_PATH)
        return 1
    _log(f"===== done {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    _status(state="done", step=len(steps), of=len(steps), name="", started=started, log=LOG_PATH)
    return 0
