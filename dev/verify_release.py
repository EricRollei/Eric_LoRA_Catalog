"""
Checks for the release plumbing: every module imports, settings resolve, the LLM answer repair, the refresh plan,
the lock files, the sidecar format, the Suggester engine and node import from this pack. No LLM, no network, nothing
written to the catalog.

    python dev/verify_release.py
"""

import importlib
import json
import os
import pkgutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


import lora_catalog  # noqa: E402

for module in pkgutil.iter_modules(lora_catalog.__path__):
    if module.name == "__main__":
        continue
    try:
        importlib.import_module("lora_catalog." + module.name)
        check(f"import {module.name}", True)
    except Exception as error:      # noqa: BLE001
        check(f"import {module.name}", False, f"{type(error).__name__}: {error}")

from lora_catalog import config, jobs, llm  # noqa: E402
from lora_catalog.refresh import STEP_NAMES, plan  # noqa: E402
from lora_catalog.sidecars import _sd_version  # noqa: E402

settings = config.load_settings()
check("catalog path resolved", bool(settings.catalog_path), settings.catalog_path)
check("chat url", settings.chat_url().endswith("/chat/completions"), settings.chat_url())
check("text model falls back", bool(settings.genre_model) and bool(settings.slider_model))

blank = config.Settings().finish()
check("blank settings: catalog in data dir", blank.catalog_path.startswith(config.DATA_DIR), blank.catalog_path)
check("blank settings: problem reported", any("No LoRA folders" in p for p in blank.problems()), blank.problems())
check("blank settings: genre model = card model", blank.genre_model == blank.card_model)

check("repair: think + fence + trailing comma", llm.parse_json('<think>x</think>```json\n{"a": 1,}\n```') == {"a": 1})
check("repair: prose around", llm.parse_json('Sure! {"b": [1, 2,],} hope that helps') == {"b": [1, 2]})
check("repair: unterminated think start", llm.parse_json('thinking... </think>{"c": 3}') == {"c": 3})
try:
    llm.parse_json("I thought about it for a long time")
    check("repair: no JSON raises", False)
except ValueError:
    check("repair: no JSON raises", True)
prompted = llm._with_schema_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                                   {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})
check("schema goes into the system prompt", "JSON schema" in prompted[0]["content"] and prompted[1]["content"] == "U")


class Fake:
    llm_backend = "auto"
    llm_base_url = "http://localhost:11434/v1"
    llm_endpoint = ""

    def chat_url(self):
        return self.llm_base_url + "/chat/completions"


check("backend: ollama by port", llm.backend(Fake()) == "ollama")
Fake.llm_base_url = "http://localhost:1234/v1"
check("backend: lmstudio by port", llm.backend(Fake()) == "lmstudio")

steps = [name for name, _args in plan()]
check("refresh: eleven steps in order", steps == STEP_NAMES, steps)
check("refresh: skip internet", "crawl" not in [n for n, _a in plan(skip_internet=True)])
check("refresh: only steps", [n for n, _a in plan(only_steps="cards,facets")] == ["cards", "facets"])
check("refresh: cards workers", plan(workers=2)[7][1][:3] == ["cards", "--workers", "2"], plan(workers=2)[7])

with jobs.job("verify-test"):
    mine = [j for j in jobs.running(exclude_self=False) if j["command"] == "verify-test"]
    check("lock file while running", len(mine) == 1 and mine[0]["pid"] == os.getpid())
    check("own lock excluded by default", not [j for j in jobs.running() if j["command"] == "verify-test"])
check("lock file removed after", not [j for j in jobs.running(exclude_self=False) if j["command"] == "verify-test"])
os.makedirs(jobs.LOCK_DIR, exist_ok=True)
stale = os.path.join(jobs.LOCK_DIR, "cards-999999.json")
with open(stale, "w", encoding="utf-8") as handle:
    json.dump({"command": "cards", "pid": 999999, "started": "x"}, handle)
check("stale lock ignored", not [j for j in jobs.running() if j["pid"] == 999999])
check("stale lock cleaned", not os.path.exists(stale))

check("sd version: SDXL family", _sd_version("Illustrious") == "SDXL" and _sd_version("Pony") == "SDXL")
check("sd version: SD1", _sd_version("SD 1.5") == "SD1")
check("sd version: other", _sd_version("Flux.1 D") == "Unknown")

with tempfile.TemporaryDirectory() as folder:
    shipped = os.path.join(config.SHIPPED_DIR, "vocab.yaml")
    check("shipped vocab exists", os.path.isfile(shipped))

# the Suggester from this pack (engine only; the node needs ComfyUI and is checked by verify_lora_suggest.py)
try:
    suggest = importlib.import_module("suggest")
    s = suggest.load_settings()
    check("suggest imports standalone", True)
    check("suggest uses the catalog's settings", os.path.normcase(s.catalog) == os.path.normcase(settings.catalog_path),
          (s.catalog, settings.catalog_path))
    check("suggest vocab is the catalog's", os.path.isfile(s.vocab), s.vocab)
except Exception as error:      # noqa: BLE001
    check("suggest imports standalone", False, f"{type(error).__name__}: {error}")

print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
