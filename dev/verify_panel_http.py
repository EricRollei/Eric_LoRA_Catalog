"""
The panel's routes over real HTTP, without ComfyUI: a stand-in PromptServer with aiohttp routes, then GET / POST the
read-only endpoints (status, settings, log) and one refused job. Starts no jobs, writes nothing.

    python dev/verify_panel_http.py        (needs aiohttp, which ComfyUI's Python has)
"""

import asyncio
import importlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
PACKAGE = os.path.basename(ROOT)

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

routes = web.RouteTableDef()
fake = types.ModuleType("server")
fake.PromptServer = type("PromptServer", (), {})
fake.PromptServer.instance = types.SimpleNamespace(routes=routes)
sys.modules["server"] = fake

module = importlib.import_module(f"{PACKAGE}.comfy.routes")
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


async def main():
    check("register", module.register() is True)
    app = web.Application()
    app.add_routes(routes)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/lora_catalog/status")
        data = await response.json()
        check("status 200", response.status == 200, response.status)
        check("status has counts and running", "counts" in data and "running" in data, list(data))
        print("  status:", {k: data[k] for k in ("counts", "problems", "editor")})
        response = await client.get("/lora_catalog/settings")
        data = await response.json()
        check("settings 200", response.status == 200)
        check("settings never return the key", "civitai_api_key" not in data and "civitai_key_set" in data, list(data))
        response = await client.get("/lora_catalog/log?name=refresh&lines=5")
        check("log 200", response.status == 200)
        response = await client.post("/lora_catalog/job", json={"command": "rm -rf"})
        check("unknown command refused", response.status == 400, response.status)
        response = await client.post("/lora_catalog/job/stop", json={"pid": 4})
        data = await response.json()
        check("stop refuses a pid that is not ours", "not one of" in data.get("result", ""), data)


asyncio.run(main())
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
