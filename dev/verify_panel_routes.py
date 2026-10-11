"""
Checks for comfy/routes.py without ComfyUI: option whitelisting, the status counts against a catalog (read only),
log tailing. Loads the pack as a package from its parent folder, the way ComfyUI does.

    python dev/verify_panel_routes.py
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
PACKAGE = os.path.basename(ROOT)

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


routes = importlib.import_module(f"{PACKAGE}.comfy.routes")
config = importlib.import_module(f"{PACKAGE}.lora_catalog.config")

check("refresh args", routes._args("refresh", {"skip_internet": True, "workers": 2}) ==
      ["refresh", "--skip-internet", "--workers", "2"], routes._args("refresh", {"skip_internet": True, "workers": 2}))
check("unknown option dropped", routes._args("facets", {"rm": "-rf"}) == ["facets"])
check("workers clamped", routes._args("cards", {"workers": 999}) == ["cards", "--workers", "16"])
try:
    routes._args("cards", {"only": "--force"})
    check("flag-looking text refused", False)
except ValueError:
    check("flag-looking text refused", True)
check("bool false leaves flag out", routes._args("cards", {"force": False}) == ["cards"])
check("sidecars write flag", routes._args("export-sidecars", {"write": True}) == ["export-sidecars", "--write"])

settings = config.load_settings()
counts = routes._counts(settings)
print("  counts:", counts)
check("counts read", isinstance(counts.get("loras"), int) or "catalog" in counts, counts)
check("tail of a missing log is empty", routes._tail("no-such-log", 10) == "")
check("tail name is sanitised", routes._tail("../../secrets", 10) == "")
public = routes._public_settings(settings)
check("public settings never carry the key", "civitai_api_key" not in public and "civitai_key_set" in public)

# the node class from the pack
node_module = importlib.import_module(f"{PACKAGE}.nodes.lora_suggest_node")
node = node_module.LoRASuggester
inputs = node.INPUT_TYPES()
choices = inputs["required"]["platform"][0]
check("platform list keeps every old id", set(node_module.LEGACY_PLATFORM_IDS) <= set(choices))
check("platform list starts with generic", choices[0] == "generic")
check("outputs unchanged + lora_text last", node.RETURN_NAMES == ("suggestions", "triggers", "prompt_with_triggers",
                                                                  "report", "lora_text"))
optional = list(inputs["optional"])
check("optional input order unchanged", optional == ["sections", "text", "exclude", "pin", "transform_max", "slider_max",
                                                    "slider_range", "slider_min_strength"], optional)
bundle = node_module._Sections(type("B", (), {"sections": {"style": "ukiyo-e", "x": ""}, "nsfw": 2})())
check("sections bundle read by attribute", bundle.sections == {"style": "ukiyo-e"} and bundle.nsfw == 2)
check("sections from dict", node_module._Sections({"subject": "a knight"}).sections == {"subject": "a knight"})
check("sections from nothing", node_module._Sections(None).sections == {})

print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
