"""
Eric LoRA Catalog for ComfyUI

    LoRA Suggester node     picks LoRAs for a prompt from your catalog (roles, stage weights, sliders, NSFW gate)
    LoRA Catalog panel      sidebar tab: refresh the catalog, card new LoRAs, open the card editor, settings

The catalog itself (lora_catalog/) also runs without ComfyUI: launch.bat / launch.sh open the card editor,
refresh.bat / refresh.sh bring the catalog up to date. See README.md.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial CC BY-NC 4.0 and Commercial Use). See LICENSE.
"""

import sys

if sys.version_info < (3, 12):
    print(f"[Eric LoRA Catalog] needs Python 3.12 or newer (this is {sys.version.split()[0]}); not loaded")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    from .nodes.lora_suggest_node import LoRASuggester

    try:
        from .comfy.routes import register as _register_routes
        if not _register_routes():
            print("[Eric LoRA Catalog] panel routes not registered (not running inside ComfyUI's server)")
    except Exception as _error:      # the node still works without the panel
        print(f"[Eric LoRA Catalog] panel routes not registered: {_error}")

    NODE_CLASS_MAPPINGS = {
        "EricLoRASuggester": LoRASuggester,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "EricLoRASuggester": "LoRA Suggester",
    }

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
