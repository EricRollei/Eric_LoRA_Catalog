"""
Forge / AUTOMATIC1111 sidecars from the cards (run.py export-sidecars). Those UIs read <lora name>.json next to each
LoRA and show it in their LoRA browser: activation text (added to the prompt when the card is clicked), preferred
weight, description and notes. This writes them from card_facets (run.py facets first).

Safety:
  - dry run unless --write: lists what would be written.
  - a sidecar that already exists and was NOT written by this tool is never touched (yours, or another tool's).
  - ours carry "lora_catalog": true (both UIs keep unknown keys); with --overwrite-ours they are refreshed from the
    current cards, without it they are left as they are.
  - LoRA Manager's <name>.metadata.json and Civitai Helper's <name>.civitai.info are different files; not touched.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List

from .config import Settings

MARK = "lora_catalog"


def _sd_version(base_model: str) -> str:
    base = (base_model or "").lower()
    if any(word in base for word in ("sdxl", "pony", "illustrious", "noobai")):
        return "SDXL"
    if "sd 1" in base or "sd1" in base:
        return "SD1"
    if "sd 2" in base or "sd2" in base:
        return "SD2"
    return "Unknown"


def _list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    try:
        parsed = json.loads(value or "[]")
        return [str(v) for v in parsed if str(v).strip()] if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def sidecar_for(row: sqlite3.Row) -> Dict[str, Any]:
    triggers = _list(row["triggers"])
    notes = [f"Role: {row['role']}" if row["role"] else "", f"Base model: {row['base_model']}" if row["base_model"] else ""]
    weight = row["weight_typical"]
    return {
        "description": str(row["effect"] or "").strip(),
        "sd version": _sd_version(row["base_model"]),
        "activation text": ", ".join(triggers[:4]),
        "preferred weight": round(float(weight), 2) if weight not in (None, "") else 0,
        "notes": "\n".join(n for n in notes if n) + "\n(from Eric LoRA Catalog)",
        MARK: True,
    }


def export(settings: Settings, only: str = "", write: bool = False, overwrite_ours: bool = False) -> Dict[str, Any]:
    counts = {"cards": 0, "new": 0, "refreshed": 0, "kept_ours": 0, "not_ours": 0, "missing_file": 0,
              "written": write}
    connection = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    sql = "select path, name, rel_dir, base_model, role, triggers, weight_typical, effect from card_facets"
    params: List[Any] = []
    if only:
        sql += " where replace(rel_dir, '\\', '/') like ?"
        params.append(only.replace("\\", "/").strip("/") + "%")
    rows = connection.execute(sql, params).fetchall()
    connection.close()
    shown = 0
    for row in rows:
        counts["cards"] += 1
        lora = str(row["path"] or "")
        if not os.path.isfile(lora):
            counts["missing_file"] += 1
            continue
        target = os.path.splitext(lora)[0] + ".json"
        existing: Dict[str, Any] = {}
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except (OSError, ValueError):
                existing = {}
            if not isinstance(existing, dict) or not existing.get(MARK):
                counts["not_ours"] += 1
                continue
            if not overwrite_ours:
                counts["kept_ours"] += 1
                continue
            kind = "refreshed"
        else:
            kind = "new"
        data = dict(existing, **sidecar_for(row))
        counts[kind] += 1
        if write:
            temp = target + ".tmp"
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=4, ensure_ascii=False)
            os.replace(temp, target)
        elif shown < 5:
            print(f"  would write {target}: activation '{data['activation text']}', weight {data['preferred weight']}")
            shown += 1
    if not write:
        print("  (dry run: nothing written; add --write to write them)")
    return counts
