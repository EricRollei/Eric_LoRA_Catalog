"""
Your verdicts on LoRA combinations: good / bad / unsure, marked by eye on a sweep's cells (2026-10-09).
Spec: Eric_Krea2 docs/spec_lora_stack_sweeps_2026-10-09.md (section 5: click-to-mark, judged by eye).

    Sweeps page in the card editor (http://127.0.0.1:8765/sweeps) - pick a sweep, press G / B / U on each cell.
    python run.py stacks                  what you have marked, good and bad, by stack
    python run.py stacks --pairs          the same folded into LoRA pairs (what the Suggester will use)

Each mark reads the cell PNG's krea2_settings chunk - the exact stack (paths, per-stage weights), the prompt the cell
used - and the seed / checkpoint from the sweep manifest, and stores them in the catalog:

    stacks   id, verdict (good|bad|unsure), source (sweep|render), sweep_id, cell_file, image_path, checkpoint,
             base_model, prompt, seed, loras (JSON: name, path, sha256, weights [s1, s2, s3]), stack_key (sorted
             sha256@weights - the same stack from two sweeps has the same key), n_loras, notes, marked_at

Only the LoRAs the cell swept are counted as "the stack"; a LoRA from your panel stack that stays in every cell
(keep_panel_stack) is stored too but flagged panel=true, so a pair is never credited to a LoRA that was always there.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import Settings

TABLE = """
CREATE TABLE IF NOT EXISTS stacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, verdict TEXT, source TEXT, sweep_id TEXT, cell_file TEXT,
    image_path TEXT, checkpoint TEXT, base_model TEXT, prompt TEXT, seed INTEGER, loras TEXT, stack_key TEXT,
    n_loras INTEGER, notes TEXT, marked_at TEXT, UNIQUE (sweep_id, cell_file));
CREATE INDEX IF NOT EXISTS stacks_key ON stacks(stack_key);
"""
VERDICTS = ("good", "bad", "unsure")
SWEEP_ID = re.compile(r"^[\w.-]+$")


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(TABLE)


class Sweeps:
    """Reads sweep folders (manifest + cell PNGs) and keeps the marks. Shared by the editor's request threads."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.sweeps_dir
        self.lock = threading.Lock()
        self.thumbs: Dict[tuple, bytes] = {}
        with self.connect() as connection:
            ensure(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.catalog_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    # -- reading sweeps ---------------------------------------------------------------------------------------------
    def folder(self, sweep_id: str) -> str:
        if not SWEEP_ID.match(sweep_id or ""):
            raise ValueError("bad sweep id")
        path = os.path.join(self.root, sweep_id)
        if not os.path.isfile(os.path.join(path, "sweep_manifest.json")):
            raise ValueError(f"no sweep {sweep_id}")
        return path

    def manifest(self, sweep_id: str) -> Dict[str, Any]:
        with open(os.path.join(self.folder(sweep_id), "sweep_manifest.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list(self) -> List[Dict[str, Any]]:
        """Sweep folders, newest first, with their LoRA mode and how many cells you have marked."""
        out = []
        if not os.path.isdir(self.root):
            return out
        with self.connect() as connection:
            marked = {row[0]: row[1] for row in connection.execute(
                "select sweep_id, count(*) from stacks where source = 'sweep' group by sweep_id")}
        for name in sorted(os.listdir(self.root), reverse=True):
            path = os.path.join(self.root, name, "sweep_manifest.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, ValueError):
                continue
            lora = (manifest.get("plan") or {}).get("lora") or {}
            cells = manifest.get("cells") or []
            out.append({"id": name, "created": manifest.get("created", ""), "checkpoint": manifest.get("checkpoint", ""),
                        "lora_mode": lora.get("mode") or "", "cells": len(cells),
                        "ok": sum(1 for c in cells if c.get("file")), "marked": marked.get(name, 0),
                        "incomplete": bool(manifest.get("incomplete"))})
        return out

    def _names(self, connection: sqlite3.Connection) -> Dict[str, str]:
        """file name / stem / folder path -> the model name, for readable captions (sweep lines use file names)."""
        names: Dict[str, str] = {}
        for row in connection.execute("select path, root, file_name, stem, model_name from loras"):
            name = row["model_name"] or row["stem"]
            keys = [row["file_name"], row["stem"], row["path"]]
            if row["root"]:
                keys.append(os.path.relpath(row["path"], row["root"]))
            for key in keys:
                if key:
                    names.setdefault(str(key).lower(), name)
        return names

    def display(self, text: str, names: Dict[str, str]) -> str:
        """'3209626_elegance_niji_kr2 @1 | 3197181_kreafineart_e10 @0.6' -> 'Elegance Niji @1 | KreaFineArt @0.6'."""
        parts = []
        for part in str(text).split(" | "):
            name, _, weight = part.rpartition(" @")
            if not name:
                name, weight = part, ""
            pretty = names.get(name.strip().lower(), name.strip())
            parts.append(f"{pretty} @{weight}" if weight else pretty)
        return " | ".join(parts)

    def cells(self, sweep_id: str) -> Dict[str, Any]:
        manifest = self.manifest(sweep_id)
        with self.connect() as connection:
            marks = {row["cell_file"]: dict(row) for row in connection.execute(
                "select cell_file, verdict, notes from stacks where sweep_id = ?", (sweep_id,))}
            names = self._names(connection)
        cells = []
        for cell in manifest.get("cells") or []:
            overrides = cell.get("overrides") or {}
            kind = ("stack" if " | " in str(overrides.get("lora_stack", "")) else
                    "single" if "lora_stack" in overrides or "lora_file" in overrides else
                    "baseline" if overrides.get("lora_baseline") else "")
            mark = marks.get(cell.get("file") or "", {})
            cells.append({"index": cell.get("index"), "file": cell.get("file"), "label": cell.get("label", ""),
                          "error": cell.get("error"), "seed": cell.get("seed"), "kind": kind,
                          "stack": overrides.get("lora_stack") or overrides.get("lora_file") or "",
                          "display": self.display(overrides.get("lora_stack") or overrides.get("lora_file") or "", names),
                          "mult": overrides.get("lora_mult"), "prompt": overrides.get("prompt", ""),
                          "clip": (cell.get("metrics") or {}).get("clip_pct"),
                          "verdict": mark.get("verdict"), "notes": mark.get("notes") or ""})
        return {"id": sweep_id, "checkpoint": manifest.get("checkpoint", ""), "created": manifest.get("created", ""),
                "lora": (manifest.get("plan") or {}).get("lora"), "cells": cells}

    def image(self, sweep_id: str, file_name: str, side: int) -> Optional[bytes]:
        """A cell PNG as a JPEG of at most `side` px (0 = full size). Only files the manifest lists."""
        manifest = self.manifest(sweep_id)
        if file_name not in {c.get("file") for c in manifest.get("cells") or []} and file_name != "contact_sheet.png":
            return None
        key = (sweep_id, file_name, side)
        if key in self.thumbs:
            return self.thumbs[key]
        from PIL import Image
        with Image.open(os.path.join(self.folder(sweep_id), file_name)) as opened:
            image = opened.convert("RGB")
        if side:
            image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88 if side else 93)
        data = buffer.getvalue()
        with self.lock:
            if len(self.thumbs) > 400:
                self.thumbs.clear()
            self.thumbs[key] = data
        return data

    # -- marking ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _chunk(png_path: str) -> Dict[str, Any]:
        from PIL import Image
        with Image.open(png_path) as opened:
            info = dict(opened.info)
        try:
            return json.loads(info.get("krea2_settings") or "{}")
        except ValueError:
            return {}

    def _lora_facts(self, connection: sqlite3.Connection, path: str) -> Dict[str, Any]:
        row = connection.execute("select sha256, model_name, stem, arch_folder from loras where path = ?",
                                 (path,)).fetchone()
        if row is None:            # moved since the sweep: path_history knows where it went
            moved = connection.execute("select new_path from path_history where old_path = ? and undone = 0 "
                                       "order by moved_at desc limit 1", (path,)).fetchone() \
                if connection.execute("select 1 from sqlite_master where name = 'path_history'").fetchone() else None
            if moved:
                row = connection.execute("select sha256, model_name, stem, arch_folder from loras where path = ?",
                                         (moved[0],)).fetchone()
        if row is None:
            return {"sha256": None, "name": os.path.splitext(os.path.basename(path))[0], "arch": ""}
        return {"sha256": (row["sha256"] or "").lower() or None, "name": row["model_name"] or row["stem"],
                "arch": row["arch_folder"] or ""}

    def mark(self, sweep_id: str, file_name: str, verdict: Optional[str], notes: Optional[str] = None) -> Dict[str, Any]:
        """Record (or with verdict None, remove) your verdict on one cell."""
        manifest = self.manifest(sweep_id)
        cell = next((c for c in manifest.get("cells") or [] if c.get("file") == file_name), None)
        if cell is None:
            return {"ok": False, "error": "no such cell"}
        with self.connect() as connection:
            if not verdict:
                connection.execute("delete from stacks where sweep_id = ? and cell_file = ?", (sweep_id, file_name))
                return {"ok": True, "verdict": None}
            if verdict not in VERDICTS:
                return {"ok": False, "error": f"verdict must be one of {VERDICTS}"}
            image_path = os.path.join(self.folder(sweep_id), file_name)
            chunk = self._chunk(image_path)
            loras = []
            for entry in chunk.get("lora") or []:
                facts = self._lora_facts(connection, str(entry.get("path") or ""))
                weights = [entry.get("weight_s1", entry.get("strength")), entry.get("weight_s2", entry.get("strength")),
                           entry.get("weight_s3", entry.get("strength"))]
                loras.append({"name": facts["name"], "path": entry.get("path"), "sha256": facts["sha256"],
                              "weights": [round(float(w), 4) if w is not None else None for w in weights],
                              "panel": not str(entry.get("adapter_name", "")).startswith("sweep_"), "arch": facts["arch"]})
            swept = [lora for lora in loras if not lora["panel"]]
            key = " + ".join(sorted(f"{lora['sha256'] or lora['name']}@{'/'.join(f'{w:g}' for w in lora['weights'] if w is not None)}"
                                    for lora in swept))
            ultra = chunk.get("ultra") or {}
            prompt = chunk.get("sweep_prompt") or ultra.get("prompt") or (manifest.get("base") or {}).get("prompt") or ""
            base_model = next((lora["arch"] for lora in loras if lora["arch"]), "")
            connection.execute("""
                insert into stacks (verdict, source, sweep_id, cell_file, image_path, checkpoint, base_model, prompt, seed,
                                    loras, stack_key, n_loras, notes, marked_at)
                values (?, 'sweep', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict (sweep_id, cell_file) do update set verdict = excluded.verdict,
                    notes = coalesce(excluded.notes, stacks.notes), marked_at = excluded.marked_at""",
                (verdict, sweep_id, file_name, image_path, manifest.get("checkpoint", ""), base_model, prompt,
                 cell.get("seed"), json.dumps(loras, ensure_ascii=False), key, len(swept), notes,
                 datetime.now().isoformat(timespec="seconds")))
        return {"ok": True, "verdict": verdict, "n_loras": len(swept)}

    # -- reading the verdicts back ----------------------------------------------------------------------------------
    def summary(self) -> List[Dict[str, Any]]:
        """One row per distinct stack (same LoRAs at the same weights), with good / bad / unsure counts."""
        groups: Dict[str, Dict[str, Any]] = {}
        with self.connect() as connection:
            for row in connection.execute("select * from stacks where n_loras > 0 order by marked_at"):
                group = groups.setdefault(row["stack_key"], {
                    "stack": " + ".join(f"{l['name']} @{'/'.join(f'{w:g}' for w in l['weights'] if w is not None)}"
                                        for l in json.loads(row["loras"]) if not l["panel"]),
                    "n_loras": row["n_loras"], "good": 0, "bad": 0, "unsure": 0, "sweeps": set()})
                group[row["verdict"]] = group.get(row["verdict"], 0) + 1
                group["sweeps"].add(row["sweep_id"])
        out = []
        for key, group in groups.items():
            group["sweeps"] = len(group["sweeps"])
            group["key"] = key
            out.append(group)
        out.sort(key=lambda g: (-g["n_loras"], -(g["good"] - g["bad"]), g["stack"].lower()))
        return out

    def pairs(self) -> List[Dict[str, Any]]:
        """Every two swept LoRAs that appeared together in a marked cell: good / bad counts (sha256 keyed)."""
        counts: Dict[tuple, Dict[str, Any]] = {}
        with self.connect() as connection:
            for row in connection.execute("select verdict, loras from stacks where n_loras > 1"):
                swept = [l for l in json.loads(row["loras"]) if not l["panel"]]
                for i, a in enumerate(swept):
                    for b in swept[i + 1:]:
                        ka, kb = sorted([(a["sha256"] or a["name"], a["name"]), (b["sha256"] or b["name"], b["name"])])
                        pair = counts.setdefault((ka[0], kb[0]), {"a": ka[1], "b": kb[1], "good": 0, "bad": 0,
                                                                   "unsure": 0})
                        pair[row["verdict"]] += 1
        return sorted(counts.values(), key=lambda p: (-(p["good"] - p["bad"]), p["a"].lower()))


def report(settings: Settings, pairs: bool = False) -> None:
    sweeps = Sweeps(settings)
    rows = sweeps.pairs() if pairs else sweeps.summary()
    if not rows:
        print("[stacks] nothing marked yet - open the Sweeps page in the card editor")
        return
    for row in rows:
        tally = f"good {row['good']}  bad {row['bad']}  unsure {row['unsure']}"
        print(f"  {tally:30s} {row['a'] + '  +  ' + row['b'] if pairs else row['stack']}")
