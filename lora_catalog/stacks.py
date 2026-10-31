"""
Your marks on sweep cells - stars, praise / flaw tags, a strength push - marked by eye (2026-10-09, reworked 2026-10-17).
Spec: Eric_Krea2 docs/spec_lora_stack_sweeps_2026-10-09.md (section 5: click-to-mark, judged by eye).

    Sweeps page in the card editor (http://127.0.0.1:8765/sweeps) - pick a sweep, press 1-5 on each cell.
    python run.py stacks                  what you have marked, by stack
    python run.py stacks --pairs          the same folded into LoRA pairs
    python run.py stacks --update         fill in prompt fit / type for older marks and refit the model (markmodel.py)

Each mark reads the cell PNG's krea2_settings chunk - the exact stack (paths, per-stage weights), the prompt the cell
used - and the seed / checkpoint from the sweep manifest, and stores them in the catalog:

    stacks   id, verdict (good|bad|unsure - from the stars since 2026-10-17), source (sweep|render), sweep_id,
             cell_file, image_path, checkpoint, base_model, prompt, seed, loras (JSON: name, path, sha256, weights
             [s1, s2, s3]), stack_key (sorted sha256@weights - the same stack from two sweeps has the same key),
             n_loras, notes, marked_at, stars, tags, subject, words, word_mode, sampler, steps, aspects,
             push (stronger|weaker: the strength you would try next), prompt_base (the prompt without the LoRA words),
             prompt_type (portrait / figure / creature / scene / object / abstract), fit (JSON {lora key: prompt fit})

The rating is the stars: how well the LoRA did its job on this prompt, judged against the baseline cell. The verdict
column is kept, filled from the stars (4-5 good, 3 unsure, 1-2 bad), so stack counts, pairs and the share sheets read
as before; old marks made with G / B / U and no stars keep their verdict (score 4 / 3 / 2). Tags say why: FLAWS (the
LoRA's fault) and PRAISE. Baseline cells (no swept LoRA) can be rated too: that measures how hard the prompt is.

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
from .sweepfiles import manifest_path

TABLE = """
CREATE TABLE IF NOT EXISTS stacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, verdict TEXT, source TEXT, sweep_id TEXT, cell_file TEXT,
    image_path TEXT, checkpoint TEXT, base_model TEXT, prompt TEXT, seed INTEGER, loras TEXT, stack_key TEXT,
    n_loras INTEGER, notes TEXT, marked_at TEXT, UNIQUE (sweep_id, cell_file));
CREATE INDEX IF NOT EXISTS stacks_key ON stacks(stack_key);
"""
VERDICTS = ("good", "bad", "unsure")
SWEEP_ID = re.compile(r"^[\w.-]+$")

# 2026-10-16: stars, why-tags and the cell's context, so a LoRA's record can be split by checkpoint, subject, weight
# and word mode. 2026-10-17: the strength push, the prompt without the LoRA words, its type and the prompt fit.
ADDED = {"stars": "INTEGER", "tags": "TEXT", "subject": "TEXT", "words": "TEXT", "word_mode": "TEXT",
         "sampler": "TEXT", "steps": "INTEGER", "aspects": "TEXT", "push": "TEXT", "prompt_base": "TEXT",
         "prompt_type": "TEXT", "fit": "TEXT"}

# Why-tags (2026-10-17). FLAWS are the LoRA's fault; PRAISE what it does well. Each flaw has a cause: "training" shows
# whatever the prompt (a badly made LoRA), "fit" depends on the prompt (outside its range), "strength" on the weight.
# Keys (2026-10-30, Eric's set): one letter per button in the Sweep Ratings zoom; the letter is highlighted in the label.
FLAWS = [
    ("signature or watermark", "training", "s"), ("unwanted text or lettering", "training", "t"),
    ("added frame or border", "training", "r"), ("changes the character or face", "training", "f"),
    ("changes the subject or scene", "fit", ""), ("changes pose or composition", "fit", ""),
    ("ignores the prompt", "fit", "i"), ("same face every time", "training", "e"),
    ("faces or hands broken", "training", "h"), ("anatomy wrong", "training", "a"),
    ("artifacts, noise or banding", "training", "x"), ("overcooked or too strong", "strength", "o"),
    ("too weak", "strength", ""), ("style didn't show", "fit", "w"), ("colour cast", "training", ""),
    ("oversaturated", "training", "v"), ("washed out or flat", "training", ""), ("soft or blurry", "training", "b"),
    ("unwanted NSFW", "training", "n"),
]
PRAISE = ["great colour", "great detail", "great composition", "great lighting", "great texture or brushwork",
          "true to its style", "keeps the subject", "follows the prompt", "strong mood", "clean and coherent"]
PRAISE_KEYS = {"great colour": "c", "great detail": "d", "great composition": "g", "great lighting": "l",
               "great texture or brushwork": "u", "keeps the subject": "k", "follows the prompt": "p",
               "strong mood": "m"}
FLAW_NAMES = [f[0] for f in FLAWS]
FLAW_CAUSE = {f[0]: f[1] for f in FLAWS}
TAG_KEYS = {f[2]: f[0] for f in FLAWS if f[2]}
TAGS = FLAW_NAMES + PRAISE
TAG_RENAME = {"too strong": "overcooked or too strong", "artifacts": "artifacts, noise or banding",
              "face/hands broken": "faces or hands broken", "changes the subject": "changes the subject or scene"}
PUSHES = ("stronger", "weaker")
STAR_VERDICT = {1: "bad", 2: "bad", 3: "unsure", 4: "good", 5: "good"}
VERDICT_SCORE = {"good": 4.0, "unsure": 3.0, "bad": 2.0}     # old marks without stars


def _cell_settings(cell: Dict[str, Any], varying: set) -> List[Dict[str, Any]]:
    """The settings line under a zoomed sweep cell (2026-10-30): per stage that ran - sampler, schedule, steps, the
    hybrid handoff, eta, cfg - then size, churn form, shift and seed. vary = differs between cells of this sweep."""
    r = cell.get("resolved") or {}
    if not r:
        return []
    out: List[Dict[str, Any]] = []

    def add(key: str, text: str) -> None:
        out.append({"t": text, "vary": key in varying})

    def num(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    stages = ["s1"]
    if not cell.get("quick_screen"):
        ran = cell.get("trajectory") or {}
        stages += [f"s{n}" for n in (2, 3) if num(r.get(f"upscale_to_stage{n}")) > 0 or f"s{n}" in ran]
    any_eta = False
    for st in stages:
        out.append({"t": st.upper(), "stage": True})
        sampler = str(r.get(f"{st}_sampler", ""))
        add(f"{st}_sampler", sampler)
        add(f"{st}_schedule", str(r.get(f"{st}_schedule", "")))
        add(f"{st}_steps", f"{r.get(f'{st}_steps')} steps")
        if sampler.startswith("lcm_hybrid"):
            handoff = num(r.get(f"{st}_hybrid_handoff"))
            if handoff > 0:
                add(f"{st}_hybrid_handoff", f"handoff \u03c3{handoff:.2f}")
            else:
                add(f"{st}_hybrid_steps", f"hybrid {r.get(f'{st}_hybrid_steps') or 'auto'}")
        eta = num(r.get(f"{st}_eta"))
        if eta > 0:
            any_eta = True
            add(f"{st}_eta", f"eta {eta:g}")
        cfg = num(r.get(f"{st}_cfg"))
        if cfg:
            add(f"{st}_cfg", f"cfg {cfg:g}")
    out.append({"t": "", "stage": True})
    add("s1_megapixels", f"{r.get('s1_megapixels')} MP" + (" (quick screen)" if cell.get("quick_screen") else ""))
    if any_eta and r.get("churn_form"):
        add("churn_form", f"churn {r.get('churn_form')}")
    if r.get("distilled_shift"):
        add("distilled_shift", f"shift {r.get('distilled_shift')}")
    add("seed", f"seed {cell.get('seed') if cell.get('seed') is not None else r.get('seed')}")
    return out


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(TABLE)
    have = {r[1] for r in connection.execute("pragma table_info(stacks)")}
    for name, kind in ADDED.items():
        if name not in have:
            connection.execute(f"alter table stacks add column {name} {kind}")
    # the 2026-10-16 tag names -> the 2026-10-17 list (once; a row is only touched when it holds an old name)
    for row in connection.execute("select id, tags from stacks where tags like '%too strong%' or tags like '%artifacts\"%'"
                                  " or tags like '%face/hands%' or tags like '%changes the subject\"%'").fetchall():
        tags = [TAG_RENAME.get(t, t) for t in json.loads(row[1] or "[]")]
        connection.execute("update stacks set tags = ? where id = ?", (json.dumps(tags, ensure_ascii=False), row[0]))


def normalize_tags(tags: List[str]) -> List[str]:
    out = []
    for t in tags:
        t = TAG_RENAME.get(str(t).strip(), str(t).strip())
        if t and t not in out:
            out.append(t)
    return out


def score_of(row) -> Optional[float]:
    """Stars when given, else an old verdict as 4 / 3 / 2; None for a mark with tags or a push only."""
    if row["stars"]:
        return float(row["stars"])
    return VERDICT_SCORE.get(row["verdict"] or "")


class Sweeps:
    """Reads sweep folders (manifest + cell PNGs) and keeps the marks. Shared by the editor's request threads."""

    MODEL_DELAY = 8.0           # seconds after the last mark before the model is refitted

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.sweeps_dir
        self.lock = threading.Lock()
        self.thumbs: Dict[tuple, bytes] = {}
        self._fit = None
        self._timer: Optional[threading.Timer] = None
        with self.connect() as connection:
            ensure(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.catalog_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @property
    def fit_index(self):
        if self._fit is None:
            from .markfit import FitIndex
            self._fit = FitIndex(self.settings.catalog_path)
        return self._fit

    # -- reading sweeps ---------------------------------------------------------------------------------------------
    def folder(self, sweep_id: str) -> str:
        if not SWEEP_ID.match(sweep_id or ""):
            raise ValueError("bad sweep id")
        path = os.path.join(self.root, sweep_id)
        if not manifest_path(path):
            raise ValueError(f"no sweep {sweep_id}")
        return path

    def manifest(self, sweep_id: str) -> Dict[str, Any]:
        with open(manifest_path(self.folder(sweep_id)), "r", encoding="utf-8") as handle:
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
            path = manifest_path(os.path.join(self.root, name)) if os.path.isdir(os.path.join(self.root, name)) else None
            if not path:
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
            base, hashed, option = name.strip().partition(" #")     # 'name #word option' (2026-10-20)
            pretty = names.get(base.strip().lower(), base.strip()) + (f" #{option.strip()}" if hashed else "")
            parts.append(f"{pretty} @{weight}" if weight else pretty)
        return " | ".join(parts)

    def cells(self, sweep_id: str) -> Dict[str, Any]:
        manifest = self.manifest(sweep_id)
        with self.connect() as connection:
            ensure(connection)
            marks = {row["cell_file"]: dict(row) for row in connection.execute(
                "select cell_file, verdict, notes, stars, tags, subject, push, prompt_type, fit from stacks "
                "where sweep_id = ?", (sweep_id,))}
            names = self._names(connection)
        cells = []
        # settings that differ between this sweep's cells (2026-10-30) - highlighted in the zoom's settings line
        seen_values: Dict[str, set] = {}
        for cell in manifest.get("cells") or []:
            for key, value in (cell.get("resolved") or {}).items():
                seen_values.setdefault(key, set()).add(json.dumps(value, sort_keys=True, default=str))
            seen_values.setdefault("seed", set()).add(str(cell.get("seed")))
        varying = {k for k, v in seen_values.items() if len(v) > 1}
        for cell in manifest.get("cells") or []:
            overrides = cell.get("overrides") or {}
            kind = ("stack" if " | " in str(overrides.get("lora_stack", "")) else
                    "single" if "lora_stack" in overrides or "lora_file" in overrides else
                    "baseline" if overrides.get("lora_baseline") else "")
            mark = marks.get(cell.get("file") or "", {})
            fits = {}
            try:
                fits = json.loads(mark.get("fit") or "{}")
            except ValueError:
                fits = {}
            fit_values = [f.get("fit") for f in fits.values() if isinstance(f, dict) and f.get("fit") is not None]
            cells.append({"index": cell.get("index"), "file": cell.get("file"), "label": cell.get("label", ""),
                          "error": cell.get("error"), "seed": cell.get("seed"), "kind": kind,
                          "stack": overrides.get("lora_stack") or overrides.get("lora_file") or "",
                          "display": self.display(overrides.get("lora_stack") or overrides.get("lora_file") or "", names),
                          "mult": overrides.get("lora_mult"), "prompt": overrides.get("prompt", ""),
                          "clip": (cell.get("metrics") or {}).get("clip_pct"),
                          "verdict": mark.get("verdict"), "notes": mark.get("notes") or "",
                          "stars": mark.get("stars"), "tags": normalize_tags(json.loads(mark["tags"])) if mark.get("tags") else [],
                          "subject": mark.get("subject") or "", "push": mark.get("push") or "",
                          "ptype": mark.get("prompt_type") or "",
                          "fit": round(sum(fit_values) / len(fit_values), 2) if fit_values else None,
                          "marked": bool(mark),
                          "info": {"prompt": overrides.get("prompt", ""), "words": cell.get("words") or "",
                                   "settings": _cell_settings(cell, varying)},
                          # a one-LoRA cell with its own prompt (a trigger test): its words can become the pick
                          "words_pick": bool(overrides.get("lora_file") and overrides.get("prompt"))})
        import glob
        has_book = any(not b.endswith(".prev.xlsx") for b in glob.glob(os.path.join(self.folder(sweep_id),
                                                                                    "sweep_review*.xlsx")))
        return {"id": sweep_id, "checkpoint": manifest.get("checkpoint", ""), "created": manifest.get("created", ""),
                "lora": (manifest.get("plan") or {}).get("lora"), "cells": cells, "tags": TAGS,
                "flaws": [{"tag": f[0], "cause": f[1], "key": f[2]} for f in FLAWS], "praise": PRAISE,
            "praise_keys": PRAISE_KEYS,
                "workbook": has_book}

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
        row = connection.execute("select path, sha256, model_name, stem, arch_folder from loras where path = ?",
                                 (path,)).fetchone()
        if row is None:            # moved since the sweep: path_history knows where it went
            moved = connection.execute("select new_path from path_history where old_path = ? and undone = 0 "
                                       "order by moved_at desc limit 1", (path,)).fetchone() \
                if connection.execute("select 1 from sqlite_master where name = 'path_history'").fetchone() else None
            if moved:
                row = connection.execute("select path, sha256, model_name, stem, arch_folder from loras where path = ?",
                                         (moved[0],)).fetchone()
        if row is None:
            return {"sha256": None, "name": os.path.splitext(os.path.basename(path))[0], "arch": "", "path": path}
        return {"sha256": (row["sha256"] or "").lower() or None, "name": row["model_name"] or row["stem"],
                "arch": row["arch_folder"] or "", "path": row["path"]}

    def _fits(self, connection: sqlite3.Connection, loras: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """{lora key: prompt fit} for the swept LoRAs (markfit.py). A failure leaves the fit unknown, never the mark."""
        from .markmodel import lora_key
        out = {}
        for lora in loras:
            if lora.get("panel"):
                continue
            try:
                value = self.fit_index.fit(lora.get("current_path") or lora.get("path") or "", prompt)
            except Exception as error:          # noqa: BLE001 - the mark matters more than its fit
                print(f"[stacks] prompt fit failed for {lora.get('name')}: {error}")
                value = None
            if value:
                value.pop("own_top", None)
                out[lora_key(lora)] = value
        return out

    def mark(self, sweep_id: str, file_name: str, verdict: Optional[str], notes: Optional[str] = None,
             stars: Optional[int] = None, tags: Optional[List[str]] = None, clear: bool = False,
             aspects: Optional[Dict[str, Any]] = None, push: Optional[str] = None) -> Dict[str, Any]:
        """Record your mark on one cell: stars 1-5 (0 = take them off), why-tags, a strength push (stronger / weaker /
        '' none), notes. Fields left None keep what the cell had. The verdict follows the stars; an explicit verdict
        (the old G / B / U) is only taken for a cell without stars. clear=True removes the mark, and so does taking off
        the last thing it held. Marking the same cell again updates it; other cells and other sweeps are separate
        observations."""
        manifest = self.manifest(sweep_id)
        cell = next((c for c in manifest.get("cells") or [] if c.get("file") == file_name), None)
        if cell is None:
            return {"ok": False, "error": "no such cell"}
        with self.connect() as connection:
            ensure(connection)
            old = connection.execute("select verdict, stars, tags, notes, push from stacks where sweep_id = ? and "
                                     "cell_file = ?", (sweep_id, file_name)).fetchone()
            if clear or (verdict == "" and stars is None and tags is None and push is None):
                connection.execute("delete from stacks where sweep_id = ? and cell_file = ?", (sweep_id, file_name))
                self._model_soon()
                return {"ok": True, "verdict": None, "stars": None, "tags": [], "push": ""}
            if stars is not None:
                stars = int(stars) if str(stars).strip() not in ("", "0") else 0
                if stars and not 1 <= stars <= 5:
                    return {"ok": False, "error": "stars must be 1-5"}
            new_stars = (stars or None) if stars is not None else (old["stars"] if old else None)
            if new_stars:
                new_verdict = STAR_VERDICT[new_stars]
            elif verdict in VERDICTS:
                new_verdict = verdict
            elif verdict:
                return {"ok": False, "error": f"verdict must be one of {VERDICTS}"}
            else:
                new_verdict = old["verdict"] if old and not old["stars"] else None
            new_tags = normalize_tags(tags) if tags is not None else \
                (normalize_tags(json.loads(old["tags"])) if old and old["tags"] else [])
            if push is not None:
                if push and push not in PUSHES:
                    return {"ok": False, "error": f"push must be one of {PUSHES} or empty"}
                new_push = push or None
            else:
                new_push = old["push"] if old else None
            keep_notes = notes if notes is not None else (old["notes"] if old else None)
            if not (new_stars or new_verdict or new_tags or new_push or keep_notes):
                connection.execute("delete from stacks where sweep_id = ? and cell_file = ?", (sweep_id, file_name))
                self._model_soon()
                return {"ok": True, "verdict": None, "stars": None, "tags": [], "push": ""}
            image_path = os.path.join(self.folder(sweep_id), file_name)
            chunk = self._chunk(image_path)
            loras = []
            for entry in chunk.get("lora") or []:
                facts = self._lora_facts(connection, str(entry.get("path") or ""))
                weights = [entry.get("weight_s1", entry.get("strength")), entry.get("weight_s2", entry.get("strength")),
                           entry.get("weight_s3", entry.get("strength"))]
                loras.append({"name": facts["name"], "path": entry.get("path"), "sha256": facts["sha256"],
                              "weights": [round(float(w), 4) if w is not None else None for w in weights],
                              "panel": not str(entry.get("adapter_name", "")).startswith("sweep_"), "arch": facts["arch"],
                              "current_path": facts["path"]})
            swept = [lora for lora in loras if not lora["panel"]]
            key = " + ".join(sorted(f"{lora['sha256'] or lora['name']}@{'/'.join(f'{w:g}' for w in lora['weights'] if w is not None)}"
                                    for lora in swept))
            ultra = chunk.get("ultra") or {}
            prompt = chunk.get("sweep_prompt") or ultra.get("prompt") or (manifest.get("base") or {}).get("prompt") or ""
            base_model = next((lora["arch"] for lora in loras if lora["arch"]), "")
            ctx = self._context(manifest, cell, chunk)
            fits = self._fits(connection, swept, ctx["prompt_base"]) if swept else {}
            for lora in loras:
                lora.pop("current_path", None)
            connection.execute("""
                insert into stacks (verdict, source, sweep_id, cell_file, image_path, checkpoint, base_model, prompt, seed,
                                    loras, stack_key, n_loras, notes, marked_at, stars, tags, subject, words, word_mode,
                                    sampler, steps, aspects, push, prompt_base, prompt_type, fit)
                values (?, 'sweep', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict (sweep_id, cell_file) do update set verdict = excluded.verdict,
                    notes = coalesce(excluded.notes, stacks.notes), marked_at = excluded.marked_at,
                    stars = excluded.stars, tags = excluded.tags, subject = excluded.subject, words = excluded.words,
                    word_mode = excluded.word_mode, sampler = excluded.sampler, steps = excluded.steps,
                    aspects = coalesce(excluded.aspects, stacks.aspects), push = excluded.push,
                    prompt_base = excluded.prompt_base, prompt_type = excluded.prompt_type, fit = excluded.fit""",
                (new_verdict, sweep_id, file_name, image_path, manifest.get("checkpoint", ""), base_model, prompt,
                 cell.get("seed"), json.dumps(loras, ensure_ascii=False), key, len(swept), keep_notes,
                 datetime.now().isoformat(timespec="seconds"), new_stars, json.dumps(new_tags, ensure_ascii=False),
                 ctx["subject"], ctx["words"], ctx["word_mode"], ctx["sampler"], ctx["steps"],
                 json.dumps(aspects, ensure_ascii=False) if aspects else None, new_push, ctx["prompt_base"],
                 ctx["prompt_type"], json.dumps(fits) if fits else None))
        self._model_soon()
        fit_values = [f["fit"] for f in fits.values() if f.get("fit") is not None]
        return {"ok": True, "verdict": new_verdict, "stars": new_stars, "tags": new_tags, "push": new_push or "",
                "n_loras": len(swept), "ptype": ctx["prompt_type"],
                "fit": round(sum(fit_values) / len(fit_values), 2) if fit_values else None}

    def _model_soon(self) -> None:
        """Refit the marks model MODEL_DELAY seconds after the last mark (marks come in bursts)."""
        with self.lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.MODEL_DELAY, self.update_model)
            self._timer.daemon = True
            self._timer.start()

    def update_model(self, log=print) -> Dict[str, int]:
        from . import markmodel
        try:
            return markmodel.update(self.settings.catalog_path, tag_stats, log=log)
        except Exception as error:          # noqa: BLE001 - a background refit must never take the editor down
            log(f"[marks] model refit failed: {error}")
            return {"loras": 0}

    @staticmethod
    def _context(manifest: Dict[str, Any], cell: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Subject and type of the prompt, the prompt without the LoRA words, the words added for the swept LoRAs and
        the trigger mode, sampler and steps."""
        from .markfit import base_prompt, prompt_type
        from .subjects import subject_of
        ultra = chunk.get("ultra") or {}
        over = cell.get("overrides") or {}
        resolved = cell.get("resolved") or {}
        # the subject comes from the prompt WITHOUT the added LoRA words (those describe the style, not the subject)
        prompt = over.get("prompt") or ultra.get("prompt") or (manifest.get("base") or {}).get("prompt") or \
            chunk.get("sweep_prompt") or ""
        words = chunk.get("sweep_triggers") or []
        words_text = ", ".join(str(w) for w in words) if words else ""
        clean = base_prompt(prompt, words_text)
        mode = ((manifest.get("plan") or {}).get("lora") or {}).get("triggers") or ""
        mode = {"off": "none", "catalog": "trigger", "recipe": "saved pick"}.get(mode, mode)
        sampler = over.get("s1_sampler") or resolved.get("s1_sampler") or ultra.get("s1_sampler") or ""
        steps = over.get("s1_steps") or resolved.get("s1_steps") or ultra.get("s1_steps")
        try:
            steps = int(steps) if steps not in (None, "") else None
        except (TypeError, ValueError):
            steps = None
        subject = subject_of(clean)
        return {"subject": subject, "words": words_text, "word_mode": mode if (words or mode) else "",
                "sampler": str(sampler), "steps": steps, "prompt_base": clean, "prompt_type": prompt_type(subject)}

    def backfill(self, log=print) -> Dict[str, int]:
        """Marks made before 2026-10-17 get their subject / words / sampler, the prompt without the LoRA words, its
        type and the prompt fit, from their cell PNG and manifest (marks whose sweep folder is gone keep what they
        have; their fit is worked out from the stored prompt)."""
        from .markfit import base_prompt, prompt_type
        from .subjects import subject_of
        counts = {"filled": 0, "no_sweep": 0}
        with self.connect() as connection:
            ensure(connection)
            rows = connection.execute("select * from stacks where prompt_type is null or (fit is null and n_loras > 0)"
                                      ).fetchall()
            manifests: Dict[str, Any] = {}
            for row in rows:
                ctx = None
                try:
                    if row["sweep_id"] not in manifests:
                        manifests[row["sweep_id"]] = self.manifest(row["sweep_id"])
                    manifest = manifests[row["sweep_id"]]
                    cell = next((c for c in manifest.get("cells") or [] if c.get("file") == row["cell_file"]), {})
                    chunk = self._chunk(os.path.join(self.folder(row["sweep_id"]), row["cell_file"]))
                    ctx = self._context(manifest, cell, chunk)
                except Exception:          # noqa: BLE001 - the sweep folder may be gone
                    counts["no_sweep"] += 1
                if ctx is None:
                    clean = base_prompt(row["prompt"] or "", row["words"] or "")
                    subject = row["subject"] or subject_of(clean)
                    ctx = {"subject": subject, "words": row["words"] or "", "word_mode": row["word_mode"] or "",
                           "sampler": row["sampler"] or "", "steps": row["steps"], "prompt_base": clean,
                           "prompt_type": prompt_type(subject)}
                loras = json.loads(row["loras"] or "[]")
                for lora in loras:
                    lora["current_path"] = self._lora_facts(connection, str(lora.get("path") or ""))["path"]
                fits = self._fits(connection, [l for l in loras if not l.get("panel")], ctx["prompt_base"])
                connection.execute("update stacks set subject = ?, words = ?, word_mode = ?, sampler = ?, steps = ?, "
                                   "prompt_base = ?, prompt_type = ?, fit = ? where id = ?",
                                   (ctx["subject"], ctx["words"], ctx["word_mode"], ctx["sampler"], ctx["steps"],
                                    ctx["prompt_base"], ctx["prompt_type"], json.dumps(fits) if fits else None, row["id"]))
                counts["filled"] += 1
        log(f"[stacks] backfill: {counts}")
        return counts

    backfill_context = backfill          # the 2026-10-16 name

    # -- the workbook's ratings (Eric_Krea2 sweep_review.xlsx) ------------------------------------------------------
    def import_workbook(self, sweep_id: str) -> Dict[str, Any]:
        """Ratings typed into a sweep's review workbook become marks: 'rating' (1-5) -> stars, notes -> notes (a note
        made on the Sweeps page wins), the other rating columns (fine detail, render quality ...) are kept with the
        mark."""
        import glob
        folder = self.folder(sweep_id)
        books = sorted(glob.glob(os.path.join(folder, "sweep_review*.xlsx")))
        books = [b for b in books if not b.endswith(".prev.xlsx")]
        if not books:
            return {"ok": False, "error": "this sweep has no review workbook"}
        from openpyxl import load_workbook
        wb = load_workbook(books[0], read_only=True, data_only=True)
        ws = wb["Review"] if "Review" in wb.sheetnames else wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows or "cell" not in [str(h or "") for h in rows[0]]:
            return {"ok": False, "error": "the workbook has no 'cell' column"}
        header = [str(h or "").strip() for h in rows[0]]
        at = {h: i for i, h in enumerate(header)}
        aspects_cols = [h for h in ("fine detail", "render quality", "contrast", "composition", "cohesiveness")
                        if h in at]
        manifest = self.manifest(sweep_id)
        by_index = {c.get("index"): c for c in manifest.get("cells") or []}
        counts = {"rated": 0, "skipped": 0}
        for row in rows[1:]:
            cell = by_index.get(row[at["cell"]]) if at["cell"] < len(row) else None
            if not cell or not cell.get("file"):
                continue

            def val(name):
                i = at.get(name)
                return row[i] if i is not None and i < len(row) and row[i] not in (None, "") else None
            rating, notes = val("rating"), val("notes")
            aspects = {h: val(h) for h in aspects_cols if val(h) is not None}
            stars = None
            try:
                stars = max(1, min(5, int(round(float(rating))))) if rating is not None else None
            except (TypeError, ValueError):
                stars = None
            if stars is None and not notes and not aspects:
                continue
            if stars is None:
                with self.connect() as connection:
                    has = connection.execute("select 1 from stacks where sweep_id = ? and cell_file = ?",
                                             (sweep_id, cell["file"])).fetchone()
                if not has:                      # notes alone do not make a mark here
                    counts["skipped"] += 1
                    continue
            with self.connect() as connection:
                old = connection.execute("select notes from stacks where sweep_id = ? and cell_file = ?",
                                         (sweep_id, cell["file"])).fetchone()
            keep_notes = old["notes"] if old and old["notes"] else (str(notes) if notes else None)
            r = self.mark(sweep_id, cell["file"], None, keep_notes, stars=stars, aspects=aspects or None)
            counts["rated" if r.get("ok") else "skipped"] += 1
        return {"ok": True, **counts, "workbook": os.path.basename(books[0])}

    # -- one LoRA's record, for its card (2026-10-16, 2026-10-17) ------------------------------------------------------
    def lora_record(self, path: str, card_weight: Optional[float] = None) -> Dict[str, Any]:
        """Every mark that involved this LoRA as a swept LoRA, summed up: by checkpoint, prompt type, subject, weight,
        word mode, alone vs in a stack and partners; checkpoint x prompt type; flaw and praise rates against the
        library; the strength range per checkpoint from your pushes and stars; what it was made for vs where it did
        well; the model's view (markmodel.py: quality with prompt and checkpoint taken out); how sure all this is; and
        offers (a weight, a word mode) when the marks clearly point somewhere."""
        from . import markmodel
        with self.connect() as connection:
            ensure(connection)
            lr = connection.execute("select sha256 from loras where path = ?", (path,)).fetchone()
            sha = ((lr["sha256"] if lr else "") or "").lower()
            like = os.path.basename(path)
            rows = connection.execute("select * from stacks where loras like ?",
                                      ("%" + (sha or like.replace("%", "")) + "%",)).fetchall()
            library = tag_rates(connection)
            model = markmodel.current(connection, self.settings.catalog_path)
            prompt_index, baselines = prompt_scores(connection)
        key = sha or like
        obs = []
        for row in rows:
            mine = None
            for l in json.loads(row["loras"] or "[]"):
                if l.get("panel"):
                    continue
                if (sha and (l.get("sha256") or "").lower() == sha) or os.path.basename(str(l.get("path") or "")) == like:
                    mine = l
            if mine is None:
                continue
            w = (mine.get("weights") or [None])[0]
            others = [l["name"] for l in json.loads(row["loras"] or "[]") if not l.get("panel") and l is not mine
                      and l.get("name") != mine.get("name")]
            fit = None
            try:
                f = json.loads(row["fit"] or "{}").get(markmodel.lora_key(mine))
                fit = f.get("fit") if isinstance(f, dict) else None
            except (ValueError, AttributeError):
                fit = None
            obs.append({"score": score_of(row), "stars": row["stars"], "verdict": row["verdict"],
                        "pk": markmodel._prompt_key(row), "prompt": row["prompt_base"] or row["prompt"] or "",
                        "checkpoint": row["checkpoint"] or "?", "subject": row["subject"] or "?",
                        "ptype": row["prompt_type"] or "?", "fit": fit, "push": row["push"] or "",
                        "weight": w, "mode": row["word_mode"] or "", "words": row["words"] or "",
                        "alone": not others, "with": others,
                        "tags": normalize_tags(json.loads(row["tags"])) if row["tags"] else [],
                        "sweep": row["sweep_id"], "file": row["cell_file"], "notes": row["notes"] or ""})
        rated = [o for o in obs if o["score"] is not None]

        def group(key_of, source=None):
            out: Dict[str, Dict[str, Any]] = {}
            for o in (rated if source is None else source):
                k = key_of(o)
                if k is None:
                    continue
                g = out.setdefault(str(k), {"key": str(k), "n": 0, "good": 0, "bad": 0, "unsure": 0, "sum": 0.0,
                                            "stars_n": 0, "stars_sum": 0})
                g["n"] += 1
                if o["verdict"]:
                    g[o["verdict"]] = g.get(o["verdict"], 0) + 1
                g["sum"] += o["score"]
                if o["stars"]:
                    g["stars_n"] += 1
                    g["stars_sum"] += o["stars"]
            result = []
            for g in out.values():
                g["score"] = round(g.pop("sum") / g["n"], 2)
                g["stars"] = round(g["stars_sum"] / g["stars_n"], 1) if g["stars_n"] else None
                g.pop("stars_sum")
                result.append(g)
            return sorted(result, key=lambda g: (-g["score"], -g["n"], g["key"]))

        def wband(o):
            return None if o["weight"] is None else f"{round(float(o['weight']) * 10) / 10:g}"
        tags: Dict[str, int] = {}
        for o in obs:
            for t in o["tags"]:
                tags[t] = tags.get(t, 0) + 1
        record = {"n": len(obs), "n_rated": len(rated), "n_stars": sum(1 for o in obs if o["stars"]),
                  "by_checkpoint": group(lambda o: o["checkpoint"]),
                  "by_type": group(lambda o: o["ptype"] if o["ptype"] != "?" else None),
                  "by_subject": group(lambda o: o["subject"]), "by_weight": group(wband),
                  "by_mode": group(lambda o: f"{o['mode'] or '-'} on {o['checkpoint']}" if o["alone"] else None),
                  "by_partner": group(lambda o: " + ".join(o["with"]) if o["with"] else None),
                  "alone_vs_stack": group(lambda o: "alone" if o["alone"] else "in a stack"),
                  "matrix": [dict(g, checkpoint=g["key"].split(" × ")[0], ptype=g["key"].split(" × ")[1])
                             for g in group(lambda o: f"{o['checkpoint']} × {o['ptype']}" if o["ptype"] != "?" else None)],
                  "tags": sorted(tags.items(), key=lambda kv: -kv[1]),
                  "notes": [{"note": o["notes"], "sweep": o["sweep"], "file": o["file"]} for o in obs if o["notes"]][-8:],
                  "offers": []}
        record["flaws"], record["praise"] = self._tag_summary(obs, library)
        _st = [o["stars"] for o in obs if o["stars"]]                       # 2026-10-26: card header badge
        record["stars_mean"] = round(sum(_st) / len(_st), 2) if _st else None
        record["synergy"], record["co_use"] = self._partners(key, path)       # 2026-10-27
        record["by_prompt"] = self._by_prompt(rated, prompt_index, baselines, key)
        record["good_for"] = self._good_for(record["by_prompt"])
        record["strength"] = self._strength(obs, card_weight)
        record["wheelhouse"] = self._wheelhouse(path, rated)
        record["model"] = markmodel.explain(model, key)
        record["confidence"] = self._confidence(rated)
        # offers: only when the marks clearly point somewhere (2+ marks, better than the next option)
        weights = [g for g in record["by_weight"] if g["n"] >= 2]
        if weights and (len(weights) == 1 or weights[0]["score"] > weights[1]["score"]):
            best = float(weights[0]["key"])
            if card_weight is None or abs(best - float(card_weight)) >= 0.05:
                record["offers"].append({"kind": "weight", "value": best,
                                         "why": f"@{best:g} scored {weights[0]['score']} over {weights[0]['n']} marks"})
        for s in record["strength"]:
            if s.get("suggest") is not None and not any(o["kind"] == "weight" for o in record["offers"]) and \
                    (card_weight is None or abs(s["suggest"] - float(card_weight)) >= 0.05):
                record["offers"].append({"kind": "weight", "value": s["suggest"], "why": s["text"]})
                break
        for ck in {o["checkpoint"] for o in rated if o["alone"]}:
            modes = group(lambda o, ck=ck: o["mode"] if o["alone"] and o["checkpoint"] == ck and o["words"] else None)
            modes = [m for m in modes if m["n"] >= 2]
            if len(modes) >= 2 and modes[0]["score"] > modes[1]["score"]:
                words = next(o["words"] for o in reversed(rated)
                             if o["alone"] and o["checkpoint"] == ck and o["mode"] == modes[0]["key"] and o["words"])
                record["offers"].append({"kind": "words", "checkpoint": ck, "mode": modes[0]["key"], "text": words,
                                         "why": f"'{modes[0]['key']}' words scored {modes[0]['score']} vs "
                                                f"{modes[1]['score']} ('{modes[1]['key']}') on {ck}"})
        return record

    def _partners(self, key: str, path: str):
        """This LoRA's partners (2026-10-27): your stack synergy (markmodel lora_synergy) and what Civitai users stack
        it with (run.py harvest -> lora_pairs). Either list may be empty."""
        syn, co = [], []
        with self.connect() as connection:
            tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
            if "lora_synergy" in tables:
                for r in connection.execute("select * from lora_synergy where a = ? or b = ?", (key, key)):
                    mine_a = r["a"] == key
                    syn.append({"name": r["b_name"] if mine_a else r["a_name"],
                                "path": r["b_path"] if mine_a else r["a_path"],
                                "synergy": round(r["synergy"], 2), "n": r["n_marks"],
                                "stars": round(r["stars"], 2), "expected": round(r["expected"], 2)})
                syn.sort(key=lambda s: -s["synergy"])
            if "lora_pairs" in tables and "loras" in tables:
                cols = {r[1] for r in connection.execute("pragma table_info(loras)")}
                if "civitai_version_id" in cols:
                    v = connection.execute("select civitai_version_id from loras where path = ?", (path,)).fetchone()
                    if v and v[0]:
                        vk = f"v:{int(v[0])}"
                        for r in connection.execute("select * from lora_pairs where a = ? or b = ? order by score desc "
                                                    "limit 8", (vk, vk)).fetchall():
                            other = r["b"] if r["a"] == vk else r["a"]
                            owned = r["owned_b"] if r["a"] == vk else r["owned_a"]
                            name, opath = other, ""
                            if other.startswith("v:"):
                                lr = connection.execute("select path from loras where civitai_version_id = ?",
                                                        (int(other[2:]),)).fetchone()
                                if lr and lr[0]:
                                    opath = lr[0]
                                    name = os.path.splitext(os.path.basename(lr[0]))[0]
                            co.append({"name": name, "path": opath, "owned": bool(owned), "users": r["n"],
                                       "score": round(r["score"] or 0, 2)})
        return syn, co

    @staticmethod
    def _tag_summary(obs: List[Dict[str, Any]], library: Dict[str, float]):
        """Each tag this LoRA got: how often (k of n marks), against how often LoRAs get it overall, per checkpoint.
        flagged = 2+ times and at least twice the library rate (or 1 in 3 when the library has none)."""
        n = len(obs)
        flaws, praise = [], []
        for tag in sorted({t for o in obs for t in o["tags"]}):
            k = sum(1 for o in obs if tag in o["tags"])
            by_ck: Dict[str, List[int]] = {}
            for o in obs:
                c = by_ck.setdefault(o["checkpoint"], [0, 0])
                c[1] += 1
                c[0] += tag in o["tags"]
            rate = k / n if n else 0.0
            base = library.get(tag, 0.0)
            item = {"tag": tag, "k": k, "n": n, "rate": round(rate, 2), "library": round(base, 3),
                    "by_checkpoint": {ck: f"{v[0]}/{v[1]}" for ck, v in by_ck.items() if v[0]},
                    "flagged": k >= 2 and (rate >= 2 * base if base else rate >= 0.33)}
            if tag in PRAISE:
                praise.append(item)
            else:
                item["cause"] = FLAW_CAUSE.get(tag, "")
                flaws.append(item)
        key = lambda t: (-int(t["flagged"]), -t["rate"], t["tag"])
        return sorted(flaws, key=key), sorted(praise, key=key)

    # -- the LoRA prompt by prompt (2026-10-18) ----------------------------------------------------------------------
    # (PLAIN_WORDS, module level: words that describe no kind of prompt - poses, hands, 'one', 'other' ...)
    REL_CLEAR = 0.75             # stars above / below the other LoRAs on the same prompt that count as clearly good / weak

    @staticmethod
    def _by_prompt(rated: List[Dict[str, Any]], index: Dict[tuple, List[tuple]], baselines: Dict[tuple, float],
                   key: str) -> List[Dict[str, Any]]:
        """One row per prompt (and checkpoint) this LoRA was rated on: its score, the mean of the OTHER LoRAs rated on
        the same prompt and checkpoint, and the difference (rel) - in a sweep every LoRA got the same prompt, trigger
        mode and seed, so rel is about the LoRA on that prompt, not about how hard the prompt is. Best first."""
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for o in rated:
            groups.setdefault(o["pk"], []).append(o)
        rows = []
        for pk, items in groups.items():
            scores = [o["score"] for o in items]
            stars = [o["stars"] for o in items if o["stars"]]
            others = [s for k, s in index.get(pk, []) if k != key]
            mine = sum(scores) / len(scores)
            mean_others = sum(others) / len(others) if others else None
            fits = [o["fit"] for o in items if o["fit"] is not None]
            rows.append({"prompt": items[0]["prompt"][:220], "checkpoint": pk[0], "ptype": items[0]["ptype"],
                         "subject": items[0]["subject"], "score": round(mine, 2), "n": len(items),
                         "stars": round(sum(stars) / len(stars), 1) if stars else None,
                         "others": round(mean_others, 2) if mean_others is not None else None, "n_others": len(others),
                         "rel": round(mine - mean_others, 2) if mean_others is not None else None,
                         "baseline": baselines.get(pk), "fit": round(sum(fits) / len(fits), 2) if fits else None,
                         "tags": sorted({t for o in items for t in o["tags"]}),
                         "notes": [o["notes"] for o in items if o["notes"]][:3]})
        return sorted(rows, key=lambda r: (r["rel"] is None, -(r["rel"] if r["rel"] is not None else r["score"])))

    @classmethod
    def _good_for(cls, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """'good for' / 'weak for' once 2+ prompts are clearly above / below the other LoRAs (2+ others rated on them):
        what those prompts share - their prompt type / subject, and words they share that the other prompts lack.
        Taken from the prompts themselves, no LLM."""
        from .markfit import _tokens
        judged = [r for r in rows if r["rel"] is not None and r["n_others"] >= 2]
        good = [r for r in judged if r["rel"] >= cls.REL_CLEAR]
        weak = [r for r in judged if r["rel"] <= -cls.REL_CLEAR]

        def describe(group: List[Dict[str, Any]], rest: List[Dict[str, Any]]) -> Dict[str, Any]:
            kinds: Dict[str, int] = {}
            for r in group:
                for label in {r["ptype"], r["subject"]} - {"?", "", "other"}:
                    kinds[label] = kinds.get(label, 0) + 1
            rest_kinds: Dict[str, int] = {}
            for r in rest:
                for label in {r["ptype"], r["subject"]}:
                    rest_kinds[label] = rest_kinds.get(label, 0) + 1
            # a kind only says something when the other prompts are mostly NOT of it (all figures says nothing)
            shared_kinds = [k for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])
                            if n >= max(2, (len(group) + 1) // 2) and rest_kinds.get(k, 0) < 0.5 * max(1, len(rest))]
            words: Dict[str, int] = {}
            for r in group:
                for w in set(_tokens(r["prompt"])) - PLAIN_WORDS:
                    words[w] = words.get(w, 0) + 1
            rest_words: Dict[str, int] = {}
            for r in rest:
                for w in set(_tokens(r["prompt"])):
                    rest_words[w] = rest_words.get(w, 0) + 1
            limit = max(1, len(rest) // 3)
            shared_words = [w for w, n in sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))
                            if n >= 2 and rest_words.get(w, 0) <= limit][:5]
            text = " / ".join(shared_kinds[:2]) + (" prompts" if shared_kinds else "")
            if shared_words:
                text += (" - " if text else "prompts with ") + ", ".join(shared_words)
            mean_rel = sum(r["rel"] for r in group) / len(group)
            return {"n": len(group), "of": len(judged), "kinds": shared_kinds, "words": shared_words,
                    "text": text or f"{len(group)} prompts with nothing obvious in common", "rel": round(mean_rel, 2)}
        out: Dict[str, Any] = {"judged": len(judged)}
        if len(good) >= 2:
            out["good"] = describe(good, [r for r in judged if r not in good])
        if len(weak) >= 2:
            out["weak"] = describe(weak, [r for r in judged if r not in weak])
        lines = []
        if out.get("good"):
            lines.append(f"good for: {out['good']['text']} ({out['good']['n']} of {len(judged)} prompts, "
                         f"{out['good']['rel']:+.1f}★ vs the other LoRAs)")
        if out.get("weak"):
            lines.append(f"weak for: {out['weak']['text']} ({out['weak']['n']} of {len(judged)} prompts, "
                         f"{out['weak']['rel']:+.1f}★ vs the other LoRAs)")
        if lines:
            out["note"] = f"Sweeps ({datetime.now():%Y-%m-%d}): " + "; ".join(lines)
        return out

    @staticmethod
    def _strength(obs: List[Dict[str, Any]], card_weight: Optional[float]) -> List[Dict[str, Any]]:
        """Per checkpoint: the weights you said to push up from (stronger) and down from (weaker), the best-scoring
        weight, and a suggested weight between the brackets - with 'too weak' / 'overcooked' tags counting as pushes."""
        out = []
        for ck in sorted({o["checkpoint"] for o in obs}):
            mine = [o for o in obs if o["checkpoint"] == ck and o["weight"] is not None]
            up = sorted({round(float(o["weight"]), 2) for o in mine
                         if o["push"] == "stronger" or "too weak" in o["tags"]})
            down = sorted({round(float(o["weight"]), 2) for o in mine
                           if o["push"] == "weaker" or "overcooked or too strong" in o["tags"]})
            scores: Dict[float, List[float]] = {}
            for o in mine:
                if o["score"] is not None:
                    scores.setdefault(round(float(o["weight"]), 2), []).append(o["score"])
            best = None
            if scores:
                best_w, vals = max(scores.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), len(kv[1])))
                best = {"weight": best_w, "score": round(sum(vals) / len(vals), 2), "n": len(vals)}
            if not up and not down and not (best and len(scores) > 1):
                continue
            lo = max(up) if up else None
            hi = min(down) if down else None
            suggest, text = None, ""
            if lo is not None and hi is not None and lo < hi:
                suggest = round((lo + hi) / 2, 2)
                text = f"on {ck}: too weak at {lo:g}, too strong at {hi:g} - try {suggest:g}"
            elif lo is not None and hi is None:
                suggest = round(lo + 0.15, 2)
                text = f"on {ck}: too weak up to {lo:g} - try {suggest:g} or more"
            elif hi is not None and lo is None:
                suggest = round(max(0.05, hi - 0.15), 2)
                text = f"on {ck}: too strong from {hi:g} - try {suggest:g} or less"
            elif lo is not None and hi is not None:
                text = f"on {ck}: pushes disagree (stronger at {lo:g}, weaker at {hi:g}) - needs more marks"
            elif best:
                text = f"on {ck}: best at {best['weight']:g} ({best['score']} over {best['n']})"
            out.append({"checkpoint": ck, "stronger_at": up, "weaker_at": down, "best": best, "suggest": suggest,
                        "text": text})
        return out

    def _wheelhouse(self, path: str, rated: List[Dict[str, Any]]) -> Dict[str, Any]:
        """What it was made for (prompt types of its own prompts) against where your marks say it does well."""
        try:
            own = self.fit_index.own_types(path)
        except Exception:          # noqa: BLE001
            own = {}
        out: Dict[str, Any] = {"own": {k: round(v, 2) for k, v in sorted(own.items(), key=lambda kv: -kv[1]) if v >= 0.05}}
        by_type: Dict[str, List[float]] = {}
        for o in rated:
            if o["ptype"] != "?":
                by_type.setdefault(o["ptype"], []).append(o["score"])
        seen = {k: (round(sum(v) / len(v), 2), len(v)) for k, v in by_type.items()}
        out["seen"] = seen
        made_for = max(own.items(), key=lambda kv: kv[1])[0] if own else ""
        good = [(k, s, n) for k, (s, n) in seen.items() if n >= 2]
        best = max(good, key=lambda t: t[1]) if good else None
        fits = [(o["fit"], o["score"]) for o in rated if o["fit"] is not None]
        inside = [s for f, s in fits if f >= 0.5]
        outside = [s for f, s in fits if f < 0.25]
        if len(inside) >= 2 and len(outside) >= 2:
            a, b = sum(inside) / len(inside), sum(outside) / len(outside)
            out["in_vs_out"] = {"inside": round(a, 2), "n_in": len(inside), "outside": round(b, 2), "n_out": len(outside)}
        out["text"] = ""
        if made_for and best and best[0] != made_for and best[1] - seen.get(made_for, (0, 0))[0] >= 0.5:
            out["disagree"] = True
            out["text"] = (f"its own prompts are mostly {made_for} ({own[made_for]:.0%}); your marks favour {best[0]} "
                           f"({best[1]} over {best[2]}" + (f", vs {seen[made_for][0]} on {made_for}" if made_for in seen
                                                           else "") + ")")
        elif made_for:
            out["text"] = f"made for {made_for} ({own[made_for]:.0%} of its own prompts)"
        return out

    @staticmethod
    def _confidence(rated: List[Dict[str, Any]]) -> Dict[str, Any]:
        stars = sum(1 for o in rated if o["stars"])
        cks = len({o["checkpoint"] for o in rated})
        types = len({o["ptype"] for o in rated if o["ptype"] != "?"})
        weight = stars + 0.5 * (len(rated) - stars)
        label = "none" if not rated else "early" if weight < 5 else "solid" if weight >= 15 and cks >= 2 else "some"
        return {"label": label, "text": f"{len(rated)} rated mark(s) ({stars} with stars) on {cks} checkpoint(s), "
                                         f"{types} prompt type(s)"}

    # -- reading the marks back ------------------------------------------------------------------------------------
    def summary(self) -> List[Dict[str, Any]]:
        """One row per distinct stack (same LoRAs at the same weights): good / bad / unsure counts and mean stars."""
        groups: Dict[str, Dict[str, Any]] = {}
        with self.connect() as connection:
            for row in connection.execute("select * from stacks where n_loras > 0 order by marked_at"):
                if not row["verdict"]:
                    continue
                group = groups.setdefault(row["stack_key"], {
                    "stack": " + ".join(f"{l['name']} @{'/'.join(f'{w:g}' for w in l['weights'] if w is not None)}"
                                        for l in json.loads(row["loras"]) if not l["panel"]),
                    "n_loras": row["n_loras"], "good": 0, "bad": 0, "unsure": 0, "sweeps": set(), "stars": []})
                group[row["verdict"]] = group.get(row["verdict"], 0) + 1
                group["sweeps"].add(row["sweep_id"])
                if row["stars"]:
                    group["stars"].append(row["stars"])
        out = []
        for key, group in groups.items():
            group["sweeps"] = len(group["sweeps"])
            group["key"] = key
            group["mean_stars"] = round(sum(group["stars"]) / len(group["stars"]), 1) if group["stars"] else None
            group.pop("stars")
            out.append(group)
        out.sort(key=lambda g: (-g["n_loras"], -(g["good"] - g["bad"]), g["stack"].lower()))
        return out

    def pairs(self) -> List[Dict[str, Any]]:
        """Every two swept LoRAs that appeared together in a marked cell: good / bad counts (sha256 keyed)."""
        counts: Dict[tuple, Dict[str, Any]] = {}
        with self.connect() as connection:
            for row in connection.execute("select verdict, loras from stacks where n_loras > 1"):
                if not row["verdict"]:
                    continue
                swept = [l for l in json.loads(row["loras"]) if not l["panel"]]
                for i, a in enumerate(swept):
                    for b in swept[i + 1:]:
                        ka, kb = sorted([(a["sha256"] or a["name"], a["name"]), (b["sha256"] or b["name"], b["name"])])
                        pair = counts.setdefault((ka[0], kb[0]), {"a": ka[1], "b": kb[1], "good": 0, "bad": 0,
                                                                   "unsure": 0})
                        pair[row["verdict"]] += 1
        return sorted(counts.values(), key=lambda p: (-(p["good"] - p["bad"]), p["a"].lower()))


# Words too plain to say what a group of prompts has in common (good for / weak for).
PLAIN_WORDS = set("""one two three other another each toward towards hand hands sit sits sitting stand stands holds hold
hold leans lean look looks smile smiles smiling flashe flashes wear wears dressed perfectly behind front near next
about around while just very slightly bright big small little woman women man men girl lady person young old""".split())


def prompt_scores(connection: sqlite3.Connection):
    """({(checkpoint, prompt): [(lora key, score), ...]} over every rated LoRA mark - a stack cell counts for each of its
    LoRAs - and {(checkpoint, prompt): baseline stars} for rated baseline cells). Used to judge a LoRA against the
    others on the same prompt."""
    from .markmodel import _prompt_key, lora_key
    index: Dict[tuple, List[tuple]] = {}
    base: Dict[tuple, List[float]] = {}
    for row in connection.execute("select loras, checkpoint, prompt_base, prompt, stars, verdict, n_loras from stacks"):
        score = score_of(row)
        if score is None:
            continue
        pk = _prompt_key(row)
        swept = [l for l in json.loads(row["loras"] or "[]") if not l.get("panel")]
        if not swept:
            if row["n_loras"] == 0 and row["stars"]:
                base.setdefault(pk, []).append(float(row["stars"]))
            continue
        for entry in swept:
            index.setdefault(pk, []).append((lora_key(entry), score))
    return index, {pk: round(sum(v) / len(v), 1) for pk, v in base.items()}


def tag_rates(connection: sqlite3.Connection) -> Dict[str, float]:
    """How often each tag is given across all LoRA marks: the library base rate a LoRA's flaw rate is judged against."""
    n = 0
    counts: Dict[str, int] = {}
    for (tags,) in connection.execute("select tags from stacks where n_loras > 0"):
        n += 1
        for t in normalize_tags(json.loads(tags or "[]")):
            counts[t] = counts.get(t, 0) + 1
    return {t: c / n for t, c in counts.items()} if n else {}


def tag_stats(connection: sqlite3.Connection) -> Dict[str, tuple]:
    """{lora key: ({flaw: [k, n]}, {praise: [k, n]})} over every mark of each LoRA (for lora_observed / the Suggester),
    plus per checkpoint under 'ck:<checkpoint>' inside each dict."""
    from .markmodel import lora_key
    out: Dict[str, tuple] = {}
    totals: Dict[str, int] = {}
    totals_ck: Dict[tuple, int] = {}
    seen: List[tuple] = []
    for row in connection.execute("select loras, tags, checkpoint from stacks where n_loras > 0"):
        tags = normalize_tags(json.loads(row["tags"] or "[]"))
        for entry in json.loads(row["loras"] or "[]"):
            if entry.get("panel"):
                continue
            key = lora_key(entry)
            totals[key] = totals.get(key, 0) + 1
            totals_ck[(key, row["checkpoint"])] = totals_ck.get((key, row["checkpoint"]), 0) + 1
            seen.append((key, row["checkpoint"], tags))
    for key, ck, tags in seen:
        flaws, praise = out.setdefault(key, ({}, {}))
        for t in tags:
            target = praise if t in PRAISE else flaws
            target.setdefault(t, [0, totals[key]])[0] += 1
            target.setdefault(f"ck:{ck}|{t}", [0, totals_ck[(key, ck)]])[0] += 1
    return out


def report(settings: Settings, pairs: bool = False) -> None:
    sweeps = Sweeps(settings)
    rows = sweeps.pairs() if pairs else sweeps.summary()
    if not rows:
        print("[stacks] nothing marked yet - open the Sweeps page in the card editor")
        return
    for row in rows:
        tally = f"good {row['good']}  bad {row['bad']}  unsure {row['unsure']}"
        stars = f"  {row['mean_stars']}★" if not pairs and row.get("mean_stars") else ""
        print(f"  {tally:30s} {row['a'] + '  +  ' + row['b'] if pairs else row['stack']}{stars}")
