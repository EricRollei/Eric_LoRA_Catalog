"""
Prompt library (2026-10-29, docs/design/PROMPT_LIBRARY.md): every prompt the catalog has collected - card preview
prompts (`examples`) and Civitai community prompts (`community_prompts`) - deduplicated into one SQLite file you can
filter and export (text for the Eric_Krea2 Sweep Plan, JSONL / CSV, SQLite), optionally rewritten by the local LLM.

    python run.py prompts build                         merge new prompts in (also the last step of run.py refresh)
    python run.py prompts stats
    python run.py prompts export --base Krea --max-nsfw R --style natural --out D:\\prompts\\krea2.txt

The library: settings "prompt_library_dir" (default: a "prompts" folder beside the folder holding the catalog, e.g.
L:\\Models\\prompts), file _prompt_library.sqlite. It is rebuilt from the catalog - never edit it by hand.

Tables
    prompts      id (hash of the normalised stripped text), text (triggers stripped, one line), original (one wording
                 before stripping), negative, style (natural | tags), words, nsfw (0 PG .. 4 XXX, 9 blocked, NULL
                 unknown), nsfw_text (1 = the words mention nudity), reactions_max, reactions_sum, occurrences,
                 base_models, first_seen, last_seen
    occurrences  occ (preview:<path>|<file> or community:<source>:<image>:<version>), prompt_id, source, lora_path
                 (the LoRA it came with, yours), base_model, checkpoint, sampler, steps, cfg, weight, nsfw, reactions,
                 image_id, first_seen, last_seen
    loras_used   occ, prompt_id, lora (name or Civitai key), weight, path (when yours)
    stripped     occ, prompt_id, lora_path, trigger     - what was removed, per LoRA
    facets       prompt_id, facet, value                - the card facets of the LoRAs it came with
    rewrites     prompt_id, mode, model, sys_hash, text, created_at   - the LLM rewrite cache

Trigger stripping: the card triggers of the LoRA a prompt came with, and of every other LoRA of yours that image used,
whole phrase, case-insensitive. A "trigger" that is an ordinary prompt word (it appears in 0.2% or more of all
prompts, e.g. "watercolor", "woman") is left alone - removing it would change what the prompt describes. Word options
and LoRA names are not stripped (they are usually descriptive words).

NSFW: Civitai's scale PG / PG-13 / R / X / XXX. Preview images carry Civitai's number (1/2/4/8/16; 32 = blocked, never
exported); community and harvest rows carry the older text (None / Soft / Mature / X), where "X" also covers XXX.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .config import Settings

LEVELS = ["PG", "PG-13", "R", "X", "XXX"]
BLOCKED = 9
GENERIC_SHARE = 0.002          # a trigger found in this share of all prompts (or more) is an ordinary word: kept
REWRITE_ASK = 200              # ask before more than this many new LLM rewrites (--yes skips)

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY, text TEXT, original TEXT, negative TEXT, style TEXT, words INTEGER, nsfw INTEGER,
    nsfw_text INTEGER, reactions_max INTEGER, reactions_sum INTEGER, occurrences INTEGER, base_models TEXT,
    first_seen TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS occurrences (
    occ TEXT PRIMARY KEY, prompt_id TEXT, source TEXT, lora_path TEXT, base_model TEXT, checkpoint TEXT, sampler TEXT,
    steps INTEGER, cfg REAL, weight REAL, nsfw INTEGER, reactions INTEGER, image_id INTEGER, first_seen TEXT,
    last_seen TEXT);
CREATE INDEX IF NOT EXISTS occurrences_prompt ON occurrences(prompt_id);
CREATE TABLE IF NOT EXISTS loras_used (occ TEXT, prompt_id TEXT, lora TEXT, weight REAL, path TEXT);
CREATE INDEX IF NOT EXISTS loras_used_occ ON loras_used(occ);
CREATE TABLE IF NOT EXISTS stripped (occ TEXT, prompt_id TEXT, lora_path TEXT, trigger TEXT);
CREATE INDEX IF NOT EXISTS stripped_occ ON stripped(occ);
CREATE TABLE IF NOT EXISTS facets (prompt_id TEXT, facet TEXT, value TEXT, PRIMARY KEY (prompt_id, facet, value));
CREATE TABLE IF NOT EXISTS rewrites (prompt_id TEXT, mode TEXT, model TEXT, sys_hash TEXT, text TEXT, created_at TEXT,
    PRIMARY KEY (prompt_id, mode, model, sys_hash));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


# ---------------------------------------------------------------------------------------------- where, levels

def library_dir(settings: Settings) -> str:
    d = str(getattr(settings, "prompt_library_dir", "") or "").strip()
    if not d:
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(settings.catalog_path))), "prompts")
    return os.path.expanduser(d)


def library_path(settings: Settings) -> str:
    return os.path.join(library_dir(settings), "_prompt_library.sqlite")


def level_index(value: Any) -> Optional[int]:
    """0..4 for PG .. XXX, BLOCKED for Civitai's 32, None when unknown."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or str(value).strip().isdigit():
        n = int(value)
        if n >= 32:
            return BLOCKED
        for bit, idx in ((16, 4), (8, 3), (4, 2), (2, 1), (1, 0)):
            if n >= bit:
                return idx
        return None
    s = str(value).strip().lower()
    return {"none": 0, "pg": 0, "soft": 1, "pg-13": 1, "pg13": 1, "mature": 2, "r": 2, "x": 3, "xxx": 4}.get(s)


def level_name(idx: Optional[int]) -> str:
    if idx is None:
        return ""
    return "blocked" if idx == BLOCKED else LEVELS[idx]


# ---------------------------------------------------------------------------------------------- text handling

_SPACE = re.compile(r"\s+")


def one_line(text: str) -> str:
    """Newlines become sentence or list breaks, so every prompt is one line (text exports)."""
    parts = [p.strip(" ,") for p in str(text or "").splitlines() if p.strip(" ,")]
    out = ""
    for p in parts:
        if out:
            out += " " if out[-1] in ".!?;:" else ", "
        out += p
    return _SPACE.sub(" ", out).strip()


def tidy(text: str) -> str:
    """Punctuation left behind by stripping: no ', ,', no leading / trailing separators, no space before commas."""
    t = re.sub(r"\s+([,.;:!?])", r"\1", text)
    t = re.sub(r"([,;:])(\s*[,;:])+", r"\1", t)
    t = re.sub(r"\.\s*,", ".", t)
    t = re.sub(r",\s*\.", ".", t)
    t = re.sub(r"^\s*[,.;:]+\s*", "", t)
    t = re.sub(r"\s*[,;:]+\s*$", "", t)
    return _SPACE.sub(" ", t).strip()


def normalise(text: str) -> str:
    return _SPACE.sub(" ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def prompt_id(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()[:16]


def _phrase_re(phrases: Iterable[str]) -> Optional[re.Pattern]:
    ps = sorted({p.strip() for p in phrases if p and p.strip()}, key=len, reverse=True)
    if not ps:
        return None
    return re.compile(r"(?<![\w@#])(" + "|".join(re.escape(p) for p in ps) + r")(?![\w])", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------- catalog reading

def _tables(connection: sqlite3.Connection) -> set:
    return {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}


def _json_list(value: Any) -> List[str]:
    if not value:
        return []
    try:
        v = json.loads(value)
    except (TypeError, ValueError):
        return [s.strip() for s in str(value).split(",") if s.strip()]
    return [str(x) for x in v] if isinstance(v, list) else []


class CatalogView:
    """What the build needs from the catalog, read once."""

    def __init__(self, connection: sqlite3.Connection):
        tables = _tables(connection)
        self.cards: Dict[str, Dict[str, Any]] = {}
        if "card_facets" in tables:
            for r in connection.execute("select path, name, triggers, facets, base_model, arch_folder from card_facets"):
                try:
                    facets = json.loads(r["facets"] or "{}")
                except ValueError:
                    facets = {}
                self.cards[r["path"]] = {"name": r["name"] or "", "triggers": _json_list(r["triggers"]),
                                         "facets": facets if isinstance(facets, dict) else {},
                                         "base": r["base_model"] or "", "arch": r["arch_folder"] or ""}
        self.by_version: Dict[int, str] = {}
        self.by_stem: Dict[str, str] = {}
        self.base_of: Dict[str, str] = {}
        for r in connection.execute("select path, civitai_version_id, stem, base_model from loras"):
            if r["civitai_version_id"]:
                self.by_version[int(r["civitai_version_id"])] = r["path"]
            if r["stem"]:
                self.by_stem[str(r["stem"]).lower()] = r["path"]
            self.base_of[r["path"]] = r["base_model"] or ""
        self.harvest: Dict[int, List[str]] = {}
        if "harvest_images" in tables:
            for image_id, loras in connection.execute("select image_id, loras from harvest_images"):
                try:
                    self.harvest[int(image_id)] = [str(x.get("key") if isinstance(x, dict) else x)
                                                   for x in json.loads(loras or "[]")]
                except (TypeError, ValueError):
                    pass

    def triggers(self, path: str) -> List[str]:
        return (self.cards.get(path) or {}).get("triggers") or []

    def path_for(self, ref: str) -> Optional[str]:
        """A <lora:stem> name or a Civitai key (v:123) -> your file, when you have it."""
        ref = str(ref or "")
        if ref.startswith("v:") and ref[2:].isdigit():
            return self.by_version.get(int(ref[2:]))
        return self.by_stem.get(os.path.splitext(os.path.basename(ref))[0].lower())


def _num(text: Any) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):            # '0.0.5' and other typos in people's prompts
        return None


def _occurrences(connection: sqlite3.Connection, view: CatalogView) -> Iterable[Dict[str, Any]]:
    from .community import LORA_TAG, clean_prompt
    tables = _tables(connection)
    if "examples" in tables:
        for r in connection.execute("select * from examples where prompt is not null and prompt != ''"):
            used = [(m.group(2), _num(m.group(3))) for m in LORA_TAG.finditer(r["prompt"])]
            yield {"occ": f"preview:{r['path']}|{r['media_file']}", "source": "preview", "lora_path": r["path"],
                   "raw": r["prompt"], "cleaned": clean_prompt(r["prompt"]), "negative": r["negative"] or "",
                   "base_model": view.base_of.get(r["path"]) or (view.cards.get(r["path"]) or {}).get("base", ""),
                   "checkpoint": r["base_checkpoint"], "sampler": r["sampler"], "steps": r["steps"], "cfg": r["cfg"],
                   "weight": r["weight"], "nsfw": level_index(r["nsfw_level"]), "reactions": None, "image_id": None,
                   "used": used}
    if "community_prompts" in tables:
        for r in connection.execute("select * from community_prompts where prompt is not null and prompt != ''"):
            path = view.by_version.get(int(r["version_id"])) if r["version_id"] else None
            used = [(m.group(2), _num(m.group(3))) for m in LORA_TAG.finditer(r["prompt"])]
            used += [(k, None) for k in view.harvest.get(int(r["image_id"] or 0), [])]
            yield {"occ": f"community:{r['source']}:{r['image_id']}:{r['version_id']}",
                   "source": f"community-{r['source']}", "lora_path": path, "raw": r["prompt"],
                   "cleaned": r["cleaned"] or clean_prompt(r["prompt"]), "negative": r["negative"] or "",
                   "base_model": r["base_model"] or (view.base_of.get(path, "") if path else ""), "checkpoint": None,
                   "sampler": None, "steps": None, "cfg": None, "weight": r["weight"],
                   "nsfw": level_index(r["nsfw_num"] if "nsfw_num" in r.keys() and r["nsfw_num"] is not None
                                       else r["nsfw_level"]),
                   "reactions": r["reactions"], "image_id": r["image_id"],
                   "used": used}


# ---------------------------------------------------------------------------------------------- build

def _generic_triggers(view: CatalogView, texts: List[str]) -> set:
    """Triggers that are ordinary prompt words: found in GENERIC_SHARE of all prompts or more."""
    cands = {t.lower().strip() for card in view.cards.values() for t in card["triggers"] if t and t.strip()}
    cands = {t for t in cands if 1 <= len(t.split()) <= 3}
    if not cands or not texts:
        return set()
    df: Counter = Counter()
    for text in texts:
        words = re.findall(r"[\w@#'-]+", text.lower())
        seen = set()
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i:i + n])
                if g in cands:
                    seen.add(g)
        df.update(seen)
    limit = max(25, GENERIC_SHARE * len(texts))
    return {t for t, n in df.items() if n >= limit}


def build(settings: Settings, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Merge the catalog's prompts into the library. Safe to run any time; nothing already there is lost."""
    from .community import NSFW_WORDS, prompt_style
    started = time.time()
    os.makedirs(library_dir(settings), exist_ok=True)
    cat = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    cat.row_factory = sqlite3.Row
    lib = sqlite3.connect(library_path(settings), timeout=30)
    lib.executescript(SCHEMA)
    now = datetime.now().isoformat(timespec="seconds")
    try:
        view = CatalogView(cat)
        occs = list(_occurrences(cat, view))
        generic = _generic_triggers(view, [o["cleaned"] for o in occs])
        regex_cache: Dict[Tuple[str, ...], Optional[re.Pattern]] = {}
        known = {r[0] for r in lib.execute("select id from prompts")}
        new_prompts, kept, skipped = 0, 0, 0
        seen_occ = set()
        for o in occs:
            paths = [p for p in [o["lora_path"]] + [view.path_for(ref) for ref, _w in o["used"]] if p]
            paths = list(dict.fromkeys(paths))
            trig: Dict[str, str] = {}                 # lower trigger -> lora path
            for p in paths:
                for t in view.triggers(p):
                    if t and t.strip() and t.strip().lower() not in generic:
                        trig.setdefault(t.strip().lower(), p)
            key = tuple(sorted(trig))
            if key not in regex_cache:
                regex_cache[key] = _phrase_re(key)
            rx = regex_cache[key]
            base = one_line(o["cleaned"])
            removed = []
            if rx is not None:
                removed = [m.group(1).lower() for m in rx.finditer(base)]
                base = rx.sub(" ", base)
            text = tidy(base)
            if len(normalise(text)) < 3:
                skipped += 1
                continue
            pid = prompt_id(text)
            if pid not in known:
                lib.execute("insert into prompts (id, text, original, negative, style, words, nsfw_text, first_seen) "
                            "values (?, ?, ?, ?, ?, ?, ?, ?)",
                            (pid, text, one_line(o["cleaned"]), one_line(o["negative"]), prompt_style(text),
                             len(text.split()), 1 if NSFW_WORDS.search(text) else 0, now))
                known.add(pid)
                new_prompts += 1
            lib.execute("""insert into occurrences (occ, prompt_id, source, lora_path, base_model, checkpoint, sampler,
                           steps, cfg, weight, nsfw, reactions, image_id, first_seen, last_seen)
                           values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           on conflict(occ) do update set prompt_id = excluded.prompt_id, lora_path = excluded.lora_path,
                           base_model = excluded.base_model, nsfw = excluded.nsfw, reactions = excluded.reactions,
                           weight = excluded.weight, last_seen = excluded.last_seen""",
                        (o["occ"], pid, o["source"], o["lora_path"], o["base_model"], o["checkpoint"], o["sampler"],
                         o["steps"], o["cfg"], o["weight"], o["nsfw"], o["reactions"], o["image_id"], now, now))
            lib.execute("delete from loras_used where occ = ?", (o["occ"],))
            lib.executemany("insert into loras_used values (?, ?, ?, ?, ?)",
                            [(o["occ"], pid, ref, w, view.path_for(ref)) for ref, w in o["used"]])
            lib.execute("delete from stripped where occ = ?", (o["occ"],))
            lib.executemany("insert into stripped values (?, ?, ?, ?)",
                            [(o["occ"], pid, trig.get(t, ""), t) for t in dict.fromkeys(removed)])
            if o["lora_path"]:
                for facet, values in ((view.cards.get(o["lora_path"]) or {}).get("facets") or {}).items():
                    for v in values if isinstance(values, list) else [values]:
                        lib.execute("insert or ignore into facets values (?, ?, ?)", (pid, str(facet), str(v)))
            seen_occ.add(o["occ"])
            kept += 1
            if kept % 5000 == 0:
                lib.commit()
                log(f"[prompts] {kept} of {len(occs)} ...")
        # aggregates, from every occurrence the library holds (old ones stay: a prompt is still a good prompt)
        agg = lib.execute("""select prompt_id, max(case when nsfw = 9 then 1 else 0 end), max(nsfw), max(reactions),
                             sum(coalesce(reactions, 0)), count(*), group_concat(distinct base_model), min(first_seen),
                             max(last_seen) from occurrences group by prompt_id""").fetchall()
        lib.executemany("""update prompts set nsfw = ?, reactions_max = ?, reactions_sum = ?, occurrences = ?,
                           base_models = ?, first_seen = ?, last_seen = ? where id = ?""",
                        [(BLOCKED if blk else mx, rmax, rsum, n, bases or "", first, last, pid)
                         for pid, blk, mx, rmax, rsum, n, bases, first, last in agg])
        lib.execute("insert or replace into meta values ('built_at', ?)", (now,))
        lib.commit()
        total = lib.execute("select count(*) from prompts").fetchone()[0]
    finally:
        cat.close()
        lib.close()
    out = {"occurrences": kept, "new_prompts": new_prompts, "prompts": total, "skipped_empty": skipped,
           "generic_triggers_kept": len(generic), "seconds": round(time.time() - started, 1),
           "file": library_path(settings)}
    log(f"[prompts] {total} prompts ({new_prompts} new) from {kept} occurrences in {out['seconds']} s -> {out['file']}")
    return out


# ---------------------------------------------------------------------------------------------- selecting

def _your_ratings(settings: Settings) -> Dict[str, float]:
    """LoRA path -> your stars: the card rating, else the mean stars of your sweep marks with it."""
    out: Dict[str, float] = {}
    cat = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    try:
        tables = _tables(cat)
        if "stacks" in tables:
            acc: Dict[str, List[float]] = defaultdict(list)
            for loras, stars in cat.execute("select loras, stars from stacks where stars is not null"):
                for e in json.loads(loras or "[]"):
                    if e.get("path") and not e.get("panel"):
                        acc[e["path"]].append(float(stars))
            out.update({p: sum(v) / len(v) for p, v in acc.items()})
        if "card_facets" in tables:
            for path, rating in cat.execute("select path, rating from card_facets where rating is not null"):
                out[path] = float(rating)
    finally:
        cat.close()
    return out


def select(settings: Settings, base: str = "", max_nsfw: str = "X", clean_text: bool = False, min_reactions: int = 0,
           lora_rating: float = 0, style: str = "", facets: Optional[List[str]] = None, min_words: int = 0,
           max_words: int = 0, source: str = "", sample: int = 0, seed: int = 0, top: int = 0,
           text: Optional[List[str]] = None, order: str = "likes") -> List[Dict[str, Any]]:
    lib = sqlite3.connect(f"file:{library_path(settings)}?mode=ro", uri=True, timeout=30)
    lib.row_factory = sqlite3.Row
    try:
        cap = level_index(max_nsfw)
        if cap is None:
            raise ValueError(f"--max-nsfw must be one of {', '.join(LEVELS)}")
        where, args = ["(nsfw is null or nsfw <= ?)" if cap >= 3 else "nsfw <= ?"], [cap]
        if clean_text:
            where.append("nsfw_text = 0")
        if min_reactions:
            where.append("coalesce(reactions_max, 0) >= ?")
            args.append(int(min_reactions))
        if style:
            where.append("style = ?")
            args.append(style)
        if min_words:
            where.append("words >= ?")
            args.append(int(min_words))
        if max_words:
            where.append("words <= ?")
            args.append(int(max_words))
        if base:
            where.append("id in (select prompt_id from occurrences where lower(base_model) like ?)")
            args.append(f"%{base.lower()}%")
        if source:
            where.append("id in (select prompt_id from occurrences where source like ?)")
            args.append(f"{source}%")
        for f in facets or []:
            k, eq, v = f.partition("=")
            if not eq:                                # 2026-10-29: --facet surreal = that value in any facet
                where.append("id in (select prompt_id from facets where lower(value) like ?)")
                args.append("%" + k.strip().strip('"').lower().replace(" ", "_") + "%")
                continue
            where.append("id in (select prompt_id from facets where lower(facet) = ? and lower(value) like ?)")
            args += [k.strip().lower(), "%" + v.strip().strip('"').lower().replace(" ", "_") + "%"]
        for word in text or []:                       # 2026-10-29: --text surreal = the prompt itself says it
            where.append("lower(text) like ?")
            args.append("%" + word.strip().strip('"').lower() + "%")
        # order (2026-10-29): likes = most community likes first; newest = most recently posted on Civitai first
        # (Civitai image ids only grow, so the highest image id = the newest image; card previews have none)
        by = ("(select max(coalesce(image_id, 0)) from occurrences o where o.prompt_id = prompts.id) desc"
              if order == "newest" else "coalesce(reactions_max, 0) desc")
        rows = [dict(r) for r in lib.execute(f"select * from prompts where {' and '.join(where)} "
                                              f"order by {by}, id", args)]
        if lora_rating:
            ratings = _your_ratings(settings)
            came = defaultdict(set)
            for pid, path in lib.execute("select prompt_id, lora_path from occurrences where lora_path is not null"):
                came[pid].add(path)
            rows = [r for r in rows if any(ratings.get(p, 0) >= lora_rating for p in came.get(r["id"], ()))]
        if top:                                   # 2026-10-29: the N with the most community likes (rows are sorted so)
            rows = rows[:top]
        if sample and len(rows) > sample:
            rows = random.Random(seed).sample(rows, sample)
        _attach(lib, rows)
    finally:
        lib.close()
    return rows


def _attach(lib: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    """Facets, LoRAs it came with / used, stripped triggers - for the rich formats."""
    for r in rows:
        r["nsfw_level"] = level_name(r["nsfw"])
        r["facets"] = {}
        for facet, value in lib.execute("select facet, value from facets where prompt_id = ?", (r["id"],)):
            r["facets"].setdefault(facet, []).append(value)
        r["came_with"] = sorted({os.path.splitext(os.path.basename(p))[0] for (p,) in lib.execute(
            "select distinct lora_path from occurrences where prompt_id = ? and lora_path is not null", (r["id"],))})
        r["loras_used"] = sorted({l for (l,) in lib.execute("select distinct lora from loras_used where prompt_id = ?",
                                                             (r["id"],))})
        r["triggers_stripped"] = sorted({t for (t,) in lib.execute("select distinct trigger from stripped "
                                                                    "where prompt_id = ?", (r["id"],))})


def stats(settings: Settings) -> str:
    lib = sqlite3.connect(f"file:{library_path(settings)}?mode=ro", uri=True, timeout=30)
    try:
        total = lib.execute("select count(*) from prompts").fetchone()[0]
        built = (lib.execute("select v from meta where k = 'built_at'").fetchone() or ["?"])[0]
        lines = [f"{total} prompts in {library_path(settings)} (built {built})", "", "by NSFW level (worst image seen):"]
        for n, c in lib.execute("select nsfw, count(*) from prompts group by nsfw order by nsfw"):
            lines.append(f"  {level_name(n) or 'unknown':8s} {c}")
        lines += ["", "by style:"]
        lines += [f"  {s:8s} {c}" for s, c in lib.execute("select style, count(*) from prompts group by style")]
        lines += ["", "by source (occurrences):"]
        lines += [f"  {s:18s} {c}" for s, c in lib.execute("select source, count(*) from occurrences group by source")]
        lines += ["", "top base models (prompts):"]
        lines += [f"  {b or '(unknown)':20s} {c}" for b, c in lib.execute(
            "select base_model, count(distinct prompt_id) from occurrences group by base_model order by 2 desc limit 12")]
        lines.append(f"\nrewrites cached: {lib.execute('select count(*) from rewrites').fetchone()[0]}")
    finally:
        lib.close()
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- rewriting

TIDY_SYSTEM = (
    "You clean up prompts for an image model. Rewrite the prompt you are given as clear, natural English describing "
    "the same image: turn comma-separated tag lists into flowing sentences, drop quality tags (masterpiece, best "
    "quality, 8k, highres, absurdres, score_9 ...), drop weights and brackets, fix grammar. Keep every subject, "
    "object, setting, style, colour and composition detail that is there. Do not add anything new. Output only the "
    "rewritten prompt, nothing else.")

FALLBACK_EXPANDER = (
    "You are a prompt expander for the Krea 2 image model. Given a short user prompt, rewrite it into a rich, flowing "
    "natural-language image description: the main subject and what it is doing first, then setting, composition and "
    "camera, lighting, colour palette, and finally the style or medium. Keep everything the user asked for, add only "
    "concrete visual detail that fits it, and keep the description under about 350 words. Output only the expanded "
    "prompt, with no preamble or explanation.")


def krea2_expander_system() -> Tuple[str, str]:
    """The Eric_Krea2 Magic Prompt node's built-in expander system prompt (read from its source), else a fallback."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "Eric_Krea2", "nodes", "krea2_prompt.py")
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_DEFAULT_EXPANDER_SYSTEM"
                                                    for t in node.targets):
                return ast.literal_eval(node.value), "Eric_Krea2 Magic Prompt"
    except (OSError, SyntaxError, ValueError):
        pass
    return FALLBACK_EXPANDER, "built-in fallback"


def _clean_answer(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    lines = [l for l in t.splitlines() if l.strip()]
    if lines and lines[0].lower().rstrip(":").startswith(("here is", "here's", "rewritten prompt", "expanded prompt",
                                                          "sure", "certainly")):
        lines = lines[1:]
    return one_line(" ".join(lines).strip().strip('"').strip())


def rewrite(settings: Settings, rows: List[Dict[str, Any]], mode: str, system_file: str = "", yes: bool = False,
            expand_under: int = 25, log: Callable[[str], None] = print) -> int:
    """Fills r['rewrite'] for every row (cached). Returns the number of new LLM calls."""
    from .llm import chat
    if mode == "tidy":
        system, origin = TIDY_SYSTEM, "tidy"
    else:
        system, origin = krea2_expander_system()
    if system_file:
        system, origin = open(system_file, encoding="utf-8").read().strip(), os.path.basename(system_file)
    model = (getattr(settings, "prompt_model", "") or getattr(settings, "genre_model", "")
             or getattr(settings, "card_model", ""))
    sys_hash = hashlib.sha1(system.encode("utf-8")).hexdigest()[:10]
    lib = sqlite3.connect(library_path(settings), timeout=30)
    try:
        cached = {pid: text for pid, text in lib.execute(
            "select prompt_id, text from rewrites where mode = ? and model = ? and sys_hash = ?", (mode, model, sys_hash))}
        todo = []
        for r in rows:
            if mode == "expand" and r["words"] >= expand_under:
                r["rewrite"] = r["text"]
            elif r["id"] in cached:
                r["rewrite"] = cached[r["id"]]
            else:
                todo.append(r)
        if not todo:
            return 0
        log(f"[prompts] rewrite '{mode}' with {model} ({origin}): {len(todo)} new, {len(rows) - len(todo)} cached")
        if len(todo) > REWRITE_ASK and not yes:
            if not sys.stdin or not sys.stdin.isatty():
                raise SystemExit(f"{len(todo)} new rewrites - add --yes to go ahead")
            if input(f"{len(todo)} new rewrites (a few seconds each). Go ahead? [y/N] ").strip().lower() != "y":
                raise SystemExit("stopped")
        t0, done = time.time(), 0
        for r in todo:
            try:
                answer = _clean_answer(chat(settings, model, [{"role": "system", "content": system},
                                                              {"role": "user", "content": r["text"]}],
                                            temperature=0.3, max_tokens=800))
            except Exception as error:                      # one failed call does not stop the export
                log(f"[prompts] rewrite failed ({type(error).__name__}: {error}); keeping the original")
                answer = ""
            r["rewrite"] = answer or r["text"]
            if answer:
                lib.execute("insert or replace into rewrites values (?, ?, ?, ?, ?, ?)",
                            (r["id"], mode, model, sys_hash, answer, datetime.now().isoformat(timespec="seconds")))
            done += 1
            if done == 3 or done % 25 == 0:
                lib.commit()
                rate = (time.time() - t0) / done
                log(f"[prompts] {done}/{len(todo)} rewritten, about {rate * (len(todo) - done) / 60:.0f} min left")
        lib.commit()
        return done
    finally:
        lib.close()


# ---------------------------------------------------------------------------------------------- export

def _slug(value: str) -> str:
    return re.sub(r"[^\w-]+", "_", str(value)).strip("_")[:60] or "other"


def export(settings: Settings, rows: List[Dict[str, Any]], out: str, fmt: str = "", with_triggers: bool = False,
           split_by: str = "", log: Callable[[str], None] = print) -> List[str]:
    """Writes the rows; returns the files written."""
    if not out:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(library_dir(settings), f"prompts_{stamp}.{fmt or 'txt'}")
    fmt = (fmt or os.path.splitext(out)[1].lstrip(".") or "txt").lower()
    if fmt not in ("txt", "jsonl", "csv", "sqlite"):
        raise ValueError("--format must be txt, jsonl, csv or sqlite")

    def line(r: Dict[str, Any]) -> str:
        if r.get("rewrite"):
            return r["rewrite"]
        return r["original"] if with_triggers else r["text"]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    if split_by:
        for r in rows:
            for v in r["facets"].get(split_by) or ["other"]:
                groups.setdefault(_slug(v), []).append(r)
        os.makedirs(out, exist_ok=True)
        targets = {g: os.path.join(out, f"{g}.{fmt}") for g in groups}
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        groups, targets = {"": rows}, {"": out}
    written = []
    for g, items in groups.items():
        path = targets[g]
        if fmt == "txt":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(line(r) for r in items) + "\n")
        elif fmt == "jsonl":
            with open(path, "w", encoding="utf-8") as fh:
                for r in items:
                    fh.write(json.dumps({**{k: r.get(k) for k in ("id", "text", "original", "negative", "style",
                                                                  "words", "nsfw_level", "reactions_max",
                                                                  "occurrences", "base_models", "came_with",
                                                                  "loras_used", "triggers_stripped", "facets")},
                                         "prompt": line(r), "rewrite": r.get("rewrite")}, ensure_ascii=False) + "\n")
        elif fmt == "csv":
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["prompt", "original", "negative", "style", "words", "nsfw", "reactions_max",
                            "occurrences", "base_models", "came_with", "triggers_stripped", "facets", "id"])
                for r in items:
                    w.writerow([line(r), r["original"], r["negative"], r["style"], r["words"], r["nsfw_level"],
                                r["reactions_max"], r["occurrences"], r["base_models"], "; ".join(r["came_with"]),
                                "; ".join(r["triggers_stripped"]),
                                "; ".join(f"{k}={','.join(v)}" for k, v in r["facets"].items()), r["id"]])
        else:
            if os.path.exists(path):
                os.remove(path)
            db = sqlite3.connect(path)
            db.execute("create table prompts (id TEXT PRIMARY KEY, prompt TEXT, original TEXT, negative TEXT, style TEXT, "
                       "words INTEGER, nsfw TEXT, reactions_max INTEGER, occurrences INTEGER, base_models TEXT, "
                       "came_with TEXT, triggers_stripped TEXT, facets TEXT)")
            db.executemany("insert into prompts values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [(r["id"], line(r), r["original"], r["negative"], r["style"], r["words"], r["nsfw_level"],
                             r["reactions_max"], r["occurrences"], r["base_models"], json.dumps(r["came_with"]),
                             json.dumps(r["triggers_stripped"]), json.dumps(r["facets"])) for r in items])
            db.commit()
            db.close()
        written.append(path)
    log(f"[prompts] wrote {len(rows)} prompts to {len(written)} file(s): {written[0] if len(written) == 1 else out}")
    return written
