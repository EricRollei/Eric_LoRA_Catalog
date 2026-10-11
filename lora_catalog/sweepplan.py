"""
Writes LoRA stack sweeps for the Eric_Krea2 SweepPlan node (lora_mode = stacks), picked from the catalog (2026-10-09).
Spec: Eric_Krea2 docs/spec_lora_stack_sweeps_2026-10-09.md, section 2.2, with Eric's decisions (section 5).

    python run.py sweep-plan --folders Krea2/graphic-art Krea2/art-style
        3 pairs inside each folder + 3 pairs across them (Eric's first test), written as a sweep preset
    python run.py sweep-plan --folders Krea2/graphic-art Krea2/surreal --mode across --pairs 8
    python run.py sweep-plan --folders Krea2/Comic --anchor "Kaos" --pairs 6      one LoRA with six partners
    python run.py sweep-plan --folders Krea2/Comic --dry-run                       print only, write nothing

Which LoRAs (candidates): catalogued LoRAs under the folders whose card roles include --roles (default style);
never your 1-2 star ones; NSFW-rated cards only with --nsfw.
Which pairs: every allowed pair is scored, then the best are taken with each LoRA used at most --max-uses times (so a
sweep covers many LoRAs, not one favourite over and over). Score = your stars for each (4-5 star first; unrated is
neutral) + how often Civitai users put the two together (lora_pairs from run.py harvest) + a small random part
(--seed, so the same command gives the same sweep and a new seed gives fresh pairs). Pairs you have already marked
on the Sweeps page are left out (--retest to include them).
Weights: each LoRA at its card's typical weight (your edited weight wins); the second LoRA of a pair is scaled by
--blend (0.6 = the Suggester's same-family blend), so the sweep tests what the Suggester would actually make.
Names in the lines are file names (the sweep finds them in any subfolder), so presets keep working while you sort.

Output: a preset in Eric_Krea2's sweep_presets.json (backup first) - pick it in the SweepPlan node's sweep_preset list.
Every grid axis is stored blank so the panel's sampler/scheduler lists cannot multiply the cells.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import shutil
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings

# Three prompts per kind of LoRA, chosen to stress it (spec section 3); the first folder word that matches wins.
PROMPTS = {
    "comic": ["a superhero landing on a rain-soaked rooftop at night, dynamic pose",
              "close-up portrait of a weathered sea captain with a scar, dramatic expression",
              "a crowded market street in a fantasy city, wide establishing shot"],
    "illustration": ["a fox reading a book under a lamp in a cosy library",
                     "portrait of a young woman with a flower crown, three-quarter view",
                     "a lighthouse on a cliff during a storm, wide view"],
    "graphic": ["a concert poster for a jazz festival with the title MIDNIGHT BLUE",
                "a tarot card of The Moon with an ornate border",
                "a magazine cover with a fashion model and bold headline type"],
    "art-style": ["portrait of an old man in a wool coat by a window",
                  "a mountain lake at dawn with mist, wide landscape",
                  "still life of fruit, a copper jug and a folded cloth on a table"],
    "surreal": ["a whale swimming through clouds above a quiet village",
                "a woman whose hair turns into a flock of birds, portrait",
                "a staircase that loops back into itself in an empty desert"],
}
GENERIC = ["close-up portrait of a person with expressive eyes",
           "two figures running through a city street in the rain, action scene",
           "a city skyline at dusk, wide establishing shot"]
GRID_FIELDS_BLANK = ["samplers", "schedulers", "steps", "windows", "etas", "noise_types", "cfgs", "shift_modes", "mus",
                     "combos_json", "combos_file", "megapixels", "hybrid_steps", "sigma_profiles"]
STAR_SCORE = {5: 3.0, 4: 2.0, 3: 1.0}
COMMUNITY_TIE = 0.25          # unrated only: +-0.25 by Civitai score (a star step is 1.0; the random part is 0-0.75)


def _json_list(value: Any) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _json_dict(value: Any) -> dict:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


class Candidate:
    def __init__(self, row: sqlite3.Row, line_name: str):
        self.path = row["path"]
        self.name = row["name"] or row["stem"]
        self.line_name = line_name
        self.folder = row["rel_dir"]
        self.sha = (row["key"] or "").lower()
        self.version_id = row["civitai_version_id"]
        self.model_id = row["civitai_model_id"]
        self.rating = row["rating"]
        weight = row["weight_typical"]
        self.weight = round(float(weight), 2) if weight not in (None, 0) else 0.8
        self.score = STAR_SCORE.get(row["rating"] or 0, 0.0)

    def __repr__(self) -> str:
        return f"{self.name} @{self.weight:g}"


def _candidates(connection: sqlite3.Connection, folder: str, roles: List[str], nsfw: bool,
                stem_counts: Dict[str, int]) -> List[Candidate]:
    rows = connection.execute("""
        select f.path, f.name, f.key, f.rel_dir, f.rating, f.weight_typical, f.nsfw, f.facets, f.role,
               l.stem, l.file_name, l.civitai_version_id, l.civitai_model_id, l.root
        from card_facets f join loras l on l.path = f.path
        where lower(replace(f.rel_dir, '\\', '/')) = lower(?) or lower(replace(f.rel_dir, '\\', '/')) like lower(?)""",
                              (folder, folder.rstrip("/") + "/%")).fetchall()
    out = []
    for row in rows:
        card_roles = set(_json_dict(row["facets"]).get("role") or []) | {row["role"] or ""}
        if roles and not card_roles & set(roles):
            continue
        if row["rating"] and row["rating"] <= 2:
            continue
        if (row["nsfw"] or 0) >= 2 and not nsfw:
            continue
        stem = row["stem"] or os.path.splitext(row["file_name"])[0]
        # the sweep finds a bare file name in any subfolder; a name two files share needs the folder path
        line_name = stem if stem_counts.get(stem.lower(), 0) == 1 else \
            os.path.relpath(row["path"], row["root"]) if row["root"] else row["path"]
        out.append(Candidate(row, line_name))
    return out


def _civitai_bonus(connection: sqlite3.Connection) -> Dict[Tuple[int, int], float]:
    """Harvested co-use (run.py pairs): {(version a, version b): bonus 0..2}."""
    bonus = {}
    try:
        for row in connection.execute("select a, b, score from lora_pairs where owned_a = 1 and owned_b = 1"):
            if str(row["a"]).startswith("v:") and str(row["b"]).startswith("v:"):
                a, b = int(row["a"][2:]), int(row["b"][2:])
                bonus[(min(a, b), max(a, b))] = min(2.0, max(0.0, float(row["score"] or 0)) / 4.0)
    except sqlite3.Error:
        pass
    return bonus


def _tested(connection: sqlite3.Connection) -> set:
    """Pairs (sha256 sets) already marked on the Sweeps page."""
    done = set()
    try:
        for (loras,) in connection.execute("select loras from stacks where n_loras > 1"):
            swept = [l.get("sha256") for l in _json_list(loras) if not l.get("panel") and l.get("sha256")]
            for a, b in itertools.combinations(sorted(set(swept)), 2):
                done.add((a, b))
    except sqlite3.Error:
        pass
    return done


LEVEL_RANK = {"None": 0, "Soft": 1, "Mature": 2, "X": 3}
MOJIBAKE = re.compile("[├┬┐└Ãâ€][\u0080-⿿]|Â[\u0080-ÿ]")   # UTF-8 read as cp1252/cp437


def _test_prompts(connection: sqlite3.Connection, version_id: Optional[int], limit: int) -> List[str]:
    """A LoRA's picked test prompts (run.py community): natural language first, then the creator's, then the rest."""
    if not version_id:
        return []
    try:
        rows = connection.execute(
            "select cleaned, prompt_style, nsfw_level, source from community_prompts where version_id = ? and picked = 1",
            (version_id,)).fetchall()
    except sqlite3.Error:
        return []
    order = {"creator": 0, "top": 1, "newest": 2}
    # very long prompts and trigger-only ones ("Film Still by X.") last: they test the prompt, not the LoRA
    def awkward(text: str) -> int:
        return 2 if len(text.split()) < 6 else 1 if len(text) > 900 else 0

    rows = sorted(rows, key=lambda r: (awkward(r[0] or ""), r[1] != "natural", order.get(r[3], 3),
                                       LEVEL_RANK.get(r[2], 0)))
    out = []
    for r in rows:
        text = (r[0] or "").replace("\n", " ")
        if not text or MOJIBAKE.search(text):            # text broken by a wrong encoding on Civitai: skip it
            continue
        text = re.sub(r"\*{2,}", " ", text)               # words Civitai censored ('****')
        text = re.sub(r"\s{2,}", " ", text).strip(" ,.")
        if text:
            out.append(text)
    return out[:limit]


def _prompts_for(folders: List[str]) -> List[str]:
    picked: List[str] = []
    for folder in folders:
        low = folder.lower()
        for word, prompts in PROMPTS.items():
            if word in low:
                picked.append(prompts[len(picked) % len(prompts)] if len(folders) > 1 else prompts[0])
                if len(folders) == 1:
                    return prompts
                break
    for prompt in GENERIC:
        if len(picked) >= 3:
            break
        if prompt not in picked:
            picked.append(prompt)
    return picked[:3]


def plan(settings: Settings, folders: List[str], mode: str = "", pairs: int = 9, anchor: str = "",
         roles: Optional[List[str]] = None, blend: float = 0.6, max_uses: int = 2, seed: int = 1,
         retest: bool = False, nsfw: bool = False, prompts: Optional[List[str]] = None, seeds: str = "101,102",
         multipliers: str = "", max_cells: int = 200, name: str = "") -> Dict[str, Any]:
    """Pick the stacks and build the preset. Returns {name, preset, lines, cells, notes}; raises ValueError on a
    request that cannot be met."""
    folders = [f.replace("\\", "/").strip("/") for f in folders if f.strip()]
    if not folders:
        raise ValueError("give at least one folder (--folders Krea2/graphic-art ...)")
    roles = roles if roles is not None else ["style"]
    mode = mode or ("anchor" if anchor else ("within,across" if len(folders) > 1 else "within"))
    modes = [m.strip() for m in mode.split(",") if m.strip()]
    rng = random.Random(seed)
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        stem_counts: Dict[str, int] = {}
        for (stem,) in connection.execute("select lower(stem) from loras"):
            stem_counts[stem] = stem_counts.get(stem, 0) + 1
        pools = {folder: _candidates(connection, folder, roles, nsfw, stem_counts) for folder in folders}
        kept: set = set()                               # a LoRA with copies in two folders is one LoRA: keep the first
        for folder in folders:
            unique = []
            for c in pools[folder]:
                ids = {f"sha:{c.sha}" if c.sha else f"path:{c.path}"} | ({f"v:{c.version_id}"} if c.version_id else set())
                if ids & kept:
                    continue
                kept |= ids
                unique.append(c)
            pools[folder] = unique
        for folder, pool in pools.items():
            if not pool:
                raise ValueError(f"no usable LoRAs under {folder} (roles {roles}; check the folder name and cards)")
        civitai = _civitai_bonus(connection)
        tested = set() if retest else _tested(connection)
        try:                                             # unrated LoRAs: the Civitai community score as a tie-break
            community = dict(connection.execute("select version_id, score from community_stats where score is not null"))
        except sqlite3.Error:
            community = {}
        for pool in pools.values():
            for c in pool:
                if not c.rating and c.version_id in community:
                    c.score += COMMUNITY_TIE * (community[c.version_id] - 50.0) / 50.0
    finally:
        connection.close()

    def pair_score(a: Candidate, b: Candidate) -> float:
        key = (min(a.version_id or 0, b.version_id or 0), max(a.version_id or 0, b.version_id or 0))
        return a.score + b.score + civitai.get(key, 0.0) + rng.random() * 0.75

    def allowed(a: Candidate, b: Candidate) -> bool:
        if a.path == b.path or (a.sha and a.sha == b.sha):          # a LoRA and its own copy in another folder
            return False
        if a.model_id and a.model_id == b.model_id:                 # two versions of one Civitai model
            return False
        return retest or tuple(sorted((a.sha, b.sha))) not in tested

    groups: List[Tuple[str, List[Tuple[Candidate, Candidate]]]] = []
    anchor_path = None
    if "anchor" in modes:
        everyone = [c for pool in pools.values() for c in pool]
        hits = [c for c in everyone if anchor.lower() in c.name.lower() or anchor.lower() in c.line_name.lower()]
        if not hits:
            raise ValueError(f"no LoRA in those folders matches --anchor {anchor!r}")
        lead = hits[0]
        anchor_path = lead.path
        groups.append((f"{lead.name} + partners", [(lead, c) for c in everyone if allowed(lead, c)]))
    if "within" in modes:
        for folder, pool in pools.items():
            groups.append((f"within {folder}", [(a, b) for a, b in itertools.combinations(pool, 2) if allowed(a, b)]))
    if "across" in modes:
        if len(folders) < 2:
            raise ValueError("--mode across needs two folders")
        for f1, f2 in itertools.combinations(folders, 2):
            groups.append((f"{f1} x {f2}", [(a, b) for a in pools[f1] for b in pools[f2] if allowed(a, b)]))
    if not groups:
        raise ValueError(f"unknown --mode {mode!r} (within, across, anchor)")

    # share the pairs between the groups, best first, each LoRA at most max_uses times overall
    per_group = [pairs // len(groups) + (1 if i < pairs % len(groups) else 0) for i in range(len(groups))]
    uses: Dict[str, int] = {}
    chosen: List[Tuple[str, Candidate, Candidate, float]] = []
    for (label, options), want in zip(groups, per_group):
        ranked = sorted(((pair_score(a, b), a, b) for a, b in options), key=lambda item: -item[0])
        got = 0
        for score, a, b in ranked:
            if got >= want:
                break
            if any(uses.get(c.path, 0) >= max_uses for c in (a, b) if c.path != anchor_path):
                continue                                 # the anchor itself may appear in every pair
            if a.score < b.score and a.path != anchor_path:   # the stronger (higher-rated) LoRA leads at full weight
                a, b = b, a
            chosen.append((label, a, b, score))
            uses[a.path] = uses.get(a.path, 0) + 1
            uses[b.path] = uses.get(b.path, 0) + 1
            got += 1
    if not chosen:
        raise ValueError("no untested pairs left in those folders (use --retest, or other folders)")

    lines, singles = [], set()
    current = None
    for label, a, b, score in chosen:
        if label != current:
            lines.append(f"# --- {label}")
            current = label
        second = round(b.weight * blend, 2)
        lines.append(f"# {a.name}  +  {b.name}")
        lines.append(f"{a.line_name} @{a.weight:g} | {b.line_name} @{second:g}")
        singles.update({(a.line_name, a.weight), (b.line_name, second)})
    if prompts == ["from-loras"]:                       # three of the sweep's own LoRAs' test prompts, best LoRAs first
        connection = sqlite3.connect(settings.catalog_path)
        try:
            picked: List[str] = []
            members: List[Candidate] = []
            for _label, a, b, _score in sorted(chosen, key=lambda item: -item[3]):
                members += [c for c in (a, b) if c.path not in {m.path for m in members}]
            for depth in (1, 2, 3):                      # one from each LoRA first, then seconds, then thirds
                for c in members:
                    for p in _test_prompts(connection, c.version_id, depth)[depth - 1:]:
                        if p not in picked and len(picked) < 3:
                            picked.append(p)
        finally:
            connection.close()
        prompts = picked + [p for p in _prompts_for(folders) if p not in picked][:max(0, 3 - len(picked))]
        from_loras = len(picked)
    else:
        from_loras = -1
    prompts = prompts or _prompts_for(folders)
    seed_list = [s for s in (x.strip() for x in seeds.split(",")) if s]
    mults = [m for m in (x.strip() for x in multipliers.split(",")) if m]
    configs = 1 + (len(singles) + len(chosen)) * max(1, len(mults))   # the no-LoRA cell is not multiplied
    cells = configs * max(1, len(prompts)) * max(1, len(seed_list))
    notes = (f"Stack sweep from the LoRA Catalog ({datetime.now():%Y-%m-%d}): {len(chosen)} pairs from "
             f"{', '.join(folders)} ({mode}); no-LoRA + each LoRA alone + each pair; second LoRA x{blend:g}; "
             f"{len(prompts)} prompts x {len(seed_list) or 1} seeds = {cells} cells. Mark the cells on the card "
             f"editor's Sweeps page (G good / B bad / U unsure).")
    if from_loras >= 0:
        notes += (f" Prompts: {from_loras} from the LoRAs' own test prompts" +
                  (f", {len(prompts) - from_loras} generic (no community prompts for the rest yet - run.py community)"
                   if from_loras < len(prompts) else "") + ".")
    if cells > max_cells:
        raise ValueError(f"{cells} cells is over --max-cells {max_cells}: use fewer --pairs, prompts or seeds "
                         f"({configs} LoRA configs x {len(prompts)} prompts x {len(seed_list) or 1} seeds)")
    name = name or f"stacks {' x '.join(f.split('/')[-1] for f in folders)} {datetime.now():%m-%d %H%M}"
    sweep = {field: "" for field in GRID_FIELDS_BLANK}
    sweep.update({"mode": "grid", "stage_scope": "s1", "lora_mode": "stacks", "lora_files": "\n".join(lines),
                  "lora_strengths": ", ".join(mults), "include_baseline": True, "keep_panel_stack": False,
                  "lora_triggers": "append", "prompts": "\n".join(prompts),
                  "seed_policy": "seeds_axis" if len(seed_list) > 1 else "fixed", "seeds": ", ".join(seed_list),
                  "quick_screen": False, "score_metrics": True, "sort_by": "plan_order", "max_combos": cells,
                  "preset_notes": notes})
    return {"name": name, "preset": {"sweep": sweep}, "lines": lines, "cells": cells, "notes": notes,
            "pairs": [(a.name, b.name, round(s, 2)) for _l, a, b, s in chosen]}


def plan_single(settings: Settings, name: str, seeds: str = "101,102", max_cells: int = 200,
                preset_name: str = "") -> Dict[str, Any]:
    """One LoRA's characterization sheet: its own test prompts (up to 4) x three weights (what Civitai users use,
    else the card's low / typical / high) + the no-LoRA reference, as a bakeoff preset."""
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """select l.path, l.stem, l.root, l.civitai_version_id, coalesce(l.model_name, l.stem) name,
                      f.weight_low, f.weight_typical, f.weight_high, s.weight_n, s.weight_p25, s.weight_median, s.weight_p75
               from loras l left join card_facets f on f.path = l.path
               left join community_stats s on s.version_id = l.civitai_version_id
               where l.model_name like ? or l.stem like ? order by l.civitai_version_id is null, l.rel_dir limit 1""",
            (f"%{name}%", f"%{name}%")).fetchone()
        if row is None:
            raise ValueError(f"no LoRA matches {name!r}")
        stems = connection.execute("select count(*) from loras where lower(stem) = lower(?)", (row["stem"],)).fetchone()[0]
        prompts = _test_prompts(connection, row["civitai_version_id"], 4)
    finally:
        connection.close()
    if row["weight_n"] and row["weight_n"] >= 10:
        weights, source = [row["weight_p25"], row["weight_median"], row["weight_p75"]], "Civitai users"
    else:
        weights = [row["weight_low"], row["weight_typical"], row["weight_high"]]
        weights, source = ([w for w in weights if w is not None] or [0.6, 0.8, 1.0]), "the card"
    weights = sorted({round(float(w), 2) for w in weights})
    if len(weights) < 3:                                 # users agree on one or two weights: add a push past them
        weights.append(round(min(1.5, weights[-1] + 0.25), 2))
        weights = sorted(set(weights))
    if not prompts:
        prompts, note = GENERIC, "no test prompts collected yet (run.py community) - generic prompts"
    else:
        note = f"{len(prompts)} of its own test prompts"
    seed_list = [s for s in (x.strip() for x in seeds.split(",")) if s]
    cells = (1 + len(weights)) * len(prompts) * max(1, len(seed_list))
    if cells > max_cells:
        raise ValueError(f"{cells} cells is over --max-cells {max_cells}")
    line_name = row["stem"] if stems == 1 else os.path.relpath(row["path"], row["root"])
    sweep = {field: "" for field in GRID_FIELDS_BLANK}
    sweep.update({"mode": "grid", "stage_scope": "s1", "lora_mode": "bakeoff", "lora_files": line_name,
                  "lora_strengths": ", ".join(f"{w:g}" for w in weights), "include_baseline": True,
                  "keep_panel_stack": False, "lora_triggers": "append", "prompts": "\n".join(prompts),
                  "seed_policy": "seeds_axis" if len(seed_list) > 1 else "fixed", "seeds": ", ".join(seed_list),
                  "quick_screen": False, "score_metrics": True, "sort_by": "plan_order", "max_combos": cells,
                  "preset_notes": f"Characterization of {row['name']} ({datetime.now():%Y-%m-%d}): {note}, weights "
                                  f"{', '.join(f'{w:g}' for w in weights)} (from {source}) + no-LoRA, = {cells} cells."})
    return {"name": preset_name or f"single {row['name'][:40]} {datetime.now():%m-%d %H%M}", "preset": {"sweep": sweep},
            "lines": [line_name], "cells": cells, "notes": sweep["preset_notes"], "pairs": []}


def write_preset(settings: Settings, result: Dict[str, Any]) -> str:
    """Add the preset to sweep_presets.json (a dated backup of the file first). Returns the backup path."""
    path = settings.krea2_sweep_presets
    if not path:
        raise SystemExit("[sweep-plan] no preset file: set krea2_sweep_presets in settings.json (Eric_Krea2's "
                         "sweep_presets.json; found by itself when this tool sits in ComfyUI's custom_nodes)")
    data: Dict[str, Any] = {}
    backup = ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        backup_dir = os.path.join(os.path.dirname(path), "docs", "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup = os.path.join(backup_dir, f"sweep_presets_{datetime.now():%Y-%m-%d_%H%M%S}.json")
        shutil.copy2(path, backup)
    data[result["name"]] = result["preset"]
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    return backup


def run(settings: Settings, args: Any) -> None:
    try:
        if args.single:
            result = plan_single(settings, args.single, seeds=args.seeds, max_cells=args.max_cells, preset_name=args.name)
        else:
            if not args.folders:
                raise ValueError("give --folders (or --single NAME for one LoRA's characterization sheet)")
            result = plan(settings, args.folders, mode=args.mode, pairs=args.pairs, anchor=args.anchor,
                      roles=[r for r in args.roles.split(",") if r] if args.roles != "any" else [],
                      blend=args.blend, max_uses=args.max_uses, seed=args.seed, retest=args.retest, nsfw=args.nsfw,
                      prompts=[p for p in args.prompts.split("|") if p.strip()] if args.prompts else None,
                      seeds=args.seeds, multipliers=args.multipliers, max_cells=args.max_cells, name=args.name)
    except ValueError as error:
        print(f"[sweep-plan] {error}")
        raise SystemExit(1)
    print(f"[sweep-plan] {result['name']}: {len(result['pairs'])} pairs, {result['cells']} cells")
    for a, b, score in result["pairs"]:
        print(f"   {score:5.2f}  {a}  +  {b}")
    if args.single or "Prompts:" in result["notes"]:
        print(f"   {result['notes']}")
    print("\n   lora_files:\n" + "\n".join("     " + line for line in result["lines"]))
    print("\n   prompts:\n" + "\n".join("     " + p for p in result["preset"]["sweep"]["prompts"].splitlines()))
    if args.dry_run:
        print("\n[sweep-plan] dry run: nothing written")
        return
    backup = write_preset(settings, result)
    print(f"\n[sweep-plan] written to {settings.krea2_sweep_presets}" + (f" (backup {backup})" if backup else ""))
    print(f"[sweep-plan] in ComfyUI: SweepPlan node -> sweep_preset -> \"{result['name']}\" "
          f"(refresh the page so the list picks it up)")
