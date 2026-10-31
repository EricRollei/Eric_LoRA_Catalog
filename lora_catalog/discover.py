"""
Discover, stage 1 (2026-10-29): LoRAs you do NOT own, ranked by how often Civitai users stack them with the LoRAs you
rate highly (docs/design/DISCOVER_STAGE1.md).

Data: the stack harvest (harvest.py: harvest_images, lora_pairs - which LoRAs people use together in the same image,
counted by distinct users, with lift and score), your per-LoRA quality (markmodel.py: lora_observed), and Civitai's
community score for the candidates (community.quick_scores, cached version records).

    rank(X) = sum over your LoRAs P used together with X of  pair.score x taste(P)   x  (0.5 + civitai_score(X) / 100)
    taste(P) = clamp(1 + 0.5 x quality(P), 0.25, 2.0);  1.0 when P is unrated;  0 when quality(P) <= -1.5

The shelf (top folder, e.g. Krea2) chooses the Civitai base models counted - Civitai names them differently ("Krea 2",
"Krea 2 Turbo"), so the mapping is learnt from what the harvest recorded for your LoRAs in that folder, or set in
settings.json "discover_bases": {"Krea2": ["Krea 2", "Krea 2 Turbo"]}. An empty shelf = every base model.

Tables
    discover_dismissed   key (v:<versionId>), name, dismissed_at   - "Not interested": left out of Discover and the
                         run.py pairs --wishlist list until undone

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import Settings

TASTE_SLOPE = 0.5
TASTE_MIN = 0.25
TASTE_MAX = 2.0
TASTE_CUT = -1.5            # a LoRA you rated this far below average: what people stack with it says nothing to you
BASE_SHARE = 0.05           # a Civitai base counts for a shelf when it carries >= 5% of that shelf's harvested pairs

DISMISS_TABLE = """CREATE TABLE IF NOT EXISTS discover_dismissed (
    key TEXT PRIMARY KEY, name TEXT, dismissed_at TEXT)"""


def taste(quality: Optional[float]) -> float:
    """How much one of your LoRAs' company counts, from your rating model's quality (stars vs your average)."""
    if quality is None:
        return 1.0
    if quality <= TASTE_CUT:
        return 0.0
    return max(TASTE_MIN, min(TASTE_MAX, 1.0 + TASTE_SLOPE * float(quality)))


def _top(rel_dir: Any) -> str:
    return str(rel_dir or "").replace("\\", "/").strip("/").split("/")[0]


def _tables(connection: sqlite3.Connection) -> set:
    return {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}


def shelf_bases(connection: sqlite3.Connection, settings: Settings, shelf: str) -> Optional[List[str]]:
    """Civitai base-model names for a shelf. None = every base (no shelf); [] = nothing harvested for that shelf."""
    if not shelf:
        return None
    override = getattr(settings, "discover_bases", None) or {}
    for name, bases in override.items() if isinstance(override, dict) else []:
        if str(name).lower() == shelf.lower() and bases:
            return [str(b) for b in bases]
    mine = set()
    for vid, rel_dir in connection.execute("select civitai_version_id, rel_dir from loras "
                                           "where civitai_version_id is not null"):
        if _top(rel_dir).lower() == shelf.lower():
            mine.add(f"v:{int(vid)}")
    counts: Counter = Counter()
    for a, b, base in connection.execute("select a, b, base_model from lora_pairs"):
        if a in mine or b in mine:
            counts[base or ""] += 1
    total = sum(counts.values())
    return sorted(b for b, n in counts.items() if b and n >= BASE_SHARE * total) if total else []


def _qualities(connection: sqlite3.Connection, tables: set) -> Dict[str, float]:
    if "lora_observed" not in tables:
        return {}
    return {str(k).lower(): float(q) for k, q in connection.execute("select key, quality from lora_observed")
            if q is not None}


def _ratings(connection: sqlite3.Connection, tables: set) -> Dict[str, int]:
    if "card_edits" not in tables:
        return {}
    cols = {r[1] for r in connection.execute("pragma table_info(card_edits)")}
    if "rating" not in cols:
        return {}
    return {str(k).lower(): int(r) for k, r in connection.execute("select key, rating from card_edits "
                                                                   "where rating is not null")}


def dismissed(connection: sqlite3.Connection) -> Dict[str, str]:
    if "discover_dismissed" not in _tables(connection):
        return {}
    return {k: n or "" for k, n in connection.execute("select key, name from discover_dismissed")}


def rank_candidates(connection: sqlite3.Connection, settings: Settings, bases: Optional[List[str]],
                    use_taste: bool = True, skip: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """{candidate key: {total, partners: [...], weights: [...], examples: [...], base_models: set}} - no Civitai calls.
    One row per (candidate, your LoRA): the best-scoring base model among those allowed."""
    from .harvest import LocalIndex
    tables = _tables(connection)
    local = LocalIndex(connection)
    quality = _qualities(connection, tables)
    rating = _ratings(connection, tables)
    skip = skip or set()
    best: Dict[tuple, sqlite3.Row] = {}
    for r in connection.execute("select * from lora_pairs where owned_a + owned_b = 1 and score > 0"):
        if bases is not None and (r["base_model"] or "") not in bases:
            continue
        theirs, ours = (r["b"], r["a"]) if r["owned_a"] else (r["a"], r["b"])
        if theirs in skip:
            continue
        key = (theirs, ours)
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r
    out: Dict[str, Dict[str, Any]] = {}
    for (theirs, ours), r in best.items():
        row = local.owned(ours)
        sha = str(row["sha256"] or "").lower() if row is not None else ""
        q = quality.get(sha) if sha else None
        t = taste(q) if use_taste else 1.0
        their_w = json.loads((r["weights_b"] if r["a"] == ours else r["weights_a"]) or "[]")
        c = out.setdefault(theirs, {"total": 0.0, "partners": [], "weights": [], "examples": [], "base_models": set()})
        c["total"] += r["score"] * t
        c["weights"] += [w for w in their_w if isinstance(w, (int, float))]
        c["base_models"].add(r["base_model"] or "")
        c["partners"].append({
            "key": ours, "path": row["path"] if row is not None else "",
            "name": (row["model_name"] or row["stem"]) if row is not None else ours,
            "users": r["n"], "lift": r["lift"], "score": round(r["score"], 3), "quality": None if q is None else round(q, 2),
            "taste": round(t, 2), "rating": rating.get(sha), "contribution": round(r["score"] * t, 3),
            "examples": json.loads(r["examples"] or "[]")[:3]})
    for c in out.values():
        c["partners"].sort(key=lambda p: -p["contribution"])
        c["examples"] = [e for p in c["partners"][:3] for e in p["examples"]][:4]
    return out


def candidates(settings: Settings, shelf: str = "", all_bases: bool = False, top: int = 100,
               use_taste: bool = True, civitai: bool = True) -> Dict[str, Any]:
    """The Discover list. Civitai calls only for the top 3 x top by pair sum, and those records are cached."""
    from .harvest import LocalIndex, TRAINING, _names
    connection = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        tables = _tables(connection)
        if "lora_pairs" not in tables:
            return {"items": [], "bases": [], "note": "No stack harvest yet: python run.py harvest --only <folder>, "
                                                     "then python run.py pairs"}
        bases = None if all_bases else shelf_bases(connection, settings, shelf)
        if bases == []:
            return {"items": [], "bases": [], "note": f"Nothing harvested for LoRAs on {shelf} yet: python run.py "
                                                     f"harvest --only {shelf}, then python run.py pairs"}
        gone = dismissed(connection)
        ranked = rank_candidates(connection, settings, bases, use_taste, set(gone))
        local = LocalIndex(connection)
        names = _names(connection, settings, list(ranked), local)
        ranked = {k: v for k, v in ranked.items() if not TRAINING.match(names.get(k, ""))}
        pool = sorted(ranked, key=lambda k: -ranked[k]["total"])[:top * 3]
        quick: Dict[int, Dict[str, Any]] = {}
        if civitai and pool:
            from .community import quick_scores
            quick = quick_scores(settings, [int(k[2:]) for k in pool if k.startswith("v:")])

        def qinfo(key: str) -> Dict[str, Any]:
            return quick.get(int(key[2:]), {}) if key.startswith("v:") else {}

        def rank(key: str) -> float:
            s = qinfo(key).get("score")
            return ranked[key]["total"] * (0.5 + (s if s is not None else 50.0) / 100.0)

        chosen = sorted(pool, key=lambda k: -rank(k))[:top]
        records: Dict[str, Dict[str, Any]] = {}
        if civitai:
            from .crawl import CivitaiClient
            client = CivitaiClient(settings)
            for k in chosen:
                if k.startswith("v:"):
                    try:
                        records[k] = client.get("v", int(k[2:])) or {}
                    except Exception:
                        records[k] = {}
        host = str(getattr(settings, "civitai_host", "") or "https://civitai.com").rstrip("/")
        # another version of the same Civitai model already in your library (e.g. v1 owned, v2 suggested)
        owned_models: Dict[int, str] = {}
        if "civitai_model_id" in {r[1] for r in connection.execute("pragma table_info(loras)")}:
            for mid, path in connection.execute("select civitai_model_id, path from loras "
                                                "where civitai_model_id is not null"):
                owned_models.setdefault(int(mid), path)
        seeds = connection.execute("select count(*), max(fetched_at) from harvest_seeds").fetchone() \
            if "harvest_seeds" in tables else (0, None)
        items = []
        for k in chosen:
            c, qi, rec = ranked[k], qinfo(k), records.get(k, {})
            model = rec.get("model") or {}
            vid = int(k[2:]) if k.startswith("v:") else None
            model_id = rec.get("modelId")
            url = (f"{host}/models/{model_id}?modelVersionId={vid}" if model_id and vid
                   else f"{host}/model-versions/{vid}" if vid else "")
            ws = c["weights"]
            items.append({
                "key": k, "version_id": vid, "name": qi.get("model_name") or model.get("name") or names.get(k, k),
                "version": qi.get("name") or rec.get("name") or "", "base_model": qi.get("base_model") or
                rec.get("baseModel") or ", ".join(sorted(b for b in c["base_models"] if b)),
                "rank": round(rank(k), 3), "pair_sum": round(c["total"], 3), "url": url,
                "nsfw": bool(model.get("nsfw")) if model else None, "nsfw_level": rec.get("nsfwLevel"),
                "civitai": {"score": qi.get("score"), "downloads": qi.get("downloads"), "up": qi.get("up"),
                            "down": qi.get("down")},
                "weights": {"median": round(statistics.median(ws), 2), "low": round(min(ws), 2),
                            "high": round(max(ws), 2), "n": len(ws)} if ws else None,
                "examples": [f"{host}/images/{e}" for e in c["examples"]],
                "partners": c["partners"][:8], "n_partners": len(c["partners"]),
                "other_version": owned_models.get(int(model_id)) if model_id else None})
        # LoRAs you already have in another version go to the end of the list (still shown, labelled)
        items.sort(key=lambda it: (it["other_version"] is not None, -it["rank"]))
        return {"items": items, "bases": bases, "candidates": len(ranked), "dismissed": len(gone),
                "harvested": {"loras": seeds[0], "last": seeds[1]}, "taste": use_taste}
    finally:
        connection.close()


def dismiss(settings: Settings, key: str, name: str = "", undo: bool = False) -> Dict[str, Any]:
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    try:
        connection.execute(DISMISS_TABLE)
        if undo:
            connection.execute("delete from discover_dismissed where key = ?", (key,))
        else:
            connection.execute("insert or replace into discover_dismissed values (?, ?, ?)",
                               (key, name, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        connection.commit()
        n = connection.execute("select count(*) from discover_dismissed").fetchone()[0]
    finally:
        connection.close()
    return {"ok": True, "key": key, "undone": undo, "dismissed": n}
