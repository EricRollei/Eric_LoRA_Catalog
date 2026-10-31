"""
Your corrections to a LoRA's trigger recipe (2026-10-16).

`run.py triggers` has a text model write each LoRA's recipe: trigger, style phrase, official trained words, and from
those the prompt variants (none / trigger / official / trigger+phrase / full). The model gets some wrong - e.g. a style
phrase learned from JSON-style community prompts ("woman1 appearance age, ..."). Here you overrule it, field by field:

    recipe_edits    path PRIMARY KEY, trigger, style_phrase, official, full, edited_at
                    NULL = use the model's value; '' = deliberately empty (e.g. no style phrase)

    trigger        the recipe's trigger token (not the card's trigger list - that is edited in the card's Triggers)
    style_phrase   the look in words; trigger+phrase and full are built from it
    official       the official trained words (comma list)
    full           the full variant's text, used as written (otherwise built from the three above)

The variants stored in trigger_recipes are always the model's recipe with your edits applied, so every reader (Krea2
nodes, sweeps, card_facets, the Suggester) sees your words. Re-running `run.py triggers` rewrites the model's part and
applies your edits again - they are never lost.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

FIELDS = ("trigger", "style_phrase", "official", "full")
TABLE = """CREATE TABLE IF NOT EXISTS recipe_edits (
    path TEXT PRIMARY KEY, trigger TEXT, style_phrase TEXT, official TEXT, full TEXT, edited_at TEXT)"""


def ensure(connection: sqlite3.Connection) -> None:
    connection.execute(TABLE)


def get(connection: sqlite3.Connection, path: str) -> Dict[str, str]:
    """The fields you edited for this LoRA (missing key = not edited)."""
    ensure(connection)
    row = connection.execute(f"SELECT {', '.join(FIELDS)} FROM recipe_edits WHERE path = ?", (path,)).fetchone()
    if not row:
        return {}
    return {f: row[i] for i, f in enumerate(FIELDS) if row[i] is not None}


def set_field(connection: sqlite3.Connection, path: str, field: str, value: Optional[str]) -> None:
    """value None = back to the model's value; '' = deliberately empty."""
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}")
    ensure(connection)
    connection.execute("INSERT OR IGNORE INTO recipe_edits (path) VALUES (?)", (path,))
    connection.execute(f"UPDATE recipe_edits SET {field} = ?, edited_at = ? WHERE path = ?",
                       (None if value is None else re.sub(r"\s+", " ", str(value)).strip(),
                        time.strftime("%Y-%m-%d %H:%M:%S"), path))
    left = connection.execute(f"SELECT {' OR '.join(f + ' IS NOT NULL' for f in FIELDS)} FROM recipe_edits "
                              "WHERE path = ?", (path,)).fetchone()
    if left is not None and not left[0]:
        connection.execute("DELETE FROM recipe_edits WHERE path = ?", (path,))


def _norm(text: str) -> str:
    return re.sub(r"[\s,]+", " ", str(text or "")).strip().lower()


def _override_full(variants: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    out = []
    for v in variants:
        v = dict(v, covers=list(v.get("covers") or []))
        if v.get("name") == "full":
            if not v["covers"]:
                continue
            v["name"] = v["covers"].pop(0)          # the variants merged into it keep their (same) text
        elif "full" in v["covers"]:
            v["covers"].remove("full")
        out.append(v)
    for v in out:                                   # your text equals another variant: one row, two names
        if _norm(v.get("text")) == _norm(text):
            v["covers"].append("full")
            return out
    out.append({"name": "full", "text": text, "covers": []})
    return out


def apply(connection: sqlite3.Connection, path: str, recipe: Dict[str, Any], trained_words: List[str],
          edits: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """The variants for a LoRA: the model's recipe with your edits on top."""
    from .triggers import build_variants
    ed = get(connection, path) if edits is None else edits
    r = dict(recipe or {})
    tw = list(trained_words or [])
    if "trigger" in ed:
        r["trigger"] = ed["trigger"]
    if "style_phrase" in ed:
        r["style_phrase"] = ed["style_phrase"]
        r["extra_phrases"] = []                     # your phrase replaces the model's extras too
    if "official" in ed:
        r["trained_words_fixed"] = [w.strip() for w in ed["official"].split(",") if w.strip()]
        tw = list(r["trained_words_fixed"])
    variants = build_variants(r, tw)
    if "full" in ed:
        variants = _override_full(variants, ed["full"])
    return variants


def _model(connection: sqlite3.Connection, path: str):
    row = connection.execute("SELECT recipe FROM trigger_recipes WHERE path = ?", (path,)).fetchone()
    try:
        recipe = json.loads(row[0]) if row and row[0] else {}
    except (TypeError, ValueError):
        recipe = {}
    lr = connection.execute("SELECT trained_words FROM loras WHERE path = ?", (path,)).fetchone()
    try:
        trained = [str(w) for w in (json.loads(lr[0]) if lr and lr[0] else []) if w]
    except (TypeError, ValueError):
        trained = []
    return recipe, trained, row is not None


def _text(variants: List[Dict[str, Any]], name: str) -> str:
    for v in variants:
        if v.get("name") == name or name in (v.get("covers") or []):
            return str(v.get("text") or "")
    return ""


def view(connection: sqlite3.Connection, path: str) -> Dict[str, Any]:
    """For the card editor: the model's four values, your edits, and whether a recipe exists."""
    recipe, trained, exists = _model(connection, path)
    model_variants = apply(connection, path, recipe, trained, edits={})
    model = {"trigger": str(recipe.get("trigger") or ""), "style_phrase": str(recipe.get("style_phrase") or ""),
             "official": ", ".join(recipe.get("trained_words_fixed") or trained),
             "full": _text(model_variants, "full")}
    return {"model": model, "edits": get(connection, path), "has_recipe": exists}


def refresh(connection: sqlite3.Connection, path: str) -> None:
    """Rewrite this LoRA's stored variants (model recipe + your edits). A LoRA without a recipe gets a row made
    from your edits alone; `run.py triggers` later fills in the model's part and keeps your edits."""
    from .triggers import ensure as ensure_recipes
    ensure_recipes(connection)
    recipe, trained, exists = _model(connection, path)
    variants = apply(connection, path, recipe, trained)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if exists:
        connection.execute("UPDATE trigger_recipes SET variants = ?, updated_at = ? WHERE path = ?",
                           (json.dumps(variants, ensure_ascii=False), now, path))
    else:
        connection.execute("INSERT INTO trigger_recipes (path, model, recipe, variants, evidence, evidence_sig, "
                           "updated_at) VALUES (?, 'your edit', '{}', ?, '{}', '', ?)",
                           (path, json.dumps(variants, ensure_ascii=False), now))
