"""
Word options: several named sets of prompt words for one LoRA (2026-10-20).

Some LoRAs are trained on alternatives rather than one trigger - a colour-grading LoRA with one keyword per grade,
a style pack with three style names, an age or cohort pack. A trigger recipe gives one set of words; word options
give the choices, by name, so the Krea2 Multi-LoRA Stack (its word_options field / Pick button) and the sweep node
("name #option", "name #each") can pick one per slot or per cell.

    word_options   path, name, text, ord, is_default      PRIMARY KEY (path, name)

Where the options come from:
    yours      rows in word_options (set in the card editor's Prompt words > Word options box); they replace the
               automatic ones completely. A single row named '__none__' means "you cleared them: no options".
    civitai    worked out from the LoRA's Civitai trained words whenever they look like alternatives (auto_options):
               3+ entries of 6 words or fewer, all different, and either one word shared by at least 60% of them
               (and by 3 or more)
               ("cohort_fi", "cohort_se" ...; "EpicChronicles style illustration", "DriftForm style illustration")
               or all of them single made-up tokens ("18yo", "19yo" ...). When exactly one entry lacks the shared
               word and is a single token (the LoRA's own trigger, e.g. "Scandinavian"), it is put in front of every
               option instead of being one. Nothing is written for these, so they follow the Civitai data.

A default option (yours only) becomes the LoRA's words for every checkpoint when you have no pick of your own for
all checkpoints (trigger_picks.resolve); per-checkpoint picks still win. Automatic options never have a default.

card_facets.word_options holds the result as JSON [{name, text, default}] for Eric_Krea2 to read.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

TABLE = """CREATE TABLE IF NOT EXISTS word_options (
    path TEXT, name TEXT, text TEXT, ord INTEGER, is_default INTEGER DEFAULT 0,
    PRIMARY KEY (path, name))"""
NONE = "__none__"
MAX_WORDS = 6
SHARE = 0.6
STOP = {"a", "an", "the", "of", "with", "and", "in", "on", "by", "to", "for", "at"}


def ensure(connection: sqlite3.Connection) -> None:
    connection.execute(TABLE)


def _list(value: Any) -> List[str]:
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(value or "[]")
        except (TypeError, ValueError):
            items = [value] if value else []
        if not isinstance(items, list):
            items = [items]
    out: List[str] = []
    for item in items:
        text = str(item.get("text", "") if isinstance(item, dict) else item or "").strip().strip(",").strip()
        if text:
            out.append(text)
    return out


def clean_name(name: str) -> str:
    """Option names go into sweep lines ("lora #name @0.8 | other"), so no '#', '@' or '|' in them."""
    return re.sub(r"\s+", " ", re.sub(r"[#@|]", "", str(name or ""))).strip()[:60]


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[\s_\-,.:;/()]+", text.lower()) if t and t not in STOP]


def _made_up(text: str) -> bool:
    """One token that is not a plain word: has a digit or underscore, or mixed case inside it."""
    if len(text.split()) != 1:
        return False
    return bool(re.search(r"[\d_@]", text) or re.search(r"[a-z][A-Z]", text))


def auto_options(trained_words: Any) -> List[Dict[str, Any]]:
    """The options the Civitai trained words imply, or [] when they read as one set of words to use together."""
    words = _list(trained_words)
    seen, items = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            items.append(w)
    if len(items) < 3 or len(items) != len(words) or any(len(w.split()) > MAX_WORDS for w in items):
        return []
    token_sets = [set(_tokens(w)) for w in items]
    counts: Dict[str, int] = {}
    for ts in token_sets:
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
    shared = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
    base = ""
    options = items
    if shared[1] >= max(3, SHARE * len(items)):
        outside = [w for w, ts in zip(items, token_sets) if shared[0] not in ts]
        if len(outside) == 1 and len(outside[0].split()) == 1:
            base = outside[0]
            options = [w for w in items if w != base]
    elif not all(_made_up(w) for w in items):
        return []
    if len(options) < 2:
        return []
    return [{"name": clean_name(w) or str(i + 1), "text": f"{base}, {w}" if base else w, "default": False, "source": "civitai"}
            for i, w in enumerate(options)]


def mine(connection: sqlite3.Connection, path: str) -> Optional[List[Dict[str, Any]]]:
    """Your options for this LoRA ([] when you cleared them), or None when you have not set any."""
    ensure(connection)
    rows = connection.execute("select name, text, is_default from word_options where path = ? order by ord, name",
                              (path,)).fetchall()
    if not rows:
        return None
    return [{"name": r[0], "text": r[1] or "", "default": bool(r[2]), "source": "you"} for r in rows if r[0] != NONE]


def options_for(connection: sqlite3.Connection, path: str, trained_words: Any = None) -> List[Dict[str, Any]]:
    """The options in force: yours, else the automatic ones from the trained words."""
    own = mine(connection, path)
    if own is not None:
        return own
    if trained_words is None:
        row = connection.execute("select trained_words from loras where path = ?", (path,)).fetchone()
        trained_words = row[0] if row else None
    return auto_options(trained_words)


def default_text(options: List[Dict[str, Any]]) -> Optional[str]:
    hit = next((o for o in options if o.get("default")), None)
    return None if hit is None else hit.get("text") or ""


def set_options(connection: sqlite3.Connection, path: str, options: List[Dict[str, Any]]) -> None:
    """Replace your options for this LoRA. An empty list stores '__none__' (no options, not even automatic ones)."""
    ensure(connection)
    connection.execute("delete from word_options where path = ?", (path,))
    clean, names = [], set()
    for o in options or []:
        name = clean_name(o.get("name"))
        text = str(o.get("text") or "").strip()
        if not name and text:
            name = clean_name(text)
        if not name or name == NONE or name.lower() in names:
            continue
        names.add(name.lower())
        clean.append((name, text, bool(o.get("default"))))
    if not clean:
        connection.execute("insert into word_options values (?,?,?,?,?)", (path, NONE, "", 0, 0))
        return
    first_default = next((i for i, c in enumerate(clean) if c[2]), None)
    for i, (name, text, _) in enumerate(clean):
        connection.execute("insert into word_options values (?,?,?,?,?)",
                           (path, name, text, i, int(i == first_default)))


def reset(connection: sqlite3.Connection, path: str) -> None:
    """Forget your options: the automatic ones (if any) come back."""
    ensure(connection)
    connection.execute("delete from word_options where path = ?", (path,))


def view(connection: sqlite3.Connection, path: str) -> Dict[str, Any]:
    """The editor's Word options box: what is in force and where it came from."""
    own = mine(connection, path)
    row = connection.execute("select trained_words from loras where path = ?", (path,)).fetchone()
    auto = auto_options(row[0] if row else None)
    if own is not None:
        return {"options": own, "source": "you" if own else "cleared", "auto": auto}
    return {"options": auto, "source": "civitai" if auto else "", "auto": auto}


def find(options: List[Dict[str, Any]], wanted: str) -> Optional[Dict[str, Any]]:
    """An option by name (case-insensitive), else by 1-based number, else by a unique name prefix."""
    w = (wanted or "").strip().lower()
    if not w:
        return None
    for o in options:
        if o["name"].lower() == w:
            return o
    if w.isdigit() and 1 <= int(w) <= len(options):
        return options[int(w) - 1]
    hits = [o for o in options if o["name"].lower().startswith(w)]
    return hits[0] if len(hits) == 1 else None
