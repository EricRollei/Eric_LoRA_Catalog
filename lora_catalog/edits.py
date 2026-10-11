"""
Your corrections to cards, kept apart from the generated cards so a card re-run never overwrites them.

    card_edits   one row per LoRA you have looked at
        key        sha256 of the file when known (survives moving or renaming the file), else the path
        path       where the file was when you edited it
        facets     {facet: [term ids or free words]} - the full chosen set, not a diff
        artists    [names]
        creator    who made the LoRA (Civitai username, or "self-trained")
        triggers   [trigger texts]   (all of them, in your order)
        trigger_states  {trigger: priority | off}   missing = normal; off = kept on the card, not added to prompts
        weight     typical weight you trust, or null
        weight_low, weight_high   the range you trust (-10..+10; sliders go negative), or null
        rating     your 1-5 stars, or null (unrated)
        notes      free text
        status     confirmed (model was right) | edited (you changed something) | unsure (come back later)
        prefill    the model's facets at the time, so 'what the model gets wrong' can be studied later
        edited_at  ISO time

Readers (the suggester, the next card pass) should take an edit over the card wherever one exists.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS card_edits (
    key TEXT PRIMARY KEY, path TEXT, facets TEXT, artists TEXT, triggers TEXT, weight REAL, notes TEXT,
    status TEXT, prefill TEXT, edited_at TEXT, creator TEXT, rating INTEGER, weight_low REAL, weight_high REAL,
    trigger_states TEXT
);
CREATE INDEX IF NOT EXISTS card_edits_path ON card_edits(path);
"""

STATUSES = ("confirmed", "edited", "unsure")
TRIGGER_STATES = ("priority", "off")          # anything else = normal
# Columns added after the table first shipped: creator 2026-10-02, the rest 2026-10-07.
ADDED_COLUMNS = {"creator": "TEXT", "rating": "INTEGER", "weight_low": "REAL", "weight_high": "REAL",
                 "trigger_states": "TEXT"}


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    have = {row[1] for row in connection.execute("pragma table_info(card_edits)")}
    missing = [(name, kind) for name, kind in ADDED_COLUMNS.items() if name not in have]
    for name, kind in missing:
        connection.execute(f"alter table card_edits add column {name} {kind}")
    if missing:
        connection.commit()


def edit_key(sha256: Optional[str], path: str) -> str:
    return sha256.lower() if sha256 else path


def get_edit(connection: sqlite3.Connection, sha256: Optional[str], path: str) -> Optional[Dict[str, Any]]:
    row = connection.execute("select * from card_edits where key = ?", (edit_key(sha256, path),)).fetchone()
    if row is None:
        row = connection.execute("select * from card_edits where path = ?", (path,)).fetchone()
    if row is None:
        return None
    if not isinstance(row, sqlite3.Row):
        names = [d[0] for d in connection.execute("select * from card_edits limit 0").description]
        out = dict(zip(names, row))
    else:
        out = dict(row)
    for column in ("facets", "artists", "triggers", "prefill", "trigger_states"):
        try:
            out[column] = json.loads(out.get(column)) if out.get(column) else ([] if column in ("artists", "triggers") else {})
        except ValueError:
            out[column] = [] if column in ("artists", "triggers") else {}
    return out


def clean_states(triggers: List[str], states: Any) -> Dict[str, str]:
    """Only priority/off entries, only for triggers that are on the card."""
    if not isinstance(states, dict):
        return {}
    return {t: states[t] for t in triggers if states.get(t) in TRIGGER_STATES}


def active_triggers(triggers: List[str], states: Optional[Dict[str, str]]) -> List[str]:
    """What prompts should get: priority ones first, then normal ones, never off ones (order kept within each)."""
    states = states or {}
    return ([t for t in triggers if states.get(t) == "priority"] +
            [t for t in triggers if states.get(t) not in TRIGGER_STATES])


def _same(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> bool:
    keys = set(a) | set(b)
    return all(sorted(a.get(k) or []) == sorted(b.get(k) or []) for k in keys)


def save_edit(connection: sqlite3.Connection, *, sha256: Optional[str], path: str, facets: Dict[str, List[str]],
              artists: List[str], triggers: List[str], weight: Optional[float], notes: str, unsure: bool,
              prefill: Dict[str, List[str]], other_changed: bool, creator: str = "", rating: Optional[int] = None,
              weight_low: Optional[float] = None, weight_high: Optional[float] = None,
              trigger_states: Optional[Dict[str, str]] = None) -> str:
    """Store the full chosen card. Status is worked out here: unsure if asked, confirmed if nothing differs from
    what the model said, edited otherwise. A rating alone does not make a card 'edited'."""
    if unsure:
        status = "unsure"
    elif other_changed or not _same(facets, prefill):
        status = "edited"
    else:
        status = "confirmed"
    clean = {k: [str(t).strip() for t in v if str(t).strip()] for k, v in facets.items()}
    try:
        rating = int(rating) if rating not in (None, "", 0) else None
    except (TypeError, ValueError):
        rating = None
    if rating is not None and not 1 <= rating <= 5:
        rating = None
    states = clean_states(triggers, trigger_states)
    connection.execute(
        "insert or replace into card_edits (key, path, facets, artists, triggers, weight, notes, status, prefill, edited_at,"
        " creator, rating, weight_low, weight_high, trigger_states) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (edit_key(sha256, path), path, json.dumps(clean, ensure_ascii=False), json.dumps(artists, ensure_ascii=False),
         json.dumps(triggers, ensure_ascii=False), weight, notes or "", status, json.dumps(prefill, ensure_ascii=False),
         datetime.datetime.now().isoformat(timespec="seconds"), (creator or "").strip(), rating, weight_low, weight_high,
         json.dumps(states, ensure_ascii=False) if states else None))
    connection.commit()
    return status


def vocab_gaps(connection: sqlite3.Connection, vocab: Any) -> Dict[str, Dict[str, int]]:
    """Words you typed that are not in the pick lists, per facet, with how often - candidates for vocab.yaml."""
    gaps: Dict[str, Dict[str, int]] = {}
    for (facets_json,) in connection.execute("select facets from card_edits"):
        try:
            facets = json.loads(facets_json or "{}")
        except ValueError:
            continue
        for facet, terms in facets.items():
            for term in terms:
                if not vocab.known(facet, term):
                    gaps.setdefault(facet, {})
                    gaps[facet][term] = gaps[facet].get(term, 0) + 1
    return gaps
