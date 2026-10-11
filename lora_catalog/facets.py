"""
The finished, readable view of every card for other tools: one row per LoRA in `card_facets`, with the pick-list
terms already worked out. Your edit (card_edits) wins wholesale; otherwise the terms are read from the generated
card (vocab.prefill). Readers such as the LoRA Suggester only need this table and vocab.yaml.

    python run.py facets              rebuild the whole table (seconds)
    refresh_one(connection, path)     after an edit in the card editor

    card_facets
        path, key, name, file_name, rel_dir, arch_folder, base_model
        role                 style | treatment | detailer | environment | lighting | subject | layout | utility
        facets               {facet: [term ids or free words]}  (role, use, medium, period ... side_effects)
        nsfw                 0..3 for filtering: your rating when reviewed, else the card's own judgement of what
                             the LoRA is FOR (not raised by Civitai's example images, which over-count)
        nsfw_shown           0..3 the editor's pre-tick (card raised to Civitai's example-image rating)
        artists, depicts     JSON lists of names
        triggers             JSON list of the triggers prompts should get: your priority ones first, then normal
                             ones; ones you switched off are left out (they are in triggers_off)
        triggers_off         JSON list of triggers you switched off (still the LoRA's words, never added to prompts)
        rating               your 1-5 stars, or null (unrated)
        slider               JSON or null: {axis, positive_toward, positive_pole, negative_pole, range, useful, stages,
                             confidence, source} from slider_cards; useful = your editor weight low/high when you
                             set them, else the slider card's (2026-10-08). A slider's role is `slider` unless
                             your edit gives it a role other than utility.
        creator, weight_typical, weight_low, weight_high, overrides (JSON), confidence, kind, scope, effect
        favorite, exclude    LoRA Manager flags
        source               edit | card;  status = your edit's status (confirmed | edited | unsure) or ''
        updated_at

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from typing import Any, Dict, List, Optional

from . import creators, edits, sliders
from .config import Settings
from .vocab import NSFW_TERMS, _card_nsfw, load_vocab, prefill

SCHEMA = """
CREATE TABLE IF NOT EXISTS card_facets (
    path TEXT PRIMARY KEY, key TEXT, name TEXT, file_name TEXT, rel_dir TEXT, arch_folder TEXT, base_model TEXT,
    role TEXT, facets TEXT, nsfw INTEGER, nsfw_shown INTEGER, artists TEXT, depicts TEXT, triggers TEXT,
    creator TEXT, weight_typical REAL, weight_low REAL, weight_high REAL, overrides TEXT, confidence REAL,
    kind TEXT, scope TEXT, effect TEXT, favorite INTEGER, exclude INTEGER, source TEXT, status TEXT, updated_at TEXT,
    rating INTEGER, triggers_off TEXT, slider TEXT
);
CREATE INDEX IF NOT EXISTS card_facets_arch ON card_facets(arch_folder);
"""

NEWER_FACETS = ("genre", "mood")         # 2026-10-10; same rule in editor.html

COLUMNS = ["path", "key", "name", "file_name", "rel_dir", "arch_folder", "base_model", "role", "facets", "nsfw",
           "nsfw_shown", "artists", "depicts", "triggers", "creator", "weight_typical", "weight_low", "weight_high",
           "overrides", "confidence", "kind", "scope", "effect", "favorite", "exclude", "source", "status",
           "updated_at", "rating", "triggers_off", "slider"]
ADDED_COLUMNS = {"rating": "INTEGER", "triggers_off": "TEXT",       # 2026-10-07
                 "slider": "TEXT"}                                   # 2026-10-08

ROW_SQL = """select l.*, c.kind, c.scope, c.effect, c.triggers as card_triggers, c.weight_typical, c.weight_low,
                    c.weight_high, c.overrides, c.nsfw_level as card_nsfw, c.confidence, c.artists as card_artists,
                    c.medium, c.movement_era, c.palette, c.line_quality, c.suited_subjects, c.genre, c.mood
             from loras l join cards c on c.path = l.path"""


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    have = {row[1] for row in connection.execute("pragma table_info(card_facets)")}
    for name, kind in ADDED_COLUMNS.items():
        if name not in have:
            connection.execute(f"alter table card_facets add column {name} {kind}")
    edits.ensure(connection)
    creators.ensure(connection)
    sliders.ensure(connection)


def _json_list(value: Any) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _names(items: list, key: str) -> List[str]:
    return [str(item.get(key, "")) if isinstance(item, dict) else str(item) for item in items
            if (item.get(key) if isinstance(item, dict) else item)]


def facet_row(connection: sqlite3.Connection, row: sqlite3.Row, vocab: Any) -> Dict[str, Any]:
    """The card_facets row for one LoRA (row = ROW_SQL row)."""
    model_terms = prefill(vocab, row, row)
    edit = edits.get_edit(connection, row["sha256"], row["path"])
    if edit:
        terms = {facet: list(values) for facet, values in (edit.get("facets") or {}).items()}
        for facet in NEWER_FACETS:           # an edit saved before these facets existed: the model's picks fill them
            if facet not in terms:
                terms[facet] = list(model_terms.get(facet) or [])
        artists = list(edit.get("artists") or [])
        states = edit.get("trigger_states") or {}
        every = list(edit.get("triggers") or [])
        triggers = edits.active_triggers(every, states)
        triggers_off = [t for t in every if states.get(t) == "off"]
        weight = edit.get("weight") if edit.get("weight") is not None else row["weight_typical"]
        weight_low = edit.get("weight_low") if edit.get("weight_low") is not None else row["weight_low"]
        weight_high = edit.get("weight_high") if edit.get("weight_high") is not None else row["weight_high"]
        stars = edit.get("rating")
        creator = edit.get("creator") or ""
        rating = terms.get("nsfw") or []
        nsfw = NSFW_TERMS.index(rating[0]) if rating and rating[0] in NSFW_TERMS else _card_nsfw(row)
        source, status = "edit", edit.get("status") or ""
    else:
        terms = model_terms
        artists = _names(_json_list(row["card_artists"]), "name")
        triggers = _names(_json_list(row["card_triggers"]), "text")
        triggers_off = []
        weight = row["weight_typical"]
        weight_low, weight_high = row["weight_low"], row["weight_high"]
        stars = None
        creator = ""
        nsfw = _card_nsfw(row)
        source, status = "card", ""
    if not creator:
        creator = creators.lookup(connection, row["civitai_model_id"]) or (
            "self-trained" if (row["source_kind"] == "self_trained" or row["training_job"]) else "")
    shown = model_terms.get("nsfw") or ["sfw"]
    role = (terms.get("role") or ["style"])[0]
    slider = _slider_view(connection, row["path"], edit, weight_low, weight_high)
    if slider and (not edit or role in ("utility", "slider")):
        role = "slider"
        terms = dict(terms, role=["slider"] + [r for r in (terms.get("role") or []) if r not in ("slider", "utility")])
    return {
        "path": row["path"], "key": edits.edit_key(row["sha256"], row["path"]),
        "name": row["model_name"] or row["stem"], "file_name": row["file_name"], "rel_dir": row["rel_dir"],
        "arch_folder": row["arch_folder"], "base_model": row["base_model"],
        "role": role, "facets": json.dumps(terms, ensure_ascii=False),
        "nsfw": nsfw, "nsfw_shown": NSFW_TERMS.index(shown[0]) if shown[0] in NSFW_TERMS else 0,
        "artists": json.dumps(artists, ensure_ascii=False),
        "depicts": json.dumps(list(terms.get("depicts") or []), ensure_ascii=False),
        "triggers": json.dumps(triggers, ensure_ascii=False), "creator": creator,
        "weight_typical": weight, "weight_low": weight_low, "weight_high": weight_high,
        "overrides": row["overrides"] or "[]", "confidence": row["confidence"], "kind": row["kind"],
        "scope": row["scope"], "effect": row["effect"] or "", "favorite": int(bool(row["lm_favorite"])),
        "exclude": int(bool(row["lm_exclude"])), "source": source, "status": status,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rating": stars, "triggers_off": json.dumps(triggers_off, ensure_ascii=False),
        "slider": json.dumps(slider, ensure_ascii=False) if slider else None,
    }


def _slider_view(connection: sqlite3.Connection, path: str, edit: Optional[dict], weight_low: Any,
                 weight_high: Any) -> Optional[Dict[str, Any]]:
    """The slider part of a card_facets row. Your editor weights (low / high) win over the slider card's useful range."""
    card = sliders.slider_for(connection, path)
    if not card:
        return None
    useful = [card["useful_low"], card["useful_high"]]
    source = "slider_card"
    if edit and (edit.get("weight_low") is not None or edit.get("weight_high") is not None):
        useful = [weight_low if weight_low is not None else useful[0], weight_high if weight_high is not None else useful[1]]
        source = "your_edit"
    return {"axis": card["axis"], "positive_toward": card["positive_toward"], "positive_pole": card["positive_pole"],
            "negative_pole": card["negative_pole"], "range": [card["range_low"], card["range_high"]],
            "useful": useful, "stages": card["stages"], "confidence": card["confidence"], "source": source}


def _write(connection: sqlite3.Connection, record: Dict[str, Any]) -> None:
    connection.execute(f"insert or replace into card_facets ({','.join(COLUMNS)}) values ({','.join('?' for _ in COLUMNS)})",
                       [record[column] for column in COLUMNS])


def build_all(settings: Settings) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    vocab = load_vocab()
    started = datetime.datetime.now().isoformat(timespec="seconds")
    rows = list(connection.execute(ROW_SQL))
    counts = {"cards": len(rows), "from_edits": 0}
    records = [facet_row(connection, row, vocab) for row in rows]     # the slow part, before taking the write lock,
    counts["from_edits"] = sum(r["source"] == "edit" for r in records)   # so editor saves are not held up for minutes
    paths = {r["path"] for r in records}
    fresh = {r[0]: r[1] for r in connection.execute("select path, updated_at from card_facets")}
    connection.execute("delete from card_facets")
    for record in records:
        _write(connection, record)
    # an edit saved in the card editor while the rows above were being worked out: redo just those
    for path in [p for p in paths if p in fresh and fresh[p] and fresh[p] > started]:
        refreshed = connection.execute(ROW_SQL + " where l.path = ?", (path,)).fetchone()
        if refreshed is not None:
            _write(connection, facet_row(connection, refreshed, vocab))
    connection.execute("insert or replace into meta (k, v) values ('card_facets_built', ?)",
                       (datetime.datetime.now().isoformat(timespec="seconds"),))
    connection.commit()
    connection.close()
    return counts


def refresh_one(connection: sqlite3.Connection, path: str) -> Optional[str]:
    """Rewrite one LoRA's row (after an edit). Returns its role, or None when the LoRA has no card."""
    ensure(connection)
    previous = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(ROW_SQL + " where l.path = ?", (path,)).fetchone()
        if row is None:
            return None
        record = facet_row(connection, row, load_vocab())
        _write(connection, record)
        connection.commit()
        return record["role"]
    finally:
        connection.row_factory = previous
