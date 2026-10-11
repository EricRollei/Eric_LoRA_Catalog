"""
Search for the card editor: find LoRAs by name, trigger, artist, tag or any words the card uses ("fashion photo").

    index = SearchIndex(settings)
    hits = index.search("fashion photo")       -> {path: score}  (higher = better), or {} when nothing matches
    index.last                                 -> {"query", "mode" ("all" | "any" | "none"), "fts"} for the page

One in-memory sqlite FTS5 table, one document per LoRA, built from the catalog the first time it is needed and
rebuilt whenever the catalog changes (new LoRAs, a card run, an edit). Nothing is written to the catalog file.

Query syntax
    fashion photo        every word must match; each word is a prefix (photo -> photographic, photography)
    "film noir"          an exact phrase
    -anime               leave out LoRAs that mention anime
    name:arcane          only in one field: name, trigger, artist, creator, notes, tag, card, folder, description
If every word together finds nothing, any one word is enough (mode "any"), and the page says so.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings
from .local_sources import strip_html
from .vocab import load_vocab

# column, bm25 weight. Order matters: it is the FTS5 column order.
COLUMNS: List[Tuple[str, float]] = [
    ("name", 10.0), ("trigger", 6.0), ("artist", 5.0), ("creator", 5.0), ("notes", 5.0),
    ("tag", 4.0), ("card", 3.0), ("folder", 2.0), ("description", 1.0),
]
FIELD_ALIASES = {"name": "name", "file": "name", "trigger": "trigger", "triggers": "trigger", "word": "trigger",
                 "artist": "artist", "artists": "artist", "creator": "creator", "by": "creator", "notes": "notes",
                 "note": "notes", "tag": "tag", "tags": "tag", "facet": "tag", "card": "card", "effect": "card",
                 "folder": "folder", "shelf": "folder", "description": "description", "desc": "description"}
TOKEN = re.compile(r'(-?)(?:(\w+):)?(?:"([^"]+)"|(\S+))')
WORD = re.compile(r"[^\W_]+", re.UNICODE)


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


def _texts(items: list) -> List[str]:
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(str(item.get("name") or item.get("text") or ""))
        elif item is not None:
            out.append(str(item))
    return [text for text in out if text]


def _clean(text: Any) -> str:
    """Underscores, dashes and dots split words, so file names like YFG_Acrylic__Flux_ are searchable."""
    return re.sub(r"[_\-./\\]+", " ", str(text or ""))


def _words(text: str) -> List[str]:
    return WORD.findall(text.lower())


class SearchIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.memory = sqlite3.connect(":memory:", check_same_thread=False)
        self.stamp: Optional[tuple] = None
        self.count = 0
        self.last: Dict[str, Any] = {}

    # -- building ------------------------------------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.catalog_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _stamp(connection: sqlite3.Connection) -> tuple:
        """Changes whenever anything the documents are built from changes."""
        parts = []
        for sql in ("select count(*), max(indexed_at) from loras", "select count(*), max(refined_at) from cards",
                    "select count(*), max(edited_at) from card_edits", "select count(*), max(updated_at) from card_facets"):
            try:
                parts.append(tuple(connection.execute(sql).fetchone()))
            except sqlite3.Error:
                parts.append(None)
        return tuple(parts)

    def _labels(self, facets: dict) -> List[str]:
        """Pick-list ids -> their labels and aliases, so 'magazine_cover' is found by 'magazine cover'."""
        vocab = load_vocab()
        words = []
        for facet_name, ids in facets.items():
            facet = vocab.facets.get(facet_name)
            for term_id in ids or []:
                term = facet.terms.get(term_id) if facet else None
                if term is None:
                    words.append(str(term_id))
                else:
                    words.append(term.label)
                    words.extend(term.aliases)
        return words

    def _document(self, row: sqlite3.Row) -> List[str]:
        facets = _json_dict(row["edit_facets"]) or _json_dict(row["facet_facets"])
        artists = _texts(_json_list(row["edit_artists"])) or _texts(_json_list(row["facet_artists"])) \
            or _texts(_json_list(row["card_artists"]))
        triggers = _texts(_json_list(row["edit_triggers"])) + _texts(_json_list(row["card_triggers"])) \
            + _texts(_json_list(row["trained_words"]))
        card = [row["kind"], row["effect"], row["medium"], row["movement_era"], row["palette"], row["line_quality"],
                ", ".join(_texts(_json_list(row["suited_subjects"])))]
        values = {
            "name": " ".join(_clean(v) for v in (row["model_name"], row["version_name"], row["stem"]) if v),
            "trigger": " | ".join(triggers),
            "artist": " | ".join(artists),
            "creator": _clean(row["edit_creator"] or row["facet_creator"] or ""),
            "notes": " ".join(v for v in (row["edit_notes"], row["lm_notes"]) if v),
            "tag": " | ".join(_texts(_json_list(row["tags"])) + self._labels(facets)),
            "card": " ".join(str(v) for v in card if v),
            "folder": _clean(row["rel_dir"]),
            "description": strip_html(row["description"] or "", 3000),
        }
        return [values[name] for name, _weight in COLUMNS]

    def _build(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("""
            select l.path, l.model_name, l.version_name, l.stem, l.rel_dir, l.tags, l.trained_words, l.description,
                   l.lm_notes, c.kind, c.effect, c.medium, c.movement_era, c.palette, c.line_quality, c.suited_subjects,
                   c.artists as card_artists, c.triggers as card_triggers,
                   e.facets as edit_facets, e.artists as edit_artists, e.triggers as edit_triggers,
                   e.notes as edit_notes, e.creator as edit_creator,
                   f.facets as facet_facets, f.artists as facet_artists, f.creator as facet_creator
            from loras l left join cards c on c.path = l.path
            left join card_edits e on e.key = coalesce(lower(l.sha256), l.path)
            left join card_facets f on f.path = l.path""").fetchall()
        memory = self.memory
        memory.execute("drop table if exists docs")
        columns = ", ".join(name for name, _weight in COLUMNS)
        memory.execute(f"create virtual table docs using fts5(path unindexed, {columns}, "
                       f"tokenize = 'porter unicode61 remove_diacritics 2')")
        placeholders = ", ".join("?" for _ in range(len(COLUMNS) + 1))
        memory.executemany(f"insert into docs values ({placeholders})",
                           [[row["path"], *self._document(row)] for row in rows])
        memory.commit()
        self.count = len(rows)

    def ensure(self) -> None:
        with self._connect() as connection:
            stamp = self._stamp(connection)
            with self.lock:
                if stamp != self.stamp:
                    self._build(connection)
                    self.stamp = stamp

    # -- querying ------------------------------------------------------------------------------------------------
    @staticmethod
    def parse(query: str) -> Tuple[List[str], List[str]]:
        """Query text -> (FTS5 pieces that must match, pieces to leave out). Each piece is already FTS5 syntax."""
        wanted, unwanted = [], []
        for negative, field, phrase, word in TOKEN.findall(query or ""):
            column = FIELD_ALIASES.get(field.lower()) if field else None
            if field and column is None:            # "16:9" or a typo'd field: treat the whole thing as words
                word = f"{field} {phrase or word}"
                phrase = ""
            words = _words(_clean(phrase or word))
            if not words:
                continue
            if phrase:
                piece = '"' + " ".join(words) + '"'
            else:
                piece = " AND ".join(f'"{w}"*' for w in words)
                if len(words) > 1:
                    piece = f"({piece})"
            if column:
                piece = f"{column} : {piece}" if piece.startswith(("(", '"')) else f"{column} : ({piece})"
            (unwanted if negative else wanted).append(piece)
        return wanted, unwanted

    def _run(self, expression: str) -> Dict[str, float]:
        weights = ", ".join(str(weight) for _name, weight in COLUMNS)
        try:
            rows = self.memory.execute(f"select path, bm25(docs, 0, {weights}) from docs where docs match ?",
                                       (expression,)).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {path: round(-score, 4) for path, score in rows}       # bm25: lower is better

    def search(self, query: str) -> Dict[str, float]:
        """{path: score} for the LoRAs matching `query`; {} when nothing does (self.last says why)."""
        self.ensure()
        wanted, unwanted = self.parse(query)
        self.last = {"query": query, "mode": "none", "fts": ""}
        if not wanted:
            return {}
        exclude = "".join(f" NOT {piece}" for piece in unwanted)
        with self.lock:
            expression = " AND ".join(wanted) + exclude
            hits = self._run(expression)
            mode = "all"
            if not hits and len(wanted) > 1:
                expression = "(" + " OR ".join(wanted) + ")" + exclude
                hits = self._run(expression)
                mode = "any"
        self.last = {"query": query, "mode": mode if hits else "none", "fts": expression}
        return hits
