"""
Reads the LoRA Catalog's `card_facets` table (one finished row per LoRA, written by `run.py facets` and kept up to
date by the card editor) for one platform. Read-only; cached until the database file changes.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ROLES = ("style", "transform", "treatment", "detailer", "environment", "lighting", "subject", "layout", "utility",
         "slider")

# A card whose role is a free word (typed in the editor) falls back to its card kind.
KIND_ROLE = {
    "setting": "environment", "lighting_colour": "lighting", "character": "subject", "person": "subject",
    "clothing": "subject", "pose_action": "subject", "concept_object": "subject", "effect_detail": "detailer",
    "quality_realism": "detailer", "photo_look": "style", "style_medium": "style", "tool_edit": "utility",
    "speed_distill": "utility", "motion": "utility",
}


# Tool LoRAs that the cards often file as a style or subject: decided from the name.
TOOL_NAME = re.compile(r"(?<![a-z])(slider|turbo|lightning|hyper[-_ ]?sd|distill(ed)?|lcm|speed[-_ ]?up|\d+[-_ ]?steps?)(?![a-z])", re.I)


def _list(value) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


@dataclass
class LoraCard:
    path: str
    name: str
    file_name: str
    rel_dir: str
    role: str
    facets: Dict[str, List[str]]
    nsfw: int
    artists: List[str]
    depicts: List[str]
    triggers: List[str]
    creator: str
    weight: float
    weight_low: Optional[float]
    weight_high: Optional[float]
    overrides: List[str]
    confidence: float
    kind: str
    effect: str
    favorite: bool
    exclude: bool
    reviewed: bool
    words: Dict[str, int] = field(default_factory=dict)     # lexical bag for text similarity
    rating: Optional[int] = None                             # your 1-5 stars from the card editor; None = unrated
    triggers_off: List[str] = field(default_factory=list)    # triggers you switched off: never added to prompts
    roles: List[str] = field(default_factory=list)           # every known role on the card, main one first
    slider: Optional[Dict] = None                            # slider card (axis, positive_toward, useful ...), 2026-10-08
    community: Optional[float] = None                        # Civitai community score 0-100 (run.py community), 2026-10-10
    community_group: str = ""                                # what it was ranked among, e.g. "Krea 2 style"

    def has(self, facet: str, term: str) -> bool:
        return term in (self.facets.get(facet) or [])

    @property
    def lora_name(self) -> str:
        """Path below the library root, the form ComfyUI's LoRA lists use."""
        parts = (self.rel_dir or "").replace("\\", "/").strip("/")
        return f"{parts}/{self.file_name}" if parts else self.file_name


WORD = re.compile(r"[a-z][a-z0-9'-]{2,}")
STOP = set("""the and with for from that this into onto its it's their them they you your are was were has have
had not but all any can will would could should very more most much many some such than then there these those
image images picture pictures style styles lora loras model models trained training version krea flux qwen sdxl
pony using used use like look looks looking also only each other well make makes made create creates creating
applies apply adds add gives give produces produce often tends tend""".split())


def words_of(text: str) -> List[str]:
    out = []
    for word in WORD.findall((text or "").lower()):
        word = word.strip("'-")
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word not in STOP and len(word) >= 3:
            out.append(word)
    return out


_cache: Dict[Tuple[str, str], tuple] = {}


def load_cards(catalog: str, shelves: List[str], base_models: List[str], known_shelves: set) -> Tuple[List[LoraCard], str]:
    """(cards for the platform, note). note explains an empty result."""
    if not os.path.exists(catalog):
        return [], f"LoRA catalog not found at {catalog} (set lora_suggest.catalog in config.json)"
    key = (catalog, "|".join(shelves) + "#" + "|".join(base_models))
    stamp = (os.path.getmtime(catalog), os.path.getmtime(catalog + "-wal") if os.path.exists(catalog + "-wal") else 0)
    cached = _cache.get(key)
    if cached and cached[0] == stamp:
        return cached[1], cached[2]
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        if "card_facets" not in tables:
            return [], "the catalog has no card_facets table yet - run `run.py facets` in the LoRA Catalog tool"
        rows = list(connection.execute("select * from card_facets"))
        columns = set(rows[0].keys()) if rows else set()
        missing = connection.execute("select count(*) from cards where path not in (select path from card_facets)").fetchone()[0]
        community: Dict[str, Tuple[float, str]] = {}     # path -> (score, group); a popularity prior, your stars win
        if "community_stats" in tables and "loras" in tables:
            for path, score, group in connection.execute(
                    "select l.path, s.score, s.score_group from loras l join community_stats s "
                    "on s.version_id = l.civitai_version_id where s.score is not null"):
                community[path] = (float(score), group or "")
    finally:
        connection.close()
    shelf_set = {s.casefold() for s in shelves}
    base_set = {b.casefold() for b in base_models}
    cards: List[LoraCard] = []
    for row in rows:
        shelf = (row["arch_folder"] or "").casefold()
        base = (row["base_model"] or "").casefold()
        if not (shelf in shelf_set or (shelf not in known_shelves and base and base in base_set)):
            continue
        facets = {}
        try:
            facets = json.loads(row["facets"] or "{}")
        except ValueError:
            pass
        slider = None
        if "slider" in columns and row["slider"]:
            try:
                slider = json.loads(row["slider"])
            except ValueError:
                slider = None
        role = row["role"] if row["role"] in ROLES else KIND_ROLE.get(row["kind"] or "", "style")
        if role == "slider" and not slider:
            role = "utility"          # marked a slider but no slider card yet: nothing to dial with
        elif role != "slider" and row["source"] != "edit" and TOOL_NAME.search(f"{row['name'] or ''} {row['file_name'] or ''}"):
            role = "utility"
        card = LoraCard(
            path=row["path"], name=row["name"] or row["file_name"], file_name=row["file_name"], rel_dir=row["rel_dir"] or "",
            role=role, facets={k: [str(t) for t in v] for k, v in facets.items() if isinstance(v, list)},
            nsfw=int(row["nsfw"] or 0), artists=[str(a) for a in _list(row["artists"]) if a],
            depicts=[str(d) for d in _list(row["depicts"]) if d], triggers=[str(t) for t in _list(row["triggers"]) if t],
            creator=row["creator"] or "", weight=float(row["weight_typical"] or 1.0),
            weight_low=row["weight_low"], weight_high=row["weight_high"],
            overrides=[str(o) for o in _list(row["overrides"])], confidence=float(row["confidence"] or 0.0),
            kind=row["kind"] or "", effect=row["effect"] or "", favorite=bool(row["favorite"]),
            exclude=bool(row["exclude"]), reviewed=row["source"] == "edit" and row["status"] in ("confirmed", "edited"),
            rating=(int(row["rating"]) if "rating" in columns and row["rating"] else None),
            triggers_off=([str(t) for t in _list(row["triggers_off"]) if t] if "triggers_off" in columns else []),
            slider=slider if role == "slider" else None,
            community=community.get(row["path"], (None, ""))[0],
            community_group=community.get(row["path"], (None, ""))[1],
        )
        listed = [str(r) for r in (card.facets.get("role") or []) if str(r) in ROLES]
        card.roles = [role] + [r for r in listed if r != role] if role not in ("utility", "slider") or row["source"] == "edit" else [role]
        if role == "slider":
            card.roles = ["slider"]       # a dial is only ever used as a dial
        bag: Dict[str, int] = {}
        for word in words_of(" ".join([card.name, card.effect, " ".join(card.triggers), " ".join(card.artists)])):
            bag[word] = bag.get(word, 0) + 1
        card.words = bag
        cards.append(card)
    note = ""
    if missing:
        note = f"{missing} newer card(s) are not in card_facets yet - run `run.py facets` in the LoRA Catalog tool"
    _cache[key] = (stamp, cards, note)
    return cards, note
