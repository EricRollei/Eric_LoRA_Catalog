"""
The pick lists (vocab.yaml in the tool folder): the fixed words each card field may use, as a tree.

    vocab = load_vocab()
    vocab.facets["period"].terms["high_renaissance"].parent      -> "renaissance"
    vocab.match("period", "1980s fantasy paperback cover art")   -> ["fantasy_paperback", "eighties_retro"]
    prefill(vocab, lora_row, card_row)                           -> {facet: [term ids]} read from an existing card

Matching is plain word-boundary search for each term's label and aliases, longest phrase first, so a long alias
("early 20th century illustration") claims its words before a short one ("illustration") can.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .config import user_file

VOCAB_PATH = user_file("vocab.yaml")             # your copy in the data folder (the shipped one is the start)


@dataclass
class Term:
    id: str
    label: str
    aliases: List[str]
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class Facet:
    name: str
    pick: str                      # "one" | "many"
    help: str
    terms: Dict[str, Term]         # every term, groups and children, in file order
    top: List[str]                 # ids of top-level terms in file order
    _phrases: List[tuple] = field(default_factory=list)   # (phrase, term id), longest first

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "pick": self.pick, "help": self.help, "top": self.top,
                "terms": {tid: {"label": t.label, "aliases": t.aliases, "parent": t.parent, "children": t.children}
                          for tid, t in self.terms.items()}}


class Vocab:
    def __init__(self, data: Dict[str, Any]):
        self.version = data.get("version", 1)
        self.facets: Dict[str, Facet] = {}
        for name, spec in (data.get("facets") or {}).items():
            facet = Facet(name=name, pick=spec.get("pick", "many"), help=spec.get("help", ""), terms={}, top=[])
            self._read_terms(facet, spec.get("terms") or {}, parent=None)
            phrases = []
            for term in facet.terms.values():
                for phrase in {term.label.lower(), term.id.replace("_", " ").lower(), *[a.lower() for a in term.aliases]}:
                    if phrase.strip():
                        phrases.append((phrase.strip(), term.id))
            facet._phrases = sorted(phrases, key=lambda item: -len(item[0]))
            self.facets[name] = facet

    def _read_terms(self, facet: Facet, terms: Dict[str, Any], parent: Optional[str]) -> None:
        for term_id, value in terms.items():
            if term_id == "_":
                continue
            if isinstance(value, dict):                     # a group
                aliases = [str(a) for a in (value.get("_") or [])]
                facet.terms[term_id] = Term(term_id, term_id.replace("_", " "), aliases, parent)
                self._register(facet, term_id, parent)
                self._read_terms(facet, value, parent=term_id)
            else:
                aliases = [str(a) for a in (value or [])]
                facet.terms[term_id] = Term(term_id, term_id.replace("_", " "), aliases, parent)
                self._register(facet, term_id, parent)

    @staticmethod
    def _register(facet: Facet, term_id: str, parent: Optional[str]) -> None:
        if parent is None:
            facet.top.append(term_id)
        else:
            facet.terms[parent].children.append(term_id)

    def match(self, facet_name: str, text: str, limit: int = 0) -> List[str]:
        """Term ids whose label or alias appears in `text`, in order of first appearance. Each stretch of text is
        claimed by the longest phrase that fits it."""
        facet = self.facets.get(facet_name)
        if facet is None or not text:
            return []
        lowered = " " + re.sub(r"[_/,;:()\[\]]+", " ", text.lower()) + " "
        claimed = [False] * len(lowered)
        hits: List[tuple] = []
        for phrase, term_id in facet._phrases:
            for found in re.finditer(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", lowered):
                start, end = found.span()
                if any(claimed[start:end]):
                    continue
                for i in range(start, end):
                    claimed[i] = True
                hits.append((start, term_id))
        ordered: List[str] = []
        for _start, term_id in sorted(hits):
            if term_id not in ordered:
                ordered.append(term_id)
        return ordered[:limit] if limit else ordered

    def ancestors(self, facet_name: str, term_id: str) -> List[str]:
        facet = self.facets[facet_name]
        chain = []
        term = facet.terms.get(term_id)
        while term is not None and term.parent:
            chain.append(term.parent)
            term = facet.terms.get(term.parent)
        return chain

    def known(self, facet_name: str, term_id: str) -> bool:
        facet = self.facets.get(facet_name)
        return facet is not None and term_id in facet.terms

    def to_json(self) -> Dict[str, Any]:
        return {"version": self.version, "order": list(self.facets), "facets": {n: f.to_json() for n, f in self.facets.items()}}


_cache: Dict[str, Any] = {}


def term_id_for(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (word or "").strip().lower()).strip("_")


def add_term(facet_name: str, word: str, parent: str = "", path: str = VOCAB_PATH) -> str:
    """Add `word` as a button (term) of a facet in vocab.yaml, at the top level or under a top-level group, keeping
    the file's comments and layout. Returns the term id. A dated backup is written before the first change of the day;
    if the edited file does not load back with the new term, the old file is put back."""
    import datetime
    import shutil
    term_id = term_id_for(word)
    if not term_id:
        raise ValueError("empty word")
    vocab = load_vocab(path)
    facet = vocab.facets.get(facet_name)
    if facet is None:
        raise ValueError(f"no pick list called {facet_name}")
    if term_id in facet.terms:
        return term_id
    if parent and (parent not in facet.terms or facet.terms[parent].parent is not None or not facet.terms[parent].children):
        raise ValueError(f"{parent} is not a top-level group of {facet_name}")
    with open(path, "r", encoding="utf-8") as handle:
        original = handle.read()
    lines = original.splitlines()
    head = next((i for i, line in enumerate(lines) if re.match(rf"^  {re.escape(facet_name)}:\s*$", line)), None)
    if head is None:
        raise ValueError(f"{facet_name} not found in vocab.yaml")
    terms_at = next((i for i in range(head + 1, len(lines)) if re.match(r"^    terms:\s*$", lines[i])
                     or re.match(r"^  \S", lines[i])), None)
    if terms_at is None or not lines[terms_at].startswith("    terms:"):
        raise ValueError(f"{facet_name} has no terms: block")

    def block_end(start: int, indent: int) -> int:
        """Index after the last non-blank line, from start, whose indent is at least `indent`."""
        last = start
        for i in range(start + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(lines[i]) - len(lines[i].lstrip(" ")) < indent:
                break
            last = i
        return last + 1

    if parent:
        group_at = next((i for i in range(terms_at + 1, block_end(terms_at, 6))
                         if re.match(rf"^      {re.escape(parent)}:\s*$", lines[i])), None)
        if group_at is None:
            raise ValueError(f"group {parent} not found under {facet_name}")
        at, new_line = block_end(group_at, 8), f"        {term_id}: []"
    else:
        at, new_line = block_end(terms_at, 6), f"      {term_id}: []"
    lines.insert(at, new_line)
    stamp = datetime.date.today().isoformat()
    backup = f"{path}.bak_{stamp}"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    try:
        _cache.clear()
        if term_id not in load_vocab(path).facets[facet_name].terms:
            raise ValueError("term missing after write")
    except Exception:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(original)
        _cache.clear()
        raise
    return term_id


def load_vocab(path: str = VOCAB_PATH) -> Vocab:
    """Re-read when the file changes, so edits to vocab.yaml show up without restarting the editor."""
    stamp = os.path.getmtime(path)
    if _cache.get("stamp") != stamp:
        with open(path, "r", encoding="utf-8") as handle:
            _cache["vocab"] = Vocab(yaml.safe_load(handle) or {})
        _cache["stamp"] = stamp
    return _cache["vocab"]


# ---------------------------------------------------------------------------------------------------------------
# Prefill: read an existing (free-text) card into pick-list terms, so the editor opens with the model's view
# already ticked. A guess, shown as a guess; the new card pass will choose terms directly.

KIND_TO_ROLE = {
    "setting": "environment", "lighting_colour": "lighting", "character": "subject", "person": "subject",
    "clothing": "subject", "pose_action": "subject", "concept_object": "subject", "effect_detail": "detailer",
    "quality_realism": "detailer", "photo_look": "style", "style_medium": "style",
    "tool_edit": "utility", "speed_distill": "utility", "motion": "utility",
}
TREATMENT_HINTS = re.compile(r"^\s*(adds?|overlays?|applies (a )?(texture|grain|halftone|paper))|sketch lines|line overlay|"
                             r"halftone|paper texture|film grain|colou?r grade", re.I)
TRANSFORM_HINTS = re.compile(r"surreal|dream-?like|dreamscape|impossible|melting|morph|reality[- ]bending|"
                             r"glitch|hallucinat|psychedelic distortion", re.I)
DETAILER_HINTS = re.compile(r"^\s*(enhances?|improves?|sharpens?|refines?)\b|\bdetail(er| enhancer)s?\b", re.I)


def _text_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        value = parsed
    if isinstance(value, list):
        return " ; ".join(str(v.get("name", v) if isinstance(v, dict) else v) for v in value)
    return str(value)


def _get(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def guess_role(card: Any) -> List[str]:
    kind = _get(card, "kind") or ""
    effect = _get(card, "effect") or ""
    role = KIND_TO_ROLE.get(kind, "style")
    if kind in ("style_medium", "photo_look", "effect_detail"):
        era = _get(card, "movement_era") or ""
        if DETAILER_HINTS.search(effect):
            role = "detailer"
        elif TREATMENT_HINTS.search(effect):
            role = "treatment"
        elif TRANSFORM_HINTS.search(effect) or TRANSFORM_HINTS.search(str(era)):
            role = "transform"
    return [role]


NSFW_TERMS = ["sfw", "suggestive", "nudity", "explicit"]


def _card_nsfw(card: Any) -> int:
    """0 none .. 3 explicit. Rows from the editor carry the card's value as card_nsfw (l.nsfw_level is Civitai's)."""
    for key in ("card_nsfw", "nsfw_level"):
        value = _get(card, key)
        if isinstance(value, int) and 0 <= value <= 3:
            return value
    return 0


def _civitai_nsfw(lora: Any) -> int:
    """Civitai nsfwLevel is a bit mask over the example images: 1 PG, 2 PG-13, 4 R, 8 X, 16 XXX."""
    value = _get(lora, "nsfw_level")
    if not isinstance(value, int) or value <= 1 or value > 63:
        return 0
    if value & 24:
        return 3
    if value & 4:
        return 2
    return 1 if value & 2 else 0


def prefill(vocab: Vocab, lora: Any, card: Any) -> Dict[str, List[str]]:
    effect = _get(card, "effect") or ""
    name = " ".join(filter(None, [_get(lora, "model_name"), _get(lora, "version_name")]))
    tags = _text_list(_get(lora, "tags"))
    medium = _text_list(_get(card, "medium"))
    era = _text_list(_get(card, "movement_era"))
    palette = _text_list(_get(card, "palette"))
    lines = _text_list(_get(card, "line_quality"))
    subjects = _text_list(_get(card, "suited_subjects"))
    overrides = _text_list(_get(card, "overrides")).split(" ; ")
    out: Dict[str, List[str]] = {}
    out["role"] = guess_role(card) if card is not None else []
    out["use"] = vocab.match("use", " ; ".join([name, tags, effect, era]))
    out["medium"] = vocab.match("medium", " ; ".join([medium, effect]))
    out["period"] = vocab.match("period", " ; ".join([era, tags]))
    # genre / mood (2026-10-10): the card writer's own genre and mood words when the card has them (v2 cards), else
    # read from the name, tags, era and effect of older cards
    # run.py genre-mood writes the labels ('' = classified, none fits); NULL = not classified yet -> a guess from the
    # name / tags / era / effect until it is
    genre_raw, mood_raw = _get(card, "genre"), _get(card, "mood")
    genre_text, mood_text = _text_list(genre_raw), _text_list(mood_raw)
    out["genre"] = (vocab.match("genre", genre_text) if genre_raw is not None else
                    vocab.match("genre", " ; ".join([name, tags, era, effect])))
    lighting_words = re.sub(r"\b(dramatic|moody|soft|warm|dark)\s+(lighting|light|lights|shadows?|contrast|chiaroscuro|"
                            r"highlights?|tones?|colou?rs?)\b", " ", " ; ".join([effect, palette]), flags=re.I)
    out["mood"] = vocab.match("mood", mood_text) if mood_raw is not None else vocab.match("mood", lighting_words)
    out["rendering"] = vocab.match("rendering", " ; ".join([lines, effect]), limit=1)
    out["dimension"] = vocab.match("dimension", effect, limit=1) or (["two_d"] if out["role"] in (["style"], ["treatment"]) and "photo" not in effect.lower() else [])
    out["palette"] = vocab.match("palette", palette)
    out["line"] = vocab.match("line", lines)
    out["subjects"] = vocab.match("subjects", subjects)
    everything = " ; ".join([name, tags, effect])
    for facet_name in ("lighting", "composition", "pattern", "material", "photo_style", "cinematic", "film_stock"):
        out[facet_name] = vocab.match(facet_name, everything)
    out["detail"] = vocab.match("detail", everything) if out["role"] == ["detailer"] else []
    # characters / people
    kind = _get(card, "kind") or ""
    if out["role"] == ["subject"] or kind in ("person", "character", "clothing"):
        for facet_name in ("figure", "appearance", "costume"):
            out[facet_name] = vocab.match(facet_name, everything)
    # NSFW: the card's own judgement of what the LoRA is for. Civitai's rating describes the example images people
    # posted, which over-counts; the editor shows it as a hint only.
    out["nsfw"] = [NSFW_TERMS[_card_nsfw(card)]]
    out["mature_content"] = vocab.match("mature_content", everything) if out["nsfw"] != ["sfw"] else []
    side = vocab.match("side_effects", effect)
    if "text" in overrides and "adds_text" not in side:
        side.append("adds_text")
    out["side_effects"] = side
    for facet_name, picked in out.items():          # a 'one' facet keeps only its first pick
        facet = vocab.facets.get(facet_name)
        if facet is not None and facet.pick == "one":
            out[facet_name] = picked[:1]
    return out
