"""
The Suggester's `exclude` and `pin` boxes: what a line in them means and which LoRAs it matches (2026-10-08).

One entry per line, or several on a line separated by commas. Case does not matter. Matching is WHOLE WORDS:
punctuation, _ - . / % count as spaces, so `age` matches "Age Slider" and "krea2_age_v2" but not "image" or "vintage",
and `hair` matches "Hair Length Slider" but not "chair". Several words must appear together in that order:
`waist slider`.

    plain words          the LoRA's name or file name               vallejo / waist slider / hair
    folder:<words>       its folder path below the LoRA root        folder:sliders / folder:art-style
    role:<role>          its Suggester role                         role:treatment / role:slider
    axis:<axis>          sliders that move this axes.yaml axis      axis:age / axis:body weight

Pins add a value with =  (the weight in all three stages; sliders keep their stage shape):
    waist slider=-8      that LoRA at -8
    axis:age=+3          the best slider on the age axis at +3
A plain-name pin uses ONE LoRA: the best match (your stars, LoRA Manager favourite, card confidence); the report
lists the others it also matched. folder: and role: pins are not supported (they would pin an arbitrary LoRA).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

KINDS = ("folder", "role", "axis")
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass
class Term:
    kind: str                 # name | folder | role | axis
    text: str                 # normalised: lower case words separated by single spaces
    raw: str                  # as typed (for reports)
    value: Optional[float] = None
    bad_value: bool = False   # an '=' with something that is not a number


def words(text: str) -> str:
    """' w1 w2 ... ' - lower case, every run of non-letters/digits one space, padded for whole-word tests."""
    return " " + " ".join(_NON_WORD.sub(" ", (text or "").lower()).split()) + " "


def _number(text: str) -> Optional[float]:
    try:
        return float(text.strip().replace("+", ""))
    except ValueError:
        return None


def parse(text: str) -> List[Term]:
    terms: List[Term] = []
    for chunk in (text or "").splitlines():
        for raw in chunk.split(","):
            raw = raw.strip()
            if not raw:
                continue
            body, value, bad = raw, None, False
            if "=" in raw:
                left, _sep, right = raw.rpartition("=")
                value = _number(right)
                if value is None:
                    bad = True
                body = left.strip()
            kind = "name"
            head, sep, rest = body.partition(":")
            if sep and head.strip().lower() in KINDS:
                kind, body = head.strip().lower(), rest
            text_norm = words(body).strip()
            if text_norm:
                terms.append(Term(kind, text_norm, raw, value, bad))
    return terms


def _axis_id(text: str) -> str:
    return text.replace(" ", "_")


def matches(term: Term, card) -> bool:
    """Does this term select this card (a catalog.LoraCard)?"""
    if term.kind == "name":
        return f" {term.text} " in words(f"{card.name} {card.file_name}")
    if term.kind == "folder":
        return f" {term.text} " in words(card.rel_dir)
    if term.kind == "role":
        role = _axis_id(term.text)
        return card.role == role or role in (card.roles or [])
    if term.kind == "axis":
        return bool(card.slider) and str(card.slider.get("axis") or "") == _axis_id(term.text)
    return False


def first_match(terms: List[Term], card) -> Optional[Term]:
    for term in terms:
        if matches(term, card):
            return term
    return None


def describe(term: Term) -> str:
    return term.raw
