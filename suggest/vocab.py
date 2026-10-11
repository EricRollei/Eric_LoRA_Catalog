"""
The LoRA Catalog's pick lists (vocab.yaml), read as trees: each facet has terms, a term may have a parent group.
Used to read a prompt in the same words the cards are written in, and to score near misses (a prompt asking for
"renaissance" matches a card that says "high_renaissance").

The format is owned by the LoRA Catalog (lora_catalog/vocab.py; your copy is in the data folder); this is a small
read-only reader so the Suggester's engine needs nothing from the catalog's code.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class FacetTree:
    name: str
    pick: str
    parent: Dict[str, Optional[str]] = field(default_factory=dict)       # term -> parent group (or None)
    phrases: List[Tuple[str, str]] = field(default_factory=list)          # (phrase, term), longest first

    def ancestors(self, term: str) -> List[str]:
        chain, seen = [], set()
        current = self.parent.get(term)
        while current and current not in seen:
            chain.append(current)
            seen.add(current)
            current = self.parent.get(current)
        return chain


class PickLists:
    def __init__(self, data: dict):
        self.facets: Dict[str, FacetTree] = {}
        for name, spec in ((data or {}).get("facets") or {}).items():
            tree = FacetTree(name=str(name), pick=str((spec or {}).get("pick", "many")))
            self._read(tree, (spec or {}).get("terms") or {}, None)
            tree.phrases.sort(key=lambda item: -len(item[0]))
            self.facets[tree.name] = tree

    def _read(self, tree: FacetTree, terms: dict, parent: Optional[str]) -> None:
        for term, value in terms.items():
            if term == "_":
                continue
            term = str(term)
            tree.parent[term] = parent
            aliases = value.get("_") if isinstance(value, dict) else value
            words = {term.replace("_", " ").lower(), *[str(a).lower() for a in (aliases or [])]}
            tree.phrases.extend((word.strip(), term) for word in words if word.strip())
            if isinstance(value, dict):
                self._read(tree, value, term)

    def match(self, facet: str, text: str) -> List[str]:
        """Terms whose label or alias appears in text (word boundaries; longest phrase claims its words first)."""
        tree = self.facets.get(facet)
        if tree is None or not text:
            return []
        lowered = " " + re.sub(r"[_/,;:()\[\]]+", " ", text.lower()) + " "
        claimed = [False] * len(lowered)
        hits: List[Tuple[int, str]] = []
        for phrase, term in tree.phrases:
            for found in re.finditer(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", lowered):
                start, end = found.span()
                if any(claimed[start:end]):
                    continue
                for i in range(start, end):
                    claimed[i] = True
                hits.append((start, term))
        ordered: List[str] = []
        for _start, term in sorted(hits):
            if term not in ordered:
                ordered.append(term)
        return ordered

    def similarity(self, facet: str, asked: str, card_term: str) -> float:
        """1 same term; 0.7 the card is narrower than the ask (ask 'renaissance', card 'high_renaissance');
        0.5 the card is broader; 0.25 siblings; 0 otherwise. Free words (not in the lists) only match exactly."""
        if asked == card_term:
            return 1.0
        tree = self.facets.get(facet)
        if tree is None or asked not in tree.parent or card_term not in tree.parent:
            return 0.0
        if asked in tree.ancestors(card_term):
            return 0.7
        if card_term in tree.ancestors(asked):
            return 0.5
        parent = tree.parent.get(asked)
        if parent and parent == tree.parent.get(card_term):
            return 0.25
        return 0.0


_cache: Dict[str, tuple] = {}


def load_pick_lists(path: str) -> Optional[PickLists]:
    """None when the file is missing. Re-read when it changes."""
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    cached = _cache.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    with open(path, "r", encoding="utf-8") as handle:
        lists = PickLists(yaml.safe_load(handle) or {})
    _cache[path] = (stamp, lists)
    return lists
