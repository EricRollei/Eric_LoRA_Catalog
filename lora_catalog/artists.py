"""
Artist matching: tie a LoRA to the artist it emulates by its NAME, before any model looks at it.

The dictionary is a names file (settings.json artist_names_file: {"names": {"Display Name": {"folded": ...}}}, e.g.
the one video_prompter exports from its wildcard artist lists, ~15k names). Optional. A LoRA's name, file stem and
trained words are cut into words and looked up by surname:

    full name   first and last name both present ("Boris Vallejo", "frazetta_frank")      score 3
    surname     the surname alone, exactly ("Magritte")                                    score 2
    near        the surname one letter off, for names of 5+ letters ("VALEJO" -> Vallejo)  score 1

A surname shared by more than MAX_OWNERS artists ("Smith") only counts with the first name. Generic LoRA words
(style, comic, krea, epoch ...) are never looked up. Candidates are EVIDENCE for the card writer, which confirms or
rejects them against the description and the images; they are never written to a card on their own.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional

MAX_OWNERS = 4
STOP = set("""
style styles art arts artist artists artstyle lora loras lycoris locon dora krea krea2 flux flux1 flux2 sdxl sd15 sd35 sd
xl illustrious pony noob noobai qwen wan zimage chroma hidream kontext ernie ideogram model models version final epoch
step steps dim rank v1 v2 v3 v4 v5 new old improved better best edition pack set collection type kind lite turbo raw
anime manga comic comics cartoon cartoons painting paintings paint painter painted illustration illustrations
illustrator drawing drawings drawn sketch sketches ink inks pencil watercolor watercolour watercolors oil oils acrylic
gouache pastel digital concept fantasy dark light retro vintage modern classic classical abstract realistic realism
real photo photos photography photographic cinematic cinema film movie poster posters cover covers book books card
cards game games video pixel vector flat line lines lineart color colour colors colours mono monochrome black white
red blue green gold golden silver pink purple orange yellow brown grey gray neon pop punk noir deco nouveau
renaissance baroque gothic horror scifi cyber cyberpunk steampunk dieselpunk solarpunk propaganda surreal surrealism
surrealist expressionism impressionism impressionist cubism minimal minimalist graphic graphics design designs world
worlds city cities girl girls boy boys woman women man men lady ladies character characters portrait portraits
landscape landscapes animal animals creature creatures monster monsters dragon dragons robot robots mecha
aesthetic aesthetics vibe vibes mood look looks effect effects filter detail details detailer quality master
masterpiece ultra super hyper mega the and of in by for with from yfg harlem chaos abstract
strokes stroke childrens children markers marker safari crime paste eyes sweet still factory heart moody liquid
fairytale dream press clean stamp custom styles arcane gothic eligos manhwa niji spot sweet roma blindfold
""".split())


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def words_of(text: str) -> List[str]:
    """Words of a LoRA name or file stem: camelCase, underscores, digits and ids split apart."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text or "")
    return [word for word in fold(re.sub(r"\d+", " ", text)).split() if len(word) >= 3]


def within_one(a: str, b: str) -> bool:
    """True when a and b differ by at most one insertion, deletion or substitution."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
        j += 1
    return edits + (len(b) - j) + (len(a) - i) <= 1


class ArtistIndex:
    def __init__(self, path: str):
        self.available = bool(path) and os.path.isfile(path)
        self.by_surname: Dict[str, List[tuple]] = defaultdict(list)     # surname -> [(display name, first-name words)]
        self.by_length: Dict[int, List[str]] = defaultdict(list)        # surname length -> surnames (for near matches)
        if not self.available:
            return
        with open(path, "r", encoding="utf-8") as handle:
            names = json.load(handle).get("names", {})
        for display, value in names.items():
            parts = (value.get("folded") or fold(display)).split()
            if not parts or len(parts[-1]) < 4 or parts[-1] in STOP:
                continue
            self.by_surname[parts[-1]].append((display, set(parts[:-1])))
        for surname in self.by_surname:
            self.by_length[len(surname)].append(surname)

    def match(self, *texts: str, limit: int = 3) -> List[Dict[str, object]]:
        """Candidate artists for a LoRA from its name texts, best first: [{name, score, how, token}]."""
        if not self.available:
            return []
        words: List[str] = []
        for text in texts:
            for word in words_of(text):
                if word not in words:
                    words.append(word)
        present = set(words)
        found: Dict[str, Dict[str, object]] = {}
        for word in words:
            if word in STOP or len(word) < 4:
                continue
            surnames = [word] if word in self.by_surname else []
            how = "surname"
            if not surnames and len(word) >= 6:
                surnames = [s for length in (len(word) - 1, len(word), len(word) + 1)
                            for s in self.by_length.get(length, []) if within_one(word, s)]
                how = "near"
            for surname in surnames:
                owners = self.by_surname[surname]
                for display, first_names in owners:
                    full = bool(first_names) and bool(first_names & present)
                    if not full and (len(owners) > MAX_OWNERS or (how == "near" and len(owners) > 2)):
                        continue
                    score = 3 if full else (2 if how == "surname" else 1)
                    if score > found.get(display, {}).get("score", 0):
                        found[display] = {"name": display, "score": score, "how": "full name" if full else how,
                                          "token": word}
        ranked = sorted(found.values(), key=lambda item: -item["score"])
        if ranked and ranked[0]["score"] == 3:      # a full name beats every surname-only guess ("Frank" -> Lisa Frank)
            ranked = [item for item in ranked if item["score"] == 3]
        return ranked[:limit]


_index: Optional[ArtistIndex] = None


def get_index(path: str) -> ArtistIndex:
    global _index
    if _index is None:
        _index = ArtistIndex(path)
    return _index
