"""
What a prompt is of - a coarse subject label for your sweep marks (2026-10-16), so a LoRA's record can say "great on
portraits, poor on landscapes". Keyword based, instant, no model: the label with the most hits wins; ties go to the
earlier label in SUBJECTS (people before places before things).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import re

SUBJECTS = [
    ("portrait", ["portrait", "face", "headshot", "close-up of a", "closeup of a", "bust", "her eyes", "his eyes",
                  "freckles", "her face", "his face"]),
    ("figure", ["woman", "man", "girl", "boy", "lady", "gentleman", "person", "people", "figure", "model",
                "dancer", "warrior", "knight", "queen", "king", "princess", "witch", "soldier", "fisherman", "child",
                "couple", "she ", "he ", "her ", "his ", "woman1", "subject1"]),
    ("fashion", ["dress", "gown", "coat", "jacket", "outfit", "fashion", "latex", "corset", "kimono", "suit",
                 "skirt", "boots", "couture", "runway"]),
    ("animal", ["cat", "dog", "horse", "bird", "fox", "wolf", "owl", "deer", "lion", "tiger", "bear", "rabbit",
                "fish", "whale", "butterfly", "animal", "peacock", "elephant", "monkey", "snake", "cow",
                "insect", "beetle", "frog", "crow", "raven", "swan", "octopus"]),
    ("creature", ["dragon", "monster", "creature", "demon", "alien", "robot", "golem", "beast", "skeleton"]),
    ("landscape", ["landscape", "mountain", "lake", "forest", "field", "meadow", "valley", "river", "ocean", "sea",
                   "beach", "desert", "sky", "sunset", "sunrise", "dawn", "dusk", "waterfall", "hills", "coast",
                   "island", "snow", "garden"]),
    ("city", ["city", "street", "town", "village", "harbour", "harbor", "market", "skyline", "alley", "rooftop",
              "cityscape", "bridge"]),
    ("architecture", ["building", "cathedral", "church", "temple", "castle", "palace", "tower", "house", "ruins",
                      "architecture", "facade"]),
    ("interior", ["interior", "room", "kitchen", "library", "bedroom", "hall", "studio", "cafe", "bar ", "office"]),
    ("still life", ["still life", "bottle", "vase", "fruit", "pears", "apples", "bowl", "cup", "teapot", "flowers in",
                    "on a table", "on a wooden table", "tabletop"]),
    ("vehicle", ["car", "ship", "boat", "plane", "train", "motorcycle", "bicycle", "spaceship", "airship"]),
    ("food", ["food", "dish", "cake", "bread", "meal", "plate of"]),
    ("abstract", ["abstract", "pattern", "geometric", "texture", "fractal", "ornament"]),
]


def _text(prompt: str) -> str:
    """JSON prompts ({"scene": ...}) are read for their values, not their keys."""
    raw = str(prompt or "")
    if raw.lstrip().startswith("{"):
        try:
            values = []

            def walk(node):
                if isinstance(node, dict):
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)
                elif node is not None:
                    values.append(str(node))
            walk(json.loads(raw))
            raw = " ".join(values)
        except ValueError:
            pass
    return " " + re.sub(r"\s+", " ", raw.lower()) + " "


def subject_of(prompt: str) -> str:
    text = _text(prompt)
    if not text.strip():
        return ""
    best, best_hits = "other", 0
    for label, words in SUBJECTS:
        hits = 0
        for w in words:
            if w.endswith(" ") or " " in w.strip():
                hits += text.count(w)
            else:
                hits += len(re.findall(r"(?<![a-z])" + re.escape(w) + r"s?(?![a-z])", text))
        if hits > best_hits:
            best, best_hits = label, hits
    # a person described in detail is a portrait/figure even when the background names a place
    return best
