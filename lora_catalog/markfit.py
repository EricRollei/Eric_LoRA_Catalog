"""
Prompt fit and prompt type for a sweep mark (2026-10-17): was the prompt in this LoRA's wheelhouse?

A mark says how well a LoRA did on one prompt. Whether that was the LoRA or the prompt is only separable when we know
how close the prompt was to what the LoRA was made for. This module measures that, without an LLM:

    prompt_type(subject)      the coarse kind of prompt: portrait / figure / creature / scene / object / abstract / other
    FitIndex.fit(path, text)  0..1, how close the prompt is to the LoRA's own prompts:
        subject share  the share of its own prompts (creator examples x2, Civitai community prompts x1) that have the
                       same subject label (subjects.py) as this prompt - 0.45
        type share     the same at the coarse prompt-type level (portrait / figure count half for each other) - 0.20
        words          tf-idf cosine to its closest own prompts (mean of the best 3), /0.35, capped at 1   - 0.35
      A LoRA with fewer than MIN_PROMPTS own prompts falls back to its card's suited_subjects (subject and type
      match only, source "card"); none of either -> fit None (unknown).

The base prompt is the cell's prompt WITHOUT the LoRA's added trigger / style words (those describe the LoRA).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import collections
import json
import math
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .subjects import subject_of

TYPE_OF = {"portrait": "portrait", "figure": "figure", "fashion": "figure", "animal": "creature", "creature": "creature",
           "landscape": "scene", "city": "scene", "architecture": "scene", "interior": "scene",
           "still life": "object", "vehicle": "object", "food": "object", "abstract": "abstract"}
PROMPT_TYPES = ["portrait", "figure", "creature", "scene", "object", "abstract", "other"]
# A portrait is a close figure: half credit between the two (a LoRA made on women in a garden handles a face too).
RELATED = {("portrait", "figure"): 0.5, ("figure", "portrait"): 0.5}
MIN_PROMPTS = 5
MAX_PROMPTS = 200            # per LoRA, enough for a profile
WORDS_CAP = 0.35             # a mean best-3 cosine of 0.35 already means "the same kind of prompt"
WORD = re.compile(r"[a-z][a-z'-]{2,}")
STOP = set("""the and with for from that this into onto its it's their them they you your are was were has have had not
but all any can will would could should very more most much many some such than then there these those her his she
him he who what when where while over under behind front side image photo picture style detailed highly quality
masterpiece best high ultra realistic background lighting light shot view wearing holding looking standing sitting""".split())


def prompt_type(subject: str) -> str:
    if not subject:
        return ""
    return TYPE_OF.get(subject, "other")


def base_prompt(prompt: str, words: str) -> str:
    """The prompt without the LoRA words the sweep put in front of it (stored comma-joined in the mark's `words`)."""
    text = str(prompt or "")
    for word in [w.strip() for w in str(words or "").split(",") if w.strip()]:
        text = re.sub(r"(^|,\s*)" + re.escape(word) + r"\s*(,|$)", r"\1", text, count=1, flags=re.I)
    return text.strip(" ,")


def _tokens(text: str) -> List[str]:
    out = []
    for w in WORD.findall(str(text or "").lower()):
        w = w.strip("'-")
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in STOP and len(w) >= 3:
            out.append(w)
    return out


class FitIndex:
    """Per-LoRA prompt profiles, built lazily and cached for CACHE_SECONDS (marks come in fast; the prompts change
    only with a catalog refresh)."""

    CACHE_SECONDS = 1800

    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self.lock = threading.Lock()
        self.idf: Optional[Dict[str, float]] = None
        self.idf_at = 0.0
        self.profiles: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.catalog_path}?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _idf(self, connection: sqlite3.Connection) -> Dict[str, float]:
        if self.idf is not None and time.time() - self.idf_at < self.CACHE_SECONDS:
            return self.idf
        counts: collections.Counter = collections.Counter()
        n = 0
        for (text,) in connection.execute("SELECT coalesce(cleaned, prompt) FROM community_prompts "
                                          "UNION ALL SELECT prompt FROM examples WHERE prompt <> ''"):
            words = set(_tokens(text))
            if words:
                n += 1
                counts.update(words)
        self.idf = {w: math.log(1 + n / c) for w, c in counts.items()}
        self.idf_at = time.time()
        return self.idf

    def _vector(self, text: str, idf: Dict[str, float]) -> Dict[str, float]:
        tf = collections.Counter(_tokens(text))
        vec = {w: c * idf.get(w, 0.0) for w, c in tf.items() if idf.get(w)}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def profile(self, connection: sqlite3.Connection, path: str) -> Optional[Dict[str, Any]]:
        cached = self.profiles.get(path)
        if cached and time.time() - cached[0] < self.CACHE_SECONDS:
            return cached[1]
        row = connection.execute("SELECT l.civitai_version_id, c.suited_subjects FROM loras l LEFT JOIN cards c "
                                 "ON c.path = l.path WHERE l.path = ?", (path,)).fetchone()
        if row is None:
            self.profiles[path] = (time.time(), None)
            return None
        prompts: List[Tuple[str, float]] = [(p, 2.0) for (p,) in connection.execute(
            "SELECT prompt FROM examples WHERE path = ? AND prompt <> '' LIMIT ?", (path, MAX_PROMPTS))]
        if row["civitai_version_id"]:
            prompts += [(p, 1.0) for (p,) in connection.execute(
                "SELECT coalesce(cleaned, prompt) FROM community_prompts WHERE version_id = ? LIMIT ?",
                (row["civitai_version_id"], MAX_PROMPTS))]
        prompts = [(p, w) for p, w in prompts if p and len(str(p).split()) >= 3]
        claims = []
        try:
            claims = [str(x) for x in json.loads(row["suited_subjects"] or "[]")]
        except (TypeError, ValueError):
            claims = []
        claimed = {subject_of(c) for c in claims} - {"", "other"}
        if len(prompts) < MIN_PROMPTS:
            prof = {"source": "card", "subjects": {s: 1.0 / len(claimed) for s in claimed}, "vectors": [],
                    "n": len(prompts), "claimed": sorted(claimed)} if claimed else None
            self.profiles[path] = (time.time(), prof)
            return prof
        idf = self._idf(connection)
        subjects: collections.Counter = collections.Counter()
        total = 0.0
        vectors = []
        seen = set()
        for text, weight in prompts:
            key = str(text)[:150].lower()
            if key in seen:
                continue
            seen.add(key)
            subjects[subject_of(text) or "other"] += weight
            total += weight
            vectors.append(self._vector(text, idf))
        prof = {"source": "prompts", "subjects": {s: v / total for s, v in subjects.items()}, "vectors": vectors,
                "n": len(vectors), "claimed": sorted(claimed)}
        self.profiles[path] = (time.time(), prof)
        return prof

    def fit(self, path: str, prompt: str, connection: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """{fit, subject_share, type_share, words, source, n} or None when nothing is known about the LoRA's range."""
        own = connection is None
        connection = connection or self._connect()
        try:
            with self.lock:
                prof = self.profile(connection, path)
                if not prof:
                    return None
                subject = subject_of(prompt) or "other"
                kind = prompt_type(subject)
                subject_share = prof["subjects"].get(subject, 0.0)
                type_share = min(1.0, sum(v * (1.0 if prompt_type(s) == kind else RELATED.get((kind, prompt_type(s)), 0.0))
                                          for s, v in prof["subjects"].items()))
                words = None
                if prof["vectors"]:
                    idf = self._idf(connection)
                    q = self._vector(prompt, idf)
                    sims = sorted((sum(q.get(w, 0.0) * v for w, v in vec.items()) for vec in prof["vectors"]),
                                  reverse=True)[:3]
                    words = min(1.0, (sum(sims) / len(sims)) / WORDS_CAP) if sims else 0.0
                if words is None:
                    value = 0.7 * subject_share + 0.3 * type_share
                else:
                    value = 0.45 * subject_share + 0.20 * type_share + 0.35 * words
                return {"fit": round(value, 3), "subject_share": round(subject_share, 3), "type_share": round(type_share, 3),
                        "words": None if words is None else round(words, 3), "source": prof["source"], "n": prof["n"],
                        "own_top": sorted(prof["subjects"].items(), key=lambda kv: -kv[1])[:3]}
        finally:
            if own:
                connection.close()

    def own_types(self, path: str) -> Dict[str, float]:
        """What the LoRA was made for, as prompt-type shares of its own prompts (or its card's claims)."""
        connection = self._connect()
        try:
            with self.lock:
                prof = self.profile(connection, path)
        finally:
            connection.close()
        out: Dict[str, float] = {}
        for s, v in (prof or {}).get("subjects", {}).items():
            out[prompt_type(s)] = out.get(prompt_type(s), 0.0) + v
        return out
