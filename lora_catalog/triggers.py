"""
Trigger recipes: what to put in the prompt so a LoRA shows (run.py triggers).

The Civitai trigger list is often one token, sometimes a long phrase, sometimes several options, and the model page often
says "use this / try this in the prompt" in prose the API does not return. This step gathers the evidence

  - the full Civitai trained words and the card's triggers,
  - the model page description (HTML stripped, hint sentences first),
  - the creator's local example prompts, ranked by how much their image looks like the LoRA's style (CSD: cosine to
    the LoRA signature minus cosine to the average LoRA, from the per-image cache the style step fills),
  - phrases that recur in the LoRA's community prompts but are rare in Krea 2 prompts at large,

and asks the local text model (settings.genre_model) for a recipe: the trigger, a short style phrase, setting hints,
words to avoid and the page's advice. From the recipe it builds named prompt variants

  none / trigger / official / trigger+phrase / full

with duplicates merged, for the trigger-test sweep (run.py sweep-plan --trigger-test) to compare. Results go to the
trigger_recipes table; cards and card edits are not touched. Incremental: a LoRA is redone only when its evidence
changed (or with --force).

The refresh step (--refresh, 2026-10-20) is narrower: it writes recipes only for LoRAs that have none yet, or whose own
Civitai trained words or page description changed. New community prompts or a shifted phrase list (the library grows
under it) do not redo a recipe there - run `run.py triggers` by hand for that. A LoRA the model failed on twice is not
retried by the refresh (meta 'trigger_failures'; `run.py triggers --names <name>` or --force retries it).

(c) Eric Hiss. Dual license, see LICENSE.
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS trigger_recipes (
    path TEXT PRIMARY KEY,
    model TEXT,
    recipe TEXT,            -- JSON: the model's answer
    variants TEXT,          -- JSON: [{"name", "text", "covers": [names merged into it]}]
    evidence TEXT,          -- JSON: the mined phrases and top prompts the model saw (for the editor / checking)
    evidence_sig TEXT,
    updated_at TEXT
)
"""

VARIANT_ORDER = ["none", "trigger", "official", "trigger+phrase", "full"]
DEFAULT_VARIANT = "trigger+phrase"
FULL_MAX_WORDS = 90
PHRASE_MAX_WORDS = 40
PILOT_NAMES = ["michaelparkes", "v67_ukiyo-e", "zy_cinematic", "167_nckr2", "042_nckr2"]
PILOT_UNCLEAR = 20

SYSTEM = """You help a person get a LoRA adapter for the Krea 2 image model to show its effect. You read the evidence
about one LoRA and write its prompt recipe.

Evidence you get: the Civitai "trained words" (the official trigger list - sometimes one token, sometimes a long phrase,
sometimes several alternatives, sometimes junk or truncated), the model page description, the creator's own example
prompts (the ones whose images look most like the LoRA's style come first), and phrases that recur in prompts people
used with this LoRA but are rare in Krea 2 prompts in general.

Write:
- trigger: the activation token the LoRA was trained on - almost always a made-up word or handle ("mparkes",
  "n0vuschr0ma style", "zy_cinematic", "@homutan", "mobe-st4ill"), copied exactly as it appears in the trained words,
  the page or the prompts. Ordinary English words ("a stylized illustration", "cinematic") are not a trigger. A
  made-up token that appears in most of the LoRA's prompts IS its trigger even when no trained words are listed.
  Never invent one (not from the LoRA's or artist's name either). Empty string if there is none.
- trained_words_fixed: the trained words as separate items, weights like (x:1.4) removed, obvious truncation or typos
  repaired from the other evidence ("agic realism" -> "magic realism"). Empty list if there are none.
- style_phrase: at most 40 words of plain prompt text that describes the LoRA's look (medium, technique, line,
  palette, lighting, era), taken from the creator's own words first (trained words, page, example prompts), then
  from phrases users repeat. The captioner's card text is a last resort. Do not repeat the trigger. No weights, no
  subject matter (people, clothing, places, poses), no quality or rating tags ("masterpiece, best quality, highres,
  safe, mature"), no "medium:" labels.
- extra_phrases: up to 4 short style phrases the page says to add, or that most good prompts carry ("streaky",
  "cinematic still image"). Never subject matter, never quality tags. Empty list if none.
- setting_hints: things the page says about strength/weight, sampler, steps, CFG, resolution, or combining with
  other LoRAs. Quote numbers as given. Empty list if none.
- avoid: words or settings the page says to avoid. Empty list if none.
- page_advice: one or two sentences: what the page says about how to prompt it. Empty string if it says nothing.
- needs_trigger: "yes" if the page or evidence says the trigger is required, "no" if it works without one, "unclear".
- note: one short sentence on anything odd (truncated or junk trained words, a trigger shared with other LoRAs from
  the same creator, trained words that are really a whole example prompt).

Only use what the evidence supports. Do not invent tokens."""


def schema() -> Dict[str, Any]:
    strings = {"type": "array", "items": {"type": "string"}}
    return {"type": "json_schema", "json_schema": {"name": "trigger_recipe", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["trigger", "trained_words_fixed", "style_phrase", "extra_phrases", "setting_hints", "avoid",
                     "page_advice", "needs_trigger", "note"],
        "properties": {
            "trigger": {"type": "string"},
            "trained_words_fixed": strings,
            "style_phrase": {"type": "string"},
            "extra_phrases": strings,
            "setting_hints": strings,
            "avoid": strings,
            "page_advice": {"type": "string"},
            "needs_trigger": {"type": "string", "enum": ["yes", "no", "unclear"]},
            "note": {"type": "string"},
        }}}}


def ensure(connection: sqlite3.Connection) -> None:
    connection.execute(SCHEMA)
    connection.commit()


# -- text helpers ------------------------------------------------------------------------------------------------------
LORA_TAG = re.compile(r"<(?:lora|lyco|hypernet):[^>]*>", re.I)
WEIGHTED = re.compile(r"[\(\[]+([^():\[\]]+?)(?::\s*[\d.]+)?[\)\]]+")
SPLIT = re.compile(r"[,;\n\.\|]+")
WORD = re.compile(r"[a-z0-9][a-z0-9_'\-]*")
TAGS = re.compile(r"<[^>]+>")
HINT = re.compile(r"\b(trigger|activation|keyword|token|prompt|use|try|recommend|suggest|strength|weight|cfg|steps|"
                  r"sampler|scheduler|resolution|avoid|negative|works best|best with|add|include)\b", re.I)
STOP = set("""a an the and or of in on at to with for from by as is are was be this that these those it its into over
under very some any each his her their your my our up down out off then than so but not no""".split())
QUALITY = {"masterpiece", "best quality", "high quality", "highly detailed", "ultra detailed", "8k", "4k", "hd",
           "uhd", "hdr", "absurdres", "absurdre", "highres", "detailed", "intricate details", "sharp focus", "safe",
           "mature", "adult", "very aesthetic", "amazing quality", "newest", "detailed masterpiece", "digital painting",
           "score_9", "score_8_up", "score_7_up"}
MEDIUM_LABEL = re.compile(r"\bmedium\s*:\s*", re.I)


def _json_list(value: Any) -> List[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return [value]
    return parsed if isinstance(parsed, list) else [parsed]


def normalize(text: str) -> str:
    """Lower case, LoRA tags out, (word:1.3) -> word, brackets and spare spaces gone."""
    text = LORA_TAG.sub(" ", str(text or ""))
    for _ in range(3):
        text = WEIGHTED.sub(r"\1", text)
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"(?i)<\s*(br|/p|/li|/h\d|/div)\s*/?>", "\n", str(text or ""))
    text = html.unescape(TAGS.sub(" ", text))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _candidates(text: str) -> set:
    """Comma chunks of up to 6 words, and word 1-3 grams that neither start nor end on a stop word."""
    out = set()
    for chunk in SPLIT.split(normalize(text)):
        words = WORD.findall(chunk)
        if not words:
            continue
        if len(words) <= 6:
            out.add(" ".join(words))
        for n in (1, 2, 3):
            for start in range(len(words) - n + 1):
                gram = words[start:start + n]
                if gram[0] in STOP or gram[-1] in STOP or (n == 1 and len(gram[0]) < 3):
                    continue
                out.add(" ".join(gram))
    return {c for c in out if c not in QUALITY and not c.isdigit()}


class Background:
    """How often each candidate phrase appears in Krea 2 prompts at large (one count per prompt)."""

    def __init__(self, connection: sqlite3.Connection, arch: str = "Krea2"):
        self.counts: collections.Counter = collections.Counter()
        self.n = 0
        rows = connection.execute(
            "SELECT coalesce(cp.cleaned, cp.prompt) FROM community_prompts cp JOIN loras l "
            "ON l.civitai_version_id = cp.version_id WHERE l.arch_folder = ? "
            "UNION ALL SELECT e.prompt FROM examples e JOIN loras l ON l.path = e.path WHERE l.arch_folder = ?",
            (arch, arch)).fetchall()
        seen = set()
        for (prompt,) in rows:
            key = normalize(prompt or "")
            if not key or key in seen:
                continue
            seen.add(key)
            self.n += 1
            self.counts.update(_candidates(key))

    def rate(self, phrase: str) -> float:
        return (self.counts.get(phrase, 0) + 0.5) / (self.n + 1)


def mine_phrases(prompts: List[Tuple[str, float]], background: Background, top: int = 15) -> List[Dict[str, Any]]:
    """Phrases that recur in this LoRA's prompts (weighted share >= 15%, in 2+ prompts) far more than in the
    background; longer phrases win over the shorter ones they contain at a similar share."""
    total = sum(w for _p, w in prompts) or 1.0
    share: collections.Counter = collections.Counter()
    docs: collections.Counter = collections.Counter()
    for prompt, weight in prompts:
        for phrase in _candidates(prompt):
            share[phrase] += weight
            docs[phrase] += 1
    scored = []
    for phrase, w in share.items():
        s = w / total
        if docs[phrase] < 2 or s < 0.15:
            continue
        lift = s / background.rate(phrase)
        if lift < 3:
            continue
        scored.append((s * math.log(lift), phrase, s, docs[phrase], lift))
    scored.sort(reverse=True)
    kept: List[Tuple] = []
    for item in scored:
        phrase, s = item[1], item[2]
        if any(phrase in other[1] and other[2] >= s * 0.8 for other in kept):
            continue                                        # a longer phrase already carries it
        kept = [o for o in kept if not (o[1] in phrase and s >= o[2] * 0.8)]
        kept.append(item)
        if len(kept) >= top:
            break
    return [{"phrase": p, "share": round(s, 2), "prompts": d, "lift": round(l, 1)} for _sc, p, s, d, l in kept]


# -- CSD weighting of local example prompts ---------------------------------------------------------------------------
class StyleWeights:
    """lift(image) = cos(image, LoRA signature) - cos(image, average LoRA signature). None when the style data is
    missing (no vectors, no cache): prompts then keep their order."""

    def __init__(self, connection: sqlite3.Connection):
        self.ok = False
        try:
            from .style import IMAGE_CACHE, MODEL_ID, ImageCache, unpack
        except Exception:
            return
        if not os.path.exists(IMAGE_CACHE):
            return
        self.unpack = unpack
        self.signatures: Dict[str, List[float]] = {}
        total: Optional[List[float]] = None
        for path, blob in connection.execute("SELECT path, vec FROM style_vectors WHERE model = ? AND vec IS NOT NULL",
                                             (MODEL_ID,)):
            vec = unpack(blob)
            self.signatures[path] = vec
            total = vec[:] if total is None else [a + b for a, b in zip(total, vec)]
        if not total:
            return
        norm = math.sqrt(sum(v * v for v in total)) or 1.0
        self.mean = [v / norm for v in total]
        self.cache = ImageCache()
        self.ok = True

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def lift(self, lora_path: str, image_path: str) -> Optional[float]:
        if not self.ok or lora_path not in self.signatures:
            return None
        from .style import _image_key
        try:
            key = _image_key(image_path)
        except OSError:
            return None
        blob = self.cache.get([key]).get(key)
        if not blob:
            return None
        vec = self.unpack(blob)
        return self._dot(vec, self.signatures[lora_path]) - self._dot(vec, self.mean)

    def close(self) -> None:
        if self.ok:
            self.cache.close()


# -- evidence ---------------------------------------------------------------------------------------------------------
LEVEL = {"None": 0, "Soft": 1, "Mature": 2, "X": 3}


def gather(connection: sqlite3.Connection, row: sqlite3.Row, background: Background,
           weights: StyleWeights) -> Dict[str, Any]:
    path = row["path"]
    folder = os.path.dirname(path)
    examples = []
    for prompt, media in connection.execute("SELECT prompt, media_file FROM examples WHERE path = ? AND prompt IS NOT "
                                            "NULL AND prompt <> ''", (path,)):
        lift = weights.lift(path, os.path.join(folder, media)) if media else None
        examples.append({"prompt": normalize(prompt)[:900], "lift": None if lift is None else round(lift, 3)})
    examples.sort(key=lambda e: -(e["lift"] if e["lift"] is not None else -1))
    community = []
    if row["civitai_version_id"]:
        for cleaned, prompt, source, reactions in connection.execute(
                "SELECT cleaned, prompt, source, coalesce(reactions, 0) FROM community_prompts WHERE version_id = ?",
                (row["civitai_version_id"],)):
            text = normalize(cleaned or prompt)
            if len(text.split()) >= 3:
                community.append((text, source, reactions))
    pool: List[Tuple[str, float]] = []
    seen = set()                                        # near-duplicates (same start) count once
    for e in examples:
        if e["prompt"][:150] not in seen:
            seen.add(e["prompt"][:150])
            pool.append((e["prompt"], 2.0 + max(0.0, (e["lift"] or 0.0)) * 10))
    for text, source, _reactions in community:
        if text[:150] not in seen:
            seen.add(text[:150])
            pool.append((text, 2.0 if source == "creator" else 1.0))
    phrases = mine_phrases(pool, background) if len(pool) >= 3 else []
    top_community = sorted({c for c in community}, key=lambda c: (c[1] != "creator", -c[2]))[:4]
    description = strip_html(row["description"] or "")
    hints = [line for line in re.split(r"(?<=[.!?])\s+|\n", description) if HINT.search(line) and len(line) < 400]
    return {
        "trained_words": [str(w) for w in _json_list(row["trained_words"])],
        "card_triggers": [t.get("text") for t in _json_list(row["triggers"]) if isinstance(t, dict) and t.get("text")],
        "description": description[:3000],
        "hint_lines": hints[:15],
        "examples": examples[:4],
        "community_top": [c[0][:600] for c in top_community],
        "phrases": phrases,
        "n_examples": len(examples),
        "n_community": len(community),
        "example_prompts": sorted({e["prompt"] for e in examples}),   # for evidence_sig (not stored)
    }


SIG_VERSION = "v2"


def evidence_sig(row: sqlite3.Row, ev: Dict[str, Any]) -> str:
    """This LoRA's own evidence only (2026-10-17): trained words, page, its example prompts, how many community
    prompts it has, and which phrases stand out. The phrase scores and the example style scores are left out: they are
    measured against the whole library (all Krea 2 prompts, the average style vector), drift whenever LoRAs or prompts
    are added, and made every refresh redo nearly every recipe."""
    parts = [row["trained_words"] or "", row["description"] or "", json.dumps(ev["example_prompts"]),
             str(ev["n_community"]), json.dumps(sorted(p["phrase"] for p in ev["phrases"]))]
    return SIG_VERSION + ":" + hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def same_as_stored(ev: Dict[str, Any], stored_evidence: str) -> bool:
    """A recipe saved with the old signature still matches this LoRA's own evidence now - so the switch to v2
    upgrades its signature instead of redoing it. Compares what the stored evidence kept: the community and example
    counts, the phrase list, and the top example prompts. (Trained words and page text are not in the stored
    evidence; a change there since the last recipe is not caught by this one-time upgrade.)"""
    try:
        old = json.loads(stored_evidence or "{}")
    except ValueError:
        return False
    if not old or old.get("n_community") != ev["n_community"] or old.get("n_examples") != ev["n_examples"]:
        return False
    if sorted(p.get("phrase") for p in old.get("phrases") or []) != sorted(p["phrase"] for p in ev["phrases"]):
        return False
    return {e.get("prompt") for e in old.get("examples") or []} <= set(ev["example_prompts"])


def evidence_text(row: sqlite3.Row, ev: Dict[str, Any]) -> str:
    lines = [f"LoRA: {row['model_name'] or row['stem']}" + (f" - {row['version_name']}" if row["version_name"] else ""),
             f"File: {row['file_name']}   Folder: {row['rel_dir']}"]
    lines.append("Trained words (Civitai): " + (" || ".join(ev["trained_words"]) or "(none)"))
    if ev["hint_lines"]:
        lines.append("Page lines that mention prompting or settings:\n" + "\n".join("  - " + h for h in ev["hint_lines"]))
    if ev["description"]:
        lines.append("Model page description:\n" + ev["description"][:2200])
    if ev["examples"]:
        lines.append("Creator example prompts (style match to the LoRA's look in brackets, higher = more typical):")
        for e in ev["examples"]:
            lines.append(f"  [{e['lift'] if e['lift'] is not None else '?'}] {e['prompt'][:700]}")
    if ev["community_top"]:
        lines.append(f"Some prompts people used with it ({ev['n_community']} collected):")
        lines.extend("  - " + p for p in ev["community_top"])
    if ev["phrases"]:
        lines.append("Phrases that recur in its prompts but are rare elsewhere (share of its prompts, x more than usual):")
        lines.extend(f"  - {p['phrase']}  ({p['share']:.0%}, x{p['lift']})" for p in ev["phrases"])
    if row["effect"]:
        lines.append("Our captioner's card text (from the preview images; a last resort for the style phrase): "
                     + str(row["effect"]))
    return "\n".join(lines)


# -- variants -----------------------------------------------------------------------------------------------------------
def _clean_bits(texts: Iterable[str]) -> List[str]:
    """Comma pieces, quality tags out, and a piece dropped when its words are (nearly) all in an earlier piece."""
    out: List[str] = []
    kept_words: List[set] = []
    for text in texts:
        for bit in SPLIT.split(MEDIUM_LABEL.sub("", normalize(text))):
            bit = bit.strip(" -:")
            if not bit or bit in QUALITY:
                continue
            words = set(WORD.findall(bit))
            if not words or any(len(words & k) >= 0.8 * len(words) for k in kept_words):
                continue
            kept_words.append(words)
            out.append(bit)
    return out


def in_evidence(token: str, ev: Dict[str, Any]) -> bool:
    """The trigger the model gave really occurs in the trained words, the page or the prompts."""
    token = normalize(token).lstrip("@")
    if not token:
        return False
    hay = " ".join([normalize(" ".join(ev["trained_words"])), normalize(ev["description"]),
                    " ".join(e["prompt"] for e in ev["examples"]), " ".join(ev["community_top"]),
                    " ".join(p["phrase"] for p in ev["phrases"])])
    return token in hay


def frequent_token(ev: Dict[str, Any]) -> str:
    """A made-up word (digits, _ or @, or no vowels pattern aside) in 60%+ of the LoRA's prompts: its de-facto trigger."""
    for p in ev["phrases"]:
        words = p["phrase"].split()
        if p["share"] >= 0.6 and p["lift"] >= 20 and len(words) <= 3 and re.search(r"[0-9_@]", p["phrase"]):
            return p["phrase"]
    return ""


def _cap(bits: List[str], words: int) -> str:
    out, n = [], 0
    for bit in bits:
        size = len(bit.split())
        if n + size > words and out:
            break
        out.append(bit)
        n += size
    return ", ".join(out)


def check_trigger(recipe: Dict[str, Any], ev: Dict[str, Any]) -> None:
    """Blank an invented trigger; use a de-facto one (in most prompts) when the model gave none."""
    trigger = recipe.get("trigger") or ""
    if trigger and not in_evidence(trigger, ev):
        recipe["note"] = (recipe.get("note") or "") + f" [trigger '{trigger}' not found in the evidence - dropped]"
        recipe["trigger"] = trigger = ""
    if not trigger:
        found = frequent_token(ev)
        if found:
            recipe["trigger"] = found
            recipe["note"] = (recipe.get("note") or "") + f" [trigger '{found}' taken from its prompts]"


CARD_VERB = re.compile(r"^(applies|adds|gives|creates|produces|renders|transforms|turns|makes)\s+(an?\s+|the\s+)?",
                       re.I)


LOOK_KINDS = {"style_medium", "photo_look", "lighting_colour"}     # cards whose effect text describes a look


def fallback_phrase(recipe: Dict[str, Any], effect: str, kind: str = "") -> bool:
    """No style phrase (the page and prompts describe no look): use the card's effect text, first sentence, up to
    PHRASE_MAX_WORDS words. The Lox trigger test (2026-10-14) showed a recipe without a phrase stays flat on
    fine-tunes. Only for look cards (LOOK_KINDS): for a character, pose, slider or speed LoRA the card text says
    what the LoRA does, which is not prompt text. True when the phrase was filled."""
    if str(kind or "") not in LOOK_KINDS:
        return False
    if str(recipe.get("style_phrase") or "").strip() or not str(effect or "").strip():
        return False
    first = re.split(r"(?<=[.!?])\s+", str(effect).strip())[0]
    words = CARD_VERB.sub("", first).rstrip(".").split()
    recipe["style_phrase"] = " ".join(words[:PHRASE_MAX_WORDS])
    recipe["phrase_source"] = "card"
    recipe["note"] = (recipe.get("note") or "") + " [style phrase from the card text - the page describes no look]"
    return True


def refill_phrases(settings: Settings, log=print) -> int:
    """Stored recipes without a style phrase get the card fallback and rebuilt variants (no LLM call)."""
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    rows = connection.execute("SELECT r.path, r.recipe, l.trained_words, c.effect, c.kind FROM trigger_recipes r "
                              "JOIN loras l ON l.path = r.path LEFT JOIN cards c ON c.path = r.path").fetchall()
    done = 0
    for row in rows:
        recipe = json.loads(row["recipe"])
        if recipe.get("phrase_source") == "card" and str(row["kind"] or "") not in LOOK_KINDS:
            # undo an earlier fill on a card that is not a look (2026-10-15: character / pose / speed LoRAs)
            recipe["style_phrase"] = ""
            recipe.pop("phrase_source", None)
            recipe["note"] = str(recipe.get("note") or "").replace(
                " [style phrase from the card text - the page describes no look]", "")
            log(f"[triggers] phrase removed (not a look card): {os.path.basename(row['path'])}")
        elif not fallback_phrase(recipe, row["effect"], row["kind"]):
            continue
        from . import recipe_edits                  # your corrections stay on top (2026-10-16)
        variants = recipe_edits.apply(connection, row["path"], recipe,
                                      [str(w) for w in _json_list(row["trained_words"])])
        connection.execute("UPDATE trigger_recipes SET recipe = ?, variants = ?, updated_at = ? WHERE path = ?",
                           (json.dumps(recipe, ensure_ascii=False), json.dumps(variants, ensure_ascii=False),
                            time.strftime("%Y-%m-%d %H:%M:%S"), row["path"]))
        done += 1
        if recipe.get("phrase_source") == "card":
            log(f"[triggers] phrase from card: {os.path.basename(row['path'])}: {recipe['style_phrase'][:90]}")
    connection.commit()
    connection.close()
    return done


def build_variants(recipe: Dict[str, Any], trained_words: List[str]) -> List[Dict[str, Any]]:
    trigger = normalize(recipe.get("trigger") or "")
    phrase_bits = [b for b in _clean_bits([trigger, recipe.get("style_phrase") or ""]) if b != trigger]
    phrase = _cap(phrase_bits, PHRASE_MAX_WORDS)
    official = ", ".join(_clean_bits(recipe.get("trained_words_fixed") or trained_words))
    extra = [b for b in _clean_bits(recipe.get("extra_phrases") or []) if b != trigger]
    raw = {
        "none": "",
        "trigger": trigger,
        "official": official,
        "trigger+phrase": ", ".join(b for b in [trigger, phrase] if b),
        "full": _cap(_clean_bits([official or trigger, trigger, phrase] + extra), FULL_MAX_WORDS),
    }
    variants: List[Dict[str, Any]] = []
    by_text: Dict[str, Dict[str, Any]] = {}
    for name in VARIANT_ORDER:
        text = raw[name]
        if name != "none" and not text:
            continue
        key = re.sub(r"[\s,]+", " ", text).strip()
        if key in by_text:
            by_text[key]["covers"].append(name)
            continue
        item = {"name": name, "text": text, "covers": []}
        by_text[key] = item
        variants.append(item)
    return variants


def default_text(variants: List[Dict[str, Any]]) -> str:
    """The prompt text an untested LoRA uses: trigger+phrase (or whichever variant absorbed it), else the trigger."""
    for wanted in (DEFAULT_VARIANT, "trigger"):
        for v in variants:
            if v["name"] == wanted or wanted in v["covers"]:
                return v["text"]
    return ""


# -- running ------------------------------------------------------------------------------------------------------------
ROW_SQL = """SELECT l.path, l.stem, l.file_name, l.model_name, l.version_name, l.rel_dir, l.trained_words, l.description,
                    l.civitai_version_id, c.triggers, c.effect, c.medium, c.kind
             FROM loras l LEFT JOIN cards c ON c.path = l.path"""


def pilot_paths(connection: sqlite3.Connection, arch: str = "Krea2") -> List[str]:
    paths: List[str] = []
    for name in PILOT_NAMES:
        for (path,) in connection.execute("SELECT path FROM loras WHERE arch_folder = ? AND lower(file_name) LIKE ? "
                                          "AND coalesce(lm_exclude, 0) = 0", (arch, f"%{name}%")):
            paths.append(path)
    rows = connection.execute(
        "SELECT l.path, count(cp.image_id) n FROM style_vectors s JOIN loras l ON l.path = s.path "
        "JOIN community_prompts cp ON cp.version_id = l.civitai_version_id "
        "WHERE s.render_verdict = 'unclear' AND l.arch_folder = ? AND coalesce(l.lm_exclude, 0) = 0 "
        "AND lower(coalesce(l.model_name, '') || l.file_name) NOT LIKE '%slider%' "
        "GROUP BY l.path ORDER BY n DESC, l.path", (arch,)).fetchall()
    for path, _n in rows:
        if path not in paths:
            paths.append(path)
        if len(paths) >= len(PILOT_NAMES) + PILOT_UNCLEAR:
            break
    return paths


def select_rows(connection: sqlite3.Connection, names: str = "", only: str = "", pilot: bool = False,
                arch: str = "Krea2") -> List[sqlite3.Row]:
    if pilot:
        paths = pilot_paths(connection, arch)
        marks = ",".join("?" * len(paths))
        rows = connection.execute(f"{ROW_SQL} WHERE l.path IN ({marks})", paths).fetchall()
        order = {p: i for i, p in enumerate(paths)}
        return sorted(rows, key=lambda r: order[r["path"]])
    where, params = ["l.arch_folder = ?", "coalesce(l.lm_exclude, 0) = 0"], [arch]
    if only:
        where.append("replace(l.rel_dir, '\\', '/') LIKE ?")
        params.append(only.replace("\\", "/").strip("/") + "%")
    if names:
        ors = []
        for name in [n.strip().lower() for n in names.split(",") if n.strip()]:
            ors.append("(lower(l.file_name) LIKE ? OR lower(coalesce(l.model_name, '')) LIKE ?)")
            params += [f"%{name}%", f"%{name}%"]
        where.append("(" + " OR ".join(ors) + ")")
    return connection.execute(f"{ROW_SQL} WHERE {' AND '.join(where)} ORDER BY l.rel_dir, l.file_name",
                              params).fetchall()


def ask(settings: Settings, model: str, text: str) -> Dict[str, Any]:
    from .llm import chat_json
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return chat_json(settings, model, messages, schema(), temperature=0.0, max_tokens=3000, timeout=300)


def own_sig(row: sqlite3.Row) -> str:
    """The LoRA's own page: trained words + description. What makes the refresh step redo a recipe (2026-10-20)."""
    return hashlib.sha1(((row["trained_words"] or "") + "\n" + (row["description"] or "")).encode("utf-8")).hexdigest()


FAILURE_KEY = "trigger_failures"
MAX_FAILURES = 2


def _failures(connection: sqlite3.Connection) -> Dict[str, int]:
    try:
        row = connection.execute("SELECT v FROM meta WHERE k = ?", (FAILURE_KEY,)).fetchone()
        return json.loads(row[0]) if row and row[0] else {}
    except (sqlite3.Error, ValueError):
        return {}


def _save_failures(connection: sqlite3.Connection, failures: Dict[str, int]) -> None:
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        connection.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (FAILURE_KEY, json.dumps(failures)))
    except sqlite3.Error:
        pass


def run(settings: Settings, names: str = "", only: str = "", pilot: bool = False, limit: int = 0, force: bool = False,
        dry_run: bool = False, model: str = "", arch: str = "Krea2", log=print, new_only: bool = False) -> Dict[str, Any]:
    """Recipes for one shelf, a comma list of shelves, or '*' (every shelf). Each shelf has its own background corpus
    (what is common in that base model's prompts), so shelves run one after another."""
    shelves = [a.strip() for a in str(arch or "").split(",") if a.strip()]
    if not shelves or "*" in shelves:
        connection = sqlite3.connect(settings.catalog_path, timeout=30)
        shelves = [r[0] for r in connection.execute("SELECT DISTINCT arch_folder FROM loras WHERE arch_folder IS NOT NULL "
                                                     "AND arch_folder <> '' ORDER BY 1")]
        connection.close()
    total: collections.Counter = collections.Counter()
    for shelf in shelves:
        total.update(_run_shelf(settings, names=names, only=only, pilot=pilot, limit=limit, force=force,
                                dry_run=dry_run, model=model, arch=shelf, log=log, new_only=new_only))
    return dict(total)


def _run_shelf(settings: Settings, names: str = "", only: str = "", pilot: bool = False, limit: int = 0,
               force: bool = False, dry_run: bool = False, model: str = "", arch: str = "Krea2",
               log=print, new_only: bool = False) -> Dict[str, Any]:
    model = model or getattr(settings, "genre_model", "") or settings.card_model
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    rows = select_rows(connection, names=names, only=only, pilot=pilot, arch=arch)
    known = {r["path"]: (r["evidence_sig"], r["evidence"])
             for r in connection.execute("SELECT path, evidence_sig, evidence FROM trigger_recipes")}
    started = time.time()
    background = Background(connection, arch)
    weights = StyleWeights(connection)
    log(f"[triggers] {arch}: {len(rows)} LoRAs with {model}; background {background.n} prompts; "
        f"style weights {'on' if weights.ok else 'off'} ({time.time() - started:.0f}s)")
    counts = collections.Counter()
    failures = _failures(connection)
    retry_named = bool(names)                       # naming a LoRA always retries it
    for number, row in enumerate(rows, 1):
        if limit and counts["written"] + counts["shown"] >= limit:
            break
        if new_only and not force:
            # the refresh step: only LoRAs without a recipe, or whose own trained words / description changed
            old_sig, old_ev = known.get(row["path"], (None, None))
            if old_sig is not None:
                try:
                    stored = json.loads(old_ev or "{}")
                except ValueError:
                    stored = {}
                mine = own_sig(row)
                if not stored.get("own_sig"):        # recipes from before 2026-10-20: take their page as it is now
                    stored["own_sig"] = mine
                    connection.execute("UPDATE trigger_recipes SET evidence = ? WHERE path = ?",
                                       (json.dumps(stored, ensure_ascii=False), row["path"]))
                if stored["own_sig"] == mine:
                    counts["unchanged"] += 1
                    continue
            elif failures.get(row["path"], 0) >= MAX_FAILURES and not retry_named:
                counts["skipped_failed"] += 1
                continue
        ev = gather(connection, row, background, weights)
        sig = evidence_sig(row, ev)
        old_sig, old_ev = known.get(row["path"], (None, None))
        if not force and old_sig == sig:
            counts["unchanged"] += 1
            continue
        if not force and old_sig and not str(old_sig).startswith(SIG_VERSION + ":") and same_as_stored(ev, old_ev):
            # saved under the old signature and nothing of its own changed: new signature, no LLM call
            connection.execute("UPDATE trigger_recipes SET evidence_sig = ? WHERE path = ?", (sig, row["path"]))
            counts["unchanged"] += 1
            counts["sig_upgraded"] += 1
            continue
        try:
            recipe = ask(settings, model, evidence_text(row, ev))
        except Exception as error:
            counts["failed"] += 1
            failures[row["path"]] = failures.get(row["path"], 0) + 1
            _save_failures(connection, failures)
            connection.commit()
            log(f"[triggers] failed {row['stem']}: {str(error)[:160]}"
                + (" - the refresh will not try it again" if failures[row["path"]] >= MAX_FAILURES else ""))
            continue
        check_trigger(recipe, ev)
        fallback_phrase(recipe, row["kind"] and row["effect"], row["kind"])
        from . import recipe_edits                  # your corrections stay on top (2026-10-16)
        variants = recipe_edits.apply(connection, row["path"], recipe, ev["trained_words"])
        saved_ev = {k: ev[k] for k in ("phrases", "examples", "hint_lines", "n_examples", "n_community")}
        saved_ev["own_sig"] = own_sig(row)
        if failures.pop(row["path"], None) is not None:
            _save_failures(connection, failures)
        if dry_run:
            counts["shown"] += 1
            log(report_one(row["model_name"] or row["stem"], recipe, variants, saved_ev))
            continue
        connection.execute("INSERT OR REPLACE INTO trigger_recipes (path, model, recipe, variants, evidence, "
                           "evidence_sig, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (row["path"], model, json.dumps(recipe, ensure_ascii=False),
                            json.dumps(variants, ensure_ascii=False), json.dumps(saved_ev, ensure_ascii=False), sig,
                            time.strftime("%Y-%m-%d %H:%M:%S")))
        connection.commit()
        counts["written"] += 1
        log(f"[triggers] {number}/{len(rows)} {str(row['model_name'] or row['stem'])[:50]}: "
            f"{default_text(variants)[:110]}")
    connection.commit()                             # signature upgrades / own_sig fills after the last written recipe
    weights.close()
    connection.close()
    counts["seconds"] = int(time.time() - started)
    return dict(counts)


def report_one(name: str, recipe: Dict[str, Any], variants: List[Dict[str, Any]], ev: Dict[str, Any]) -> str:
    lines = [f"== {name}", f"   needs trigger: {recipe.get('needs_trigger')}   note: {recipe.get('note') or '-'}"]
    if recipe.get("page_advice"):
        lines.append(f"   page: {recipe['page_advice']}")
    if recipe.get("setting_hints"):
        lines.append("   settings: " + " | ".join(recipe["setting_hints"]))
    if recipe.get("avoid"):
        lines.append("   avoid: " + " | ".join(recipe["avoid"]))
    if ev.get("phrases"):
        lines.append("   mined: " + "; ".join(f"{p['phrase']} {p['share']:.0%}" for p in ev["phrases"][:8]))
    for v in variants:
        also = f" (= {', '.join(v['covers'])})" if v["covers"] else ""
        lines.append(f"   {v['name']:15s}{also}: {v['text'] or '(nothing)'}")
    return "\n".join(lines)


# -- the trigger-test sweep ---------------------------------------------------------------------------------------
# One cell per distinct variant (LoRA at its card weight, node triggers off, the variant text written into the prompt),
# one "words only" cell per LoRA (the fullest variant's words, no LoRA) and one plain cell (subject only, no LoRA).
# Scoring (check_test): gain(v) = CSD match to the LoRA's look minus the plain cell's; words_only says how much of
# that the words alone do. Suggested pick = the shortest variant within 90% of the best gain.
DEFAULT_SUBJECT = ("A woman in her thirties sits on a wooden chair beside a window, seen from the knees up. She wears a "
                   "plain knit sweater and dark trousers, hands resting in her lap, looking toward the viewer. A bare "
                   "wall behind her and a small potted plant on the windowsill.")
TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trigger_tests")
PICK_SHARE = 0.90
TEST_MP = 1.0


def _card_weight(connection: sqlite3.Connection, path: str) -> float:
    try:
        row = connection.execute("SELECT weight_typical FROM card_facets WHERE path = ?", (path,)).fetchone()
    except sqlite3.Error:
        row = None
    try:
        return round(float(row[0]), 3) if row and row[0] is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _join(words: str, subject: str) -> str:
    return f"{words.strip().rstrip(',.')}. {subject}" if words.strip() else subject


def plan_test(settings: Settings, names: str = "", pilot: bool = False, untested: bool = False, limit: int = 0,
              subject: str = "", megapixels: float = TEST_MP, name: str = "", arch: str = "Krea2",
              write: bool = True, log=print) -> Dict[str, Any]:
    """Build the combos for a trigger test and (with write) save it as a SweepPlan preset plus a plan file the check
    reads. LoRAs without a recipe are skipped (run.py triggers first)."""
    subject = subject.strip() or DEFAULT_SUBJECT
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    rows = select_rows(connection, names=names, pilot=pilot, arch=arch)
    tested = set()
    if untested:
        for (path,) in connection.execute("SELECT path FROM card_edits WHERE trigger_tested IS NOT NULL "
                                          "AND trigger_tested <> ''") if _has_column(connection, "card_edits",
                                                                                   "trigger_tested") else []:
            tested.add(path)
    # no private keys in the cells: the sweep engine hands every key to the render and drops '_' keys from the
    # manifest, so the check tells cells apart by prompt (plain = the subject alone; words-only = a baseline cell
    # whose prompt is one of the LoRA's variant prompts)
    combos: List[Dict[str, Any]] = [{"prompt": subject, "lora_baseline": True, "s1_megapixels": megapixels}]
    word_prompts = set()
    loras: List[Dict[str, Any]] = []
    for row in rows:
        if row["path"] in tested:
            continue
        stored = connection.execute("SELECT variants FROM trigger_recipes WHERE path = ?", (row["path"],)).fetchone()
        if not stored:
            log(f"[triggers] no recipe for {row['stem']} - skipped (run.py triggers first)")
            continue
        variants = json.loads(stored["variants"])
        weight = _card_weight(connection, row["path"])
        richest = max(variants, key=lambda v: len(v["text"].split()))["text"]
        for v in variants:
            combos.append({"prompt": _join(v["text"], subject), "lora_file": row["path"], "lora_strength": weight,
                           "s1_megapixels": megapixels})
        if richest and _join(richest, subject) not in word_prompts:     # LoRAs with the same words share one
            word_prompts.add(_join(richest, subject))
            combos.append({"prompt": _join(richest, subject), "lora_baseline": True, "s1_megapixels": megapixels})
        loras.append({"path": row["path"], "name": row["model_name"] or row["stem"], "weight": weight,
                      "variants": variants, "words_only": richest})
        if limit and len(loras) >= limit:
            break
    connection.close()
    name = name or f"Trigger test - {len(loras)} LoRAs {time.strftime('%Y-%m-%d %H%M')}"
    result = {"name": name, "subject": subject, "megapixels": megapixels, "loras": loras, "combos": combos}
    log(f"[triggers] {name}: {len(loras)} LoRAs, {len(combos)} cells "
        f"({sum(len(l['variants']) for l in loras)} variant cells, {sum(1 for l in loras if l['words_only'])} "
        f"words-only, 1 plain)")
    if write and loras:
        from .sweepplan import GRID_FIELDS_BLANK, write_preset
        sweep = {field: "" for field in GRID_FIELDS_BLANK}
        sweep.update({
            "mode": "combos", "stage_scope": "s1", "lora_mode": "none", "lora_files": "", "lora_strengths": "card",
            "include_baseline": False, "keep_panel_stack": False, "lora_triggers": "off", "prompts": "",
            "combos_json": "\n".join(json.dumps(c, ensure_ascii=False) for c in combos), "combos_file": "",
            "seed_policy": "fixed", "seeds": "", "quick_screen": False, "score_metrics": True,
            "sort_by": "plan_order", "max_combos": max(64, len(combos)), "megapixels": "",
            "preset_notes": (f"{name}: trigger recipes (run.py triggers). Per LoRA one cell per prompt variant at the "
                             f"card weight (node triggers OFF - the words are in each cell's prompt), one words-only "
                             f"cell (fullest variant, no LoRA), plus one plain no-LoRA cell. {megapixels:g} MP. "
                             f"Score it with: python run.py triggers --check <sweep folder>")})
        result["backup"] = write_preset(settings, {"name": name, "preset": {"sweep": sweep}})
        os.makedirs(TEST_DIR, exist_ok=True)
        plan_file = os.path.join(TEST_DIR, re.sub(r"[^\w\- ]+", "_", name) + ".json")
        with open(plan_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
        result["plan_file"] = plan_file
        log(f"[triggers] preset written ('{name}'), plan saved to {plan_file}")
    return result


def _has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in connection.execute(f"PRAGMA table_info({table})"))


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(str(path or "")))


def check_test(settings: Settings, sweep_dir: str, log=print) -> Dict[str, Any]:
    """Score a finished trigger-test sweep with CSD and store the suggestion per LoRA (trigger_tests table)."""
    from . import style as S
    reason = S.available(settings)
    if reason:
        raise ValueError(reason)
    manifest, cells = S._sweep_cells(sweep_dir)
    sweep_id = manifest.get("sweep_id") or os.path.basename(os.path.normpath(sweep_dir))
    baselines, renders = {}, collections.defaultdict(dict)     # baselines[prompt] = image; renders[path][prompt]
    for cell in cells:
        over = cell.get("overrides") or {}
        image = os.path.join(sweep_dir, cell["file"])
        if not os.path.isfile(image):
            continue
        prompt = str(over.get("prompt") or "")
        if over.get("lora_file"):
            renders[_norm_path(over["lora_file"])][prompt] = image
        elif over.get("lora_baseline"):
            baselines.setdefault(prompt, image)
    # plain = the baseline prompt every other baseline prompt ends with (the subject alone)
    plain_prompt = next((p for p in sorted(baselines, key=len) if all(q.endswith(p) for q in baselines)), "")
    plain = baselines.get(plain_prompt)
    words = {}                                                    # words[norm path] = words-only image
    for norm, prompts in renders.items():
        hits = [baselines[p] for p in prompts if p in baselines and p != plain_prompt]
        if hits:
            words[norm] = hits[0]
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    connection.execute("""CREATE TABLE IF NOT EXISTS trigger_tests (path TEXT, sweep TEXT, variant TEXT, text TEXT,
                          match REAL, gain REAL, words_gain REAL, suggested INTEGER, checked_at TEXT,
                          PRIMARY KEY (path, sweep, variant))""")
    index = S.StyleIndex(settings)
    index.ensure()
    images = sorted({im for d in renders.values() for im in d.values()} | set(baselines.values()))
    cache = S.ImageCache()
    keys = {p: S._image_key(p) for p in images}
    have = cache.get(list(keys.values()))
    need = [p for p in images if keys[p] not in have]
    if need:
        log(f"[triggers] embedding {len(need)} renders")
        model = S.StyleModel(settings)
        tensors = [model.load_file(p) for p in need]
        good = [(p, t) for p, t in zip(need, tensors) if t is not None]
        items = [(keys[p], S.pack(v)) for (p, _t), v in zip(good, model.embed_tensors([t for _p, t in good]))]
        cache.put(items)
        have.update(items)
        model.close()
    cache.close()
    vec = {p: S.unpack(have[k]) for p, k in keys.items() if k in have}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    report, skipped = [], {}
    by_norm = {_norm_path(p): p for p in index.by_path}
    for norm, prompts in renders.items():
        path = by_norm.get(norm)
        if path is None:
            skipped[norm] = "no style vector (no previews) or not in the catalog"
            continue
        stored = connection.execute("SELECT variants FROM trigger_recipes WHERE path = ?", (path,)).fetchone()
        variants = json.loads(stored["variants"]) if stored else []
        sig = index.items[index.by_path[path]]["vec"]
        base = S._dot(vec[plain], sig) if plain in vec else None
        w_gain = (S._dot(vec[words[norm]], sig) - base) if base is not None and words.get(norm) in vec else None
        rows = []
        for prompt, image in prompts.items():
            if image not in vec:
                continue
            if prompt == plain_prompt:
                variant = next((v for v in variants if not v["text"]), None)
            else:
                variant = next((v for v in variants if v["text"]
                                and prompt.startswith(v["text"].strip().rstrip(",.") + ".")), None)
            if variant is None:
                continue
            match = S._dot(vec[image], sig)
            rows.append({"variant": variant["name"], "covers": variant["covers"], "text": variant["text"],
                         "match": match, "gain": None if base is None else match - base})
        if not rows:
            skipped[path] = "no variant cells matched the recipe (recipe changed since the plan?)"
            continue
        best = max(r["gain"] if r["gain"] is not None else r["match"] for r in rows)
        key = (lambda r: r["gain"] if r["gain"] is not None else r["match"])
        good = [r for r in rows if best > 0 and key(r) >= PICK_SHARE * best] or [max(rows, key=key)]
        pick = min(good, key=lambda r: (len(r["text"].split()), -key(r)))
        for r in rows:
            r["suggested"] = r is pick
            connection.execute("INSERT OR REPLACE INTO trigger_tests VALUES (?,?,?,?,?,?,?,?,?)",
                               (path, sweep_id, r["variant"], r["text"], round(r["match"], 4),
                                None if r["gain"] is None else round(r["gain"], 4),
                                None if w_gain is None else round(w_gain, 4), int(r["suggested"]), now))
        name = index.items[index.by_path[path]]["name"]
        report.append({"path": path, "name": name, "rows": sorted(rows, key=lambda r: VARIANT_ORDER.index(r["variant"])),
                       "words_gain": w_gain, "pick": pick["variant"]})
    # the suggestion becomes this checkpoint's 'test' pick, and the cards pick it up (2026-10-15)
    from . import facets, trigger_picks
    trigger_picks.sync_from_tests(connection, settings.sweeps_dir, sweep_id)
    connection.commit()
    for item in report:
        facets.refresh_one(connection, item["path"])
    connection.commit()
    connection.close()
    return {"sweep": sweep_id, "loras": report, "skipped": skipped, "plain": bool(plain)}


def sync_picks(settings: Settings, log=print) -> int:
    """Older checks -> 'test' picks per checkpoint, then refresh the cards that have picks."""
    from . import facets, trigger_picks
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    facets.ensure(connection)
    n = trigger_picks.sync_from_tests(connection, settings.sweeps_dir)
    connection.commit()
    paths = [r[0] for r in connection.execute("select distinct path from trigger_picks")]
    for path in paths:
        facets.refresh_one(connection, path)
    connection.commit()
    connection.close()
    log(f"[triggers] {n} test suggestion(s) synced; {len(paths)} card(s) refreshed")
    return n


def check_report(result: Dict[str, Any]) -> str:
    lines = [f"{result['sweep']}: {len(result['loras'])} LoRAs"
             + ("" if result["plain"] else "  (no plain cell - gains not available, raw matches shown)"),
             "  gain = CSD match to the LoRA's own look minus the plain no-LoRA render; words = the same gain from the "
             "fullest words WITHOUT the LoRA.  * = suggested (shortest within 90% of the best)", ""]
    for item in result["loras"]:
        wg = "n/a" if item["words_gain"] is None else f"{item['words_gain']:+.3f}"
        lines.append(f"== {item['name']}   (words only: {wg})")
        for r in item["rows"]:
            val = f"gain {r['gain']:+.3f}" if r["gain"] is not None else f"match {r['match']:.3f}"
            also = f" (= {', '.join(r['covers'])})" if r["covers"] else ""
            lines.append(f"   {'*' if r['suggested'] else ' '} {r['variant']:15s}{also:28s} {val}   {r['text'][:80]}")
    for name, why in result["skipped"].items():
        lines.append(f"  skipped {name}: {why}")
    return "\n".join(lines)


def show(settings: Settings, names: str = "", pilot: bool = False, arch: str = "Krea2") -> str:
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    rows = select_rows(connection, names=names, pilot=pilot, arch=arch)
    out = []
    for row in rows:
        stored = connection.execute("SELECT recipe, variants, evidence FROM trigger_recipes WHERE path = ?",
                                    (row["path"],)).fetchone()
        if stored:
            out.append(report_one(row["model_name"] or row["stem"], json.loads(stored["recipe"]),
                                  json.loads(stored["variants"]), json.loads(stored["evidence"] or "{}")))
        else:
            out.append(f"== {row['model_name'] or row['stem']}: no recipe yet")
    connection.close()
    return "\n".join(out)
