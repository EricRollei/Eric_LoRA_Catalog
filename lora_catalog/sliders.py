"""
Slider cards: a second, text-only pass for slider LoRAs. The general card pass (cards.py) reads sliders badly - a
vision model looks at one example picture and describes it ("adds a slider control panel"), and has no idea that a
slider is a dial with two ends and a signed range. This pass asks a text model the slider questions only, from the
evidence that actually states them: the Civitai description, trained words, tags, and the weights people used in
gallery prompts (including <lora:name:-3> tokens).

    slider_cards
        path            the LoRA
        is_slider       1 when the evidence shows it is a dial (0 rows are kept so a re-run skips them)
        axis            an axis id from axes.yaml, or a free word when none fits (`run.py sliders --gaps` lists them)
        positive_toward which end of the axis a POSITIVE weight moves toward (one of the axis' two ends, or free text)
        positive_pole   what a positive weight does, in a few visual words
        negative_pole   what a negative weight does ('' = one-sided: only positive weights mean anything)
        range_low, range_high      the extremes the author or the examples use
        useful_low, useful_high    where results stay clean
        stages          early | late | all
        confidence      0-1
        evidence        JSON: what was relied on, the example weights seen
        refined_by, refined_at

Your edits win: the card editor's weight low / high (card_edits) replace range and useful range in card_facets.

    python run.py sliders                        every slider without a slider card
    python run.py sliders --only Krea2 --force   redo one shelf
    python run.py sliders --gaps                 axes the cards name that axes.yaml does not have

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .config import Settings, user_file
from .store import Catalog

AXES_PATH = user_file("axes.yaml")               # your copy in the data folder (the shipped one is the start)
SLIDER_NAME = re.compile(r"slider", re.I)
LORA_TOKEN = re.compile(r"<lora:([^:>]+):(-?\d+(?:\.\d+)?)", re.I)
MAX_DESCRIPTION = 3500

TABLE = """
CREATE TABLE IF NOT EXISTS slider_cards (
    path TEXT PRIMARY KEY, is_slider INTEGER, axis TEXT, positive_toward TEXT, positive_pole TEXT, negative_pole TEXT,
    range_low REAL, range_high REAL, useful_low REAL, useful_high REAL, stages TEXT, confidence REAL,
    evidence TEXT, refined_by TEXT, refined_at TEXT
);
"""


def ensure(connection) -> None:
    connection.executescript(TABLE)


def load_axes(path: str = AXES_PATH) -> Dict[str, Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return (yaml.safe_load(handle) or {}).get("axes") or {}
    except OSError:
        return {}


def is_slider_row(row: Any) -> bool:
    return bool(SLIDER_NAME.search(" ".join(str(row[key] or "") for key in ("file_name", "rel_dir", "model_name"))))


def _plain(text: str) -> str:
    """Civitai descriptions are HTML: tags out, entities decoded, whitespace folded."""
    text = re.sub(r"<(br|/p|/li|/h\d)\s*/?>", "\n", text or "", flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def _json_list(value: Any) -> List[Any]:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def example_weights(row: Any, examples: List[Any]) -> List[float]:
    """Weights this LoRA was used at: the examples table, plus <lora:...:w> tokens naming this file in the prompts."""
    stem = (row["stem"] or "").lower()
    found: List[float] = []
    for example in examples:
        if example["weight"] is not None:
            found.append(float(example["weight"]))
        for name, weight in LORA_TOKEN.findall(example["prompt"] or ""):
            if stem and (name.lower() in stem or stem in name.lower()):
                found.append(float(weight))
    return found


def axes_text(axes: Dict[str, Dict[str, Any]]) -> str:
    return "\n".join(f"  {axis}: {' <-> '.join(spec.get('ends') or [])}" for axis, spec in axes.items())


SYSTEM = """You catalogue SLIDER LoRAs for an image model. A slider is a dial: one LoRA trained so that a positive
weight pushes the picture toward one end of an attribute and a negative weight toward the other (older <-> younger,
brighter <-> darker, wider shot <-> closer shot). Some are one-sided (only positive weights do anything useful).
Weights are often far outside 0..1 (-5 .. +5, sometimes -15 .. +15).

You get the LoRA's name, Civitai tags, trigger words, description and the weights people actually used. Work out:
- is_slider: false if the evidence shows it is not a dial at all (an ordinary style or concept LoRA that happens to
  have 'slider' in its name). Then fill the rest with empty values.
- axis: the attribute it moves. Use one of the KNOWN AXES ids when one fits; otherwise a short snake_case id.
- positive_toward: which end a POSITIVE weight moves toward. For a known axis copy one of its two end names exactly.
  Read the description carefully: some authors trained it backwards (positive = younger, negative = more detail).
- positive_pole / negative_pole: what a positive / a negative weight does, in a few visual words. negative_pole ''
  when it is one-sided.
- range_low, range_high: the extremes the author recommends or the examples use. If the description gives none and
  there are no example weights, use -3 and 3 for a two-sided slider, 0 and 3 for a one-sided one.
- useful_low, useful_high: where results stay clean (the author's recommended range when given; otherwise about two
  thirds of the range).
- stages: early when it changes shape, body, framing, age or composition; late when it changes surface, detail,
  colour, texture or skin; all when it changes overall light or realism.
- confidence 0-1: 0.9 when the description states the direction and range; 0.6 when you infer from name and
  examples; 0.4 or less from the name alone.
- evidence_used: a few words.
Describe only what the evidence supports. Never describe an example image's contents as the effect."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "slider", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "is_slider": {"type": "boolean"},
            "axis": {"type": "string"},
            "positive_toward": {"type": "string"},
            "positive_pole": {"type": "string"},
            "negative_pole": {"type": "string"},
            "range_low": {"type": "number"}, "range_high": {"type": "number"},
            "useful_low": {"type": "number"}, "useful_high": {"type": "number"},
            "stages": {"type": "string", "enum": ["early", "late", "all"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_used": {"type": "string"},
        },
        "required": ["is_slider", "axis", "positive_toward", "positive_pole", "negative_pole", "range_low",
                     "range_high", "useful_low", "useful_high", "stages", "confidence", "evidence_used"],
    }},
}


def evidence_text(row: Any, weights: List[float], examples: List[Any], axes: Dict[str, Dict[str, Any]]) -> str:
    parts = [f"KNOWN AXES (id: negative end <-> positive end):\n{axes_text(axes)}",
             f"File: {row['file_name']}  (shelf: {row['rel_dir']})"]
    if row["model_name"]:
        parts.append(f"Civitai name: {row['model_name']} - {row['version_name'] or ''}")
    tags = _json_list(row["tags"])
    if tags:
        parts.append("Civitai tags: " + ", ".join(str(t) for t in tags[:15]))
    words = _json_list(row["trained_words"])
    if words:
        parts.append("Trained words: " + " | ".join(str(w) for w in words[:10]))
    description = _plain(row["description"] or "")
    if description:
        parts.append("Description:\n" + description[:MAX_DESCRIPTION])
    if weights:
        parts.append("Weights used in gallery examples: " + ", ".join(f"{w:g}" for w in weights[:20]))
    prompts = [e["prompt"] for e in examples if e["prompt"]][:3]
    if prompts:
        parts.append("Example prompts:\n" + "\n".join(f"  - {p[:250]}" for p in prompts))
    if not description and not weights:
        parts.append("(No description and no example weights: judge from the name; keep confidence low.)")
    return "\n\n".join(parts)


def call_llm(settings: Settings, model: str, text: str) -> Dict[str, Any]:
    from .llm import chat_json
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    # a truncated or malformed answer gets one retry
    return chat_json(settings, model, messages, SCHEMA, temperature=0.1, max_tokens=3000, timeout=600, retries=1)


def _clean(card: Dict[str, Any], axes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Order the ranges, snap positive_toward to an axis end when it is a near spelling."""
    low, high = sorted((float(card.get("range_low") or 0), float(card.get("range_high") or 0)))
    ulow, uhigh = sorted((float(card.get("useful_low") or low), float(card.get("useful_high") or high)))
    card["range_low"], card["range_high"] = low, high
    card["useful_low"], card["useful_high"] = max(low, ulow), min(high, uhigh)
    axis = str(card.get("axis") or "").strip().lower().replace(" ", "_")
    card["axis"] = axis
    ends = [str(e) for e in (axes.get(axis) or {}).get("ends") or []]
    toward = str(card.get("positive_toward") or "").strip()
    for end in ends:
        if toward.lower() == end.lower() or toward.lower() in end.lower() or end.lower() in toward.lower():
            card["positive_toward"] = end
            break
    return card


def write_slider_cards(settings: Settings, model: str, only: str = "", force: bool = False, limit: int = 0,
                       workers: int = 1) -> Dict[str, int]:
    catalog = Catalog(settings.catalog_path)
    ensure(catalog.connection)
    axes = load_axes()
    where, params = "1=1", []
    if only:
        where, params = "rel_dir like ?", [only.replace("\\", "/") + "%"]
    rows = [row for row in catalog.query(f"select * from loras where {where} order by rel_dir, file_name", tuple(params))
            if is_slider_row(row)]
    done = {row["path"] for row in catalog.query("select path from slider_cards")}
    todo = [row for row in rows if force or row["path"] not in done]
    already = len(rows) - len(todo)
    if limit:
        todo = todo[:limit]
    print(f"[sliders] {len(todo)} slider LoRAs to card with {model} ({already} already done)", flush=True)
    counts = {"written": 0, "not_slider": 0, "failed": 0}
    started = time.time()

    def prepare(row) -> Tuple[Any, List[float], str]:
        examples = catalog.query("select * from examples where path=?", (row["path"],))
        weights = example_weights(row, examples)
        return row, weights, evidence_text(row, weights, examples, axes)

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    pending = iter(todo)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        flying = {}
        for _ in range(max(1, workers)):
            row = next(pending, None)
            if row is None:
                break
            job = prepare(row)
            flying[pool.submit(call_llm, settings, model, job[2])] = job
        number = 0
        while flying:
            finished, _ = wait(flying, return_when=FIRST_COMPLETED)
            for future in finished:
                row, weights, _text = flying.pop(future)
                following = next(pending, None)
                if following is not None:
                    job = prepare(following)
                    flying[pool.submit(call_llm, settings, model, job[2])] = job
                number += 1
                try:
                    card = _clean(future.result(), axes)
                except Exception as error:
                    counts["failed"] += 1
                    print(f"[sliders] failed {row['file_name']}: {str(error)[:160]}", flush=True)
                    continue
                catalog.connection.execute(
                    """insert or replace into slider_cards (path, is_slider, axis, positive_toward, positive_pole,
                       negative_pole, range_low, range_high, useful_low, useful_high, stages, confidence, evidence,
                       refined_by, refined_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row["path"], int(bool(card.get("is_slider"))), card["axis"], card.get("positive_toward"),
                     card.get("positive_pole"), card.get("negative_pole"), card["range_low"], card["range_high"],
                     card["useful_low"], card["useful_high"], card.get("stages"), card.get("confidence"),
                     json.dumps({"used": card.get("evidence_used"), "example_weights": weights[:20]}),
                     model, time.strftime("%Y-%m-%d %H:%M")))
                counts["written" if card.get("is_slider") else "not_slider"] += 1
                if number % 5 == 0:
                    catalog.commit()
                    rate = (time.time() - started) / number
                    print(f"[sliders] {number}/{len(todo)}  {rate:.1f} s each  eta {(len(todo) - number) * rate / 60:.0f} min", flush=True)
    catalog.close()
    counts["seconds"] = int(time.time() - started)
    return counts


def gaps(settings: Settings) -> Dict[str, int]:
    """Axis ids slider cards use that axes.yaml does not define, with counts."""
    catalog = Catalog(settings.catalog_path)
    ensure(catalog.connection)
    known = set(load_axes())
    found: Dict[str, int] = {}
    for row in catalog.query("select axis from slider_cards where is_slider=1"):
        if row["axis"] and row["axis"] not in known:
            found[row["axis"]] = found.get(row["axis"], 0) + 1
    catalog.close()
    return found


def slider_for(connection, path: str) -> Optional[Dict[str, Any]]:
    """The slider card for one LoRA as a dict, or None (no table, no row, or not a slider)."""
    try:
        row = connection.execute("select * from slider_cards where path=? and is_slider=1", (path,)).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    keys = ["path", "is_slider", "axis", "positive_toward", "positive_pole", "negative_pole", "range_low", "range_high",
            "useful_low", "useful_high", "stages", "confidence", "evidence", "refined_by", "refined_at"]
    return {key: row[index] for index, key in enumerate(keys)}
