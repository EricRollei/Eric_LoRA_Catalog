"""
Slider ladders: draft many graded prompt phrases per axis with the LLM, for review before they are used.

    python run.py ladders --draft                 every axis without a ladder in axes.yaml -> axes_ladders_draft.yaml
    python run.py ladders --draft --axis age      one axis (redrafts it)
    python run.py ladders --accept                copy the draft file to axes_ladders.yaml (the live one)
    python run.py ladders --tidy                  re-apply the guards below to the draft file (after a run or your edits)

Guards applied to every draft: a phrase axes.yaml lists at strength 1 (an everyday word you chose to keep off) stays
below the firing line (EVERYDAY_CAP) unless you move it in the draft yourself after tidy; drafted rungs whose order
looks wrong are not changed, only listed by --tidy for your review.

Files (next to axes.yaml):
    axes.yaml                 hand-written: ends, sets, neg/pos lists, hand ladders, exact per-slider weights
    axes_ladders_draft.yaml   the LLM's drafts; edit freely (delete or move rungs); not read by the Suggester
    axes_ladders.yaml         accepted ladders; read by the Suggester and merged into axes.yaml's axes. A rung that
                              axes.yaml itself has (same phrase) keeps the axes.yaml position.

A rung is  "phrase": position   with position -1 .. +1 (-1 = the slider's full useful low end, +1 = its full high end).
Review the drafts with video_prompter\\dev\\slider_cue_table.py --draft (shows each rung's weight on every slider).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List

import yaml

from .config import DATA_DIR, Settings, user_file
from .sliders import AXES_PATH

EVERYDAY_CAP = 0.3          # the Suggester's default firing line is 0.35
DRAFT_PATH = os.path.join(DATA_DIR, "axes_ladders_draft.yaml")
LIVE_PATH = user_file("axes_ladders.yaml")        # your copy in the data folder (the shipped one is the start)

SYSTEM = """You write prompt-cue ladders for slider LoRAs (image-model adapters that act as dials: a negative weight
pushes the picture toward one end of an attribute, a positive weight toward the other).

For ONE axis you get its two ends, the cues it has now, the named word sets you may use for context, and the other
axes (so you stay out of their territory). Write a graded ladder: the phrases a person would actually type in an
image prompt to ask for a place on this axis, each with a position from -1 to +1:
  -1 = the most extreme request for the first end, +1 = the most extreme for the second end, 0 = neutral.
  Grade carefully: mild asks near +-0.2 .. 0.3, clear asks +-0.4 .. 0.6, strong +-0.7 .. 0.85, extreme +-0.9 .. 1.0.

The current cues are a COARSE three-step list (positions only 0.33 / 0.67 / 1.0). Your job is to make it a fine ladder:
- At least half of your rungs must be NEW phrases that are not in the current cues (synonyms, compound phrases,
  descriptive phrases people really use: "in her late sixties", "silver-haired", "crow's feet", "baby-faced").
- Do NOT use the positions 0.33, 0.67 or 1.0 (or their negatives), except that the single most extreme phrase of each
  end may be +-1.0. Give every rung its own considered position (0.15, 0.25, 0.4, 0.52, 0.6, 0.72, 0.8, 0.88, 0.95 ...).
- Order matters more than anything: a phrase that asks for MORE of an end must sit further out than one that asks
  for less. Check each neighbouring pair (is 'portly' really more than 'fat'? is 'wiry' really thinner than 'bony'?).
- A phrase must ask for THIS attribute: 'wiry' is lean and muscular, not extremely thin.

Worked example (waist, ends narrower <-> wider), placed by an artist who calibrated a real slider:
  wasp waist -1.0, tiny waist -0.71, hourglass waist -0.57, thin waist -0.36, no waist +0.29, thick waist +0.43,
  fat waist +0.57.

Rules:
- 8 to 14 rungs per end when the end makes sense in prompts; fewer when it does not (a one-sided attribute).
- Phrases lowercase, 1-4 words, exactly as typed in prompts ("wasp waist", "plus-size", "extreme close-up").
- Everyday words that usually describe a style or mood rather than a degree of THIS attribute ("detailed", "dark",
  "illustration", "busy", "old") get a position below 0.35 in size (they then do nothing by default), OR are written
  with context: "old + @people" means it counts only when a word of the people set is within four words;
  "wet + skin|hair|body" lists context words inline. Use context for any word with a common other meaning.
- No phrase that belongs to another axis. No duplicates. Keep the existing cues you agree with (you may move them).
- Only phrases describing the attribute; never people's names, brands or LoRA trigger words.
- why: a few words, only when the placement is not obvious."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "ladder", "strict": True, "schema": {
        "type": "object",
        "properties": {"rungs": {"type": "array", "items": {"type": "object", "properties": {
            "phrase": {"type": "string"}, "position": {"type": "number", "minimum": -1, "maximum": 1},
            "why": {"type": "string"}}, "required": ["phrase", "position", "why"]}}},
        "required": ["rungs"],
    }},
}


def _raw(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except OSError:
        return {}


def _existing_cues(spec: Dict[str, Any]) -> List[str]:
    out = []
    for sign, key in ((-1, "neg"), (1, "pos")):
        for level, phrases in (spec.get(key) or {}).items():
            for phrase in phrases or []:
                out.append(f"{phrase}: {sign * int(level) / 3:+.2f}")
    for phrase, position in (spec.get("ladder") or {}).items():
        out.append(f"{phrase}: {float(position):+.2f}")
    return out


def _prompt(name: str, spec: Dict[str, Any], sets: Dict[str, List[str]], others: List[str]) -> str:
    ends = spec.get("ends") or ["?", "?"]
    lines = [f"AXIS: {name}   first end (-1): {ends[0]}   second end (+1): {ends[1]}",
             "CURRENT CUES (phrase: position):\n" + ("\n".join("  " + c for c in _existing_cues(spec)) or "  none"),
             "WORD SETS for context (@name): " + "; ".join(f"@{k} = {', '.join(v[:12])}{' ...' if len(v) > 12 else ''}"
                                                       for k, v in sets.items()),
             "OTHER AXES (not yours): " + ", ".join(others)]
    return "\n\n".join(lines)


def call_llm(settings: Settings, model: str, text: str) -> List[Dict[str, Any]]:
    from .llm import chat_json
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    last: Exception = RuntimeError("no attempt")
    best: List[Dict[str, Any]] = []
    for _attempt in range(3):
        try:
            answer, content = chat_json(settings, model, messages, SCHEMA, temperature=0.5, max_tokens=9000,
                                        timeout=900, retries=0, with_text=True)
            rungs = answer.get("rungs") or []
        except ValueError as error:       # empty or cut-off answer (reasoning used the budget): ask again
            last = error
            continue
        coarse = sum(1 for r in rungs if round(abs(float(r.get("position") or 0)), 2) in (0.33, 0.67))
        best = rungs
        if rungs and coarse <= len(rungs) // 4:
            return rungs
        # too many rungs on the old 0.33 / 0.67 steps: one more try, told so
        messages = messages[:2] + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": f"{coarse} of your {len(rungs)} rungs still sit on 0.33 or "
                                    "0.67. Place each one on its own position as instructed, and check the order of neighbours. "
                                    "Return the whole ladder again."}]
    if best:
        return best
    raise last


def everyday_labels(spec: Dict[str, Any]) -> set:
    """Phrases axes.yaml lists at strength 1 for this axis."""
    out = set()
    for key in ("neg", "pos"):
        for level, phrases in ((spec or {}).get(key) or {}).items():
            if str(level) == "1":
                out.update(" ".join(str(p).lower().split()) for p in phrases or [])
    return out


def guard(rungs: List[Dict[str, Any]], everyday: set) -> List[Dict[str, Any]]:
    for rung in rungs:
        if rung["phrase"] in everyday and abs(rung["position"]) > EVERYDAY_CAP:
            rung["position"] = EVERYDAY_CAP if rung["position"] > 0 else -EVERYDAY_CAP
            rung["why"] = (rung.get("why") + "; " if rung.get("why") else "") + "kept everyday (strength 1 in axes.yaml)"
    return sorted(rungs, key=lambda r: r["position"])


def tidy() -> Dict[str, int]:
    """Re-apply the guards to the draft file in place."""
    axes = _raw(AXES_PATH).get("axes") or {}
    raw = _raw(DRAFT_PATH).get("axes") or {}
    drafts, moved = {}, 0
    for name, spec in raw.items():
        rungs = [{"phrase": " ".join(str(p).lower().split()), "position": float(v), "why": ""}
                 for p, v in ((spec or {}).get("ladder") or {}).items()]
        before = {r["phrase"]: r["position"] for r in rungs}
        drafts[name] = guard(rungs, everyday_labels(axes.get(name) or {}))
        moved += sum(1 for r in drafts[name] if before[r["phrase"]] != r["position"])
    write_draft(drafts, "tidy")
    return {"axes": len(drafts), "capped": moved}


def _clean(rungs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for rung in rungs:
        phrase = " ".join(str(rung.get("phrase") or "").lower().split())
        try:
            position = max(-1.0, min(1.0, round(float(rung.get("position")), 2)))
        except (TypeError, ValueError):
            continue
        if not phrase or phrase in seen or position == 0:
            continue
        seen.add(phrase)
        out.append({"phrase": phrase, "position": position, "why": str(rung.get("why") or "").strip()})
    return sorted(out, key=lambda r: r["position"])


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_draft(drafts: Dict[str, List[Dict[str, Any]]], model: str, path: str = DRAFT_PATH) -> None:
    """Hand-formatted YAML so each rung is one editable line with its reason as a comment."""
    lines = ["# Slider ladder DRAFTS - written by the LLM, not used by the Suggester until accepted.",
             "# Delete rungs you do not want, change positions (-1 .. +1), add your own. Then: python run.py ladders --accept",
             "# Review the weights each rung gives on every slider: video_prompter\\dev\\slider_cue_table.py --draft",
             f"# Drafted {time.strftime('%Y-%m-%d %H:%M')} with {model}.", "", "axes:"]
    for axis in sorted(drafts):
        lines.append(f"  {axis}:")
        lines.append("    ladder:")
        for rung in drafts[axis]:
            comment = f"   # {rung['why']}" if rung["why"] else ""
            lines.append(f"      {_quote(rung['phrase'])}: {rung['position']:+.2f}{comment}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def draft(settings: Settings, model: str, only_axis: str = "") -> Dict[str, int]:
    data = _raw(AXES_PATH)
    axes = data.get("axes") or {}
    sets = {str(k): [str(w) for w in (v or [])] for k, v in (data.get("sets") or {}).items()}
    existing = (_raw(DRAFT_PATH).get("axes") or {}) if os.path.exists(DRAFT_PATH) else {}
    drafts = {name: [{"phrase": str(p), "position": float(v), "why": ""} for p, v in ((spec or {}).get("ladder") or {}).items()]
              for name, spec in existing.items()}
    todo = [name for name, spec in axes.items()
            if (only_axis and name == only_axis) or (not only_axis and not (spec or {}).get("ladder") and name not in drafts)]
    print(f"[ladders] drafting {len(todo)} axes with {model}", flush=True)
    counts = {"axes": 0, "rungs": 0, "failed": 0}
    started = time.time()
    for number, name in enumerate(todo, 1):
        others = [other for other in axes if other != name]
        try:
            rungs = guard(_clean(call_llm(settings, model, _prompt(name, axes[name] or {}, sets, others))),
                          everyday_labels(axes[name] or {}))
        except Exception as error:
            counts["failed"] += 1
            print(f"[ladders] {name} failed: {str(error)[:160]}", flush=True)
            continue
        drafts[name] = rungs
        counts["axes"] += 1
        counts["rungs"] += len(rungs)
        write_draft(drafts, model)          # after every axis, so a stopped run keeps what it has
        print(f"[ladders] {number}/{len(todo)} {name}: {len(rungs)} rungs  ({time.time() - started:.0f} s)", flush=True)
    counts["seconds"] = int(time.time() - started)
    return counts


LIVE_HEADER = [
    "# LIVE slider ladders - read by the LoRA Suggester (video_prompter) on every run; no ComfyUI restart needed.",
    "# Each line:  \"phrase\": position   -1 .. +1 on the axis (-1 = the slider's full useful low end, +1 = its high end).",
    "# 0.3 = seen but off by default (the node's slider_min_strength 2 fires at 0.35). Quote phrases with + : # or @.",
    "# A ladder here REPLACES that axis's neg/pos lists in axes.yaml; a hand ladder inside axes.yaml wins over this file.",
    "# Edit freely. Full guide: SLIDERS_AND_AXES.md. Check: video_prompter\\dev\\slider_cue_table.py krea2",
]


def accept() -> str:
    if not os.path.exists(DRAFT_PATH):
        return "no draft file"
    if os.path.exists(LIVE_PATH):
        shutil.copy2(LIVE_PATH, LIVE_PATH + time.strftime(".bak_%Y%m%d_%H%M%S"))
    with open(DRAFT_PATH, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    body = lines[next((i for i, line in enumerate(lines) if not line.startswith("#")), len(lines)):]
    stamp = f"# Accepted {time.strftime('%Y-%m-%d %H:%M')} from axes_ladders_draft.yaml."
    with open(LIVE_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(LIVE_HEADER + [stamp] + body) + "\n")
    return f"accepted -> {LIVE_PATH}"
