"""
Slider axes: read a prompt for values on the dials that slider LoRAs move (age, waist, shot size, saturation ...).

The axes and their cue words live in the LoRA Catalog's axes.yaml (owned by that tool; this is a small read-only
reader so the packages stay independent, like vocab.py).

A cue has a POSITION on its axis, -1 .. +1: -1 = the slider's full useful low end, +1 = its full useful high end,
0 = neutral. Each slider scales the position to its own range, so one ladder serves every slider on the axis.

    sets:
      people: [man, woman, fisherman, ...]          named word sets, used as @people in cues
    axes:
      waist:
        ends: [narrower, wider]
        stages: early
        ladder:                                      phrase: position (any number of rungs, both directions)
          wasp waist: -1.0
          tiny waist: -0.7
          hourglass waist: -0.57
          thick waist: 0.43
        neg: {1: [...], 2: [...], 3: [...]}          older form, still read: strength s = position s/3
        pos: {1: [...], 2: [...], 3: [...]}
        numbers: {pattern: ..., map: [[6, -3], [85, 3]]}     map values are strengths (-3..3) -> position /3
    sliders:                                         optional exact weights for one slider you have calibrated
      waist slider:                                  matched against the LoRA's name or file name (part of it)
        axis: waist
        weights: {wasp waist: -14, tiny waist: -10}  the phrase gives exactly this weight on that slider

Cue forms (ladder keys and neg/pos entries alike):
    "elderly"                   the phrase as whole words
    "old + @people"             only when a word of the set is within NEAR words of it (either side)
    "wet + skin|hair|body"      the same with words listed inline (a trailing 's' is allowed)

    read_axes(axes, "a woman with a wasp waist")  ->  {"waist": AxisValue(position=-1.0, phrase="wasp waist")}

A modifier in front of a cue scales it: very / extremely x1.25, slightly / a little x0.6 (kept within -1..1). Two cues
on one axis pointing opposite ways leave the axis unset; the reason is kept for the report.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import yaml

UP = ("very", "extremely", "super", "incredibly", "really", "exceptionally", "highly")
DOWN = ("slightly", "a little", "somewhat", "a bit", "mildly", "faintly")
NEAR = 4          # words either side of a cue that a "+ context" word may be in
EVERYDAY = 0.35   # a cue below this is an everyday word; a modifier ('very old') never lifts it to or past this
WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")


@dataclass
class Cue:
    phrase: str
    position: float                                # -1 .. 1
    context: Optional[FrozenSet[str]] = None       # None = the phrase alone is enough
    label: str = ""                                # as written in axes.yaml, for reports and exact-weight lookup
    source: str = "axes"                           # axes (hand-written) | ladders (accepted) | draft (preview only)


@dataclass
class Axis:
    name: str
    ends: Tuple[str, str]
    stages: str
    cues: List[Cue] = field(default_factory=list)  # longest phrase first
    number_pattern: Optional[re.Pattern] = None
    number_map: List[Tuple[float, float]] = field(default_factory=list)     # (number, position)


@dataclass
class SliderWeights:
    """Exact weights for one calibrated slider (axes.yaml `sliders:`)."""
    match: str                                     # part of the LoRA's name or file name, lower case
    axis: str
    weights: Dict[str, float]                      # cue label -> exact weight


@dataclass
class AxisValue:
    position: float        # -1..1, 0 = conflict (see note)
    phrase: str            # the cue label that won
    note: str = ""

    @property
    def strength(self) -> int:
        """The old 1-3 scale, signed (for reports and older callers)."""
        if not self.position:
            return 0
        step = max(1, min(3, round(abs(self.position) * 3)))
        return step if self.position > 0 else -step


@dataclass
class AxesFile:
    axes: Dict[str, Axis]
    sliders: List[SliderWeights]
    problems: List[str]


def _parse_cue(text: str, position: float, sets: Dict[str, List[str]]) -> Optional[Cue]:
    label = str(text).strip().lower()
    if not label:
        return None
    position = max(-1.0, min(1.0, float(position)))
    if " + " not in label:
        return Cue(label, position, None, label)
    phrase, _sep, rest = label.partition(" + ")
    words: List[str] = []
    for part in rest.split("|"):
        part = part.strip()
        if part.startswith("@"):
            words.extend(str(w).lower() for w in sets.get(part[1:], []))
        elif part:
            words.append(part)
    return Cue(phrase.strip(), position, frozenset(words), label)


def _level_cues(spec: dict, sign: int, sets: Dict[str, List[str]]) -> List[Cue]:
    out = []
    for level, phrases in (spec or {}).items():
        try:
            level = max(1, min(3, int(level)))
        except (TypeError, ValueError):
            continue
        for phrase in phrases or []:
            cue = _parse_cue(phrase, sign * level / 3.0, sets)
            if cue:
                out.append(cue)
    return out


def _merge_ladders(axes: Dict[str, Axis], extra: dict, sets: Dict[str, List[str]], source: str,
                   problems: List[str]) -> None:
    """A ladder from axes_ladders.yaml (or the draft) REPLACES that axis's coarse neg/pos cues, so a rung deleted
    from the ladder does not come back from the old lists. Rungs of axes.yaml's own hand-written ladder always stay
    and win over a merged rung with the same phrase."""
    for name, spec in ((extra or {}).get("axes") or {}).items():
        axis = axes.get(str(name))
        if axis is None:
            problems.append(f"{source}: axis '{name}' is not in axes.yaml")
            continue
        rungs = []
        for phrase, position in ((spec or {}).get("ladder") or {}).items():
            try:
                cue = _parse_cue(phrase, float(position), sets)
            except (TypeError, ValueError):
                problems.append(f"{source}/{name}: '{phrase}' has no number")
                continue
            if cue is not None:
                cue.source = source
                rungs.append(cue)
        if not rungs:
            continue
        hand = [c for c in axis.cues if c.source == "axes_ladder"]
        kept_labels = {c.label for c in hand}
        axis.cues = hand + [c for c in rungs if c.label not in kept_labels]
        axis.cues.sort(key=lambda c: (-len(c.phrase), c.context is None))


def parse_axes_file(data: dict, extras: Optional[List[Tuple[dict, str]]] = None) -> AxesFile:
    data = data or {}
    sets = {str(k): [str(w) for w in (v or [])] for k, v in (data.get("sets") or {}).items()}
    axes: Dict[str, Axis] = {}
    problems: List[str] = []
    for name, spec in (data.get("axes") or {}).items():
        spec = spec or {}
        ends = [str(e) for e in spec.get("ends") or []] + ["", ""]
        axis = Axis(name=str(name), ends=(ends[0], ends[1]), stages=str(spec.get("stages") or "all"))
        cues = _level_cues(spec.get("neg"), -1, sets) + _level_cues(spec.get("pos"), 1, sets)
        for phrase, position in (spec.get("ladder") or {}).items():
            try:
                cue = _parse_cue(phrase, float(position), sets)
            except (TypeError, ValueError):
                problems.append(f"{name}: ladder value for '{phrase}' is not a number")
                continue
            if cue:
                cue.source = "axes_ladder"      # hand-written ladder in axes.yaml: never replaced by a merged rung
                cues = [c for c in cues if c.label != cue.label] + [cue]     # a ladder rung replaces a same-named cue
        # longest first; at equal length a cue with a context goes first (it is the more specific reading)
        axis.cues = sorted(cues, key=lambda c: (-len(c.phrase), c.context is None))
        numbers = spec.get("numbers") or {}
        if numbers.get("pattern"):
            try:
                axis.number_pattern = re.compile(str(numbers["pattern"]), re.I)
                axis.number_map = sorted((float(a), max(-1.0, min(1.0, float(b) / 3.0))) for a, b in numbers.get("map") or [])
            except (re.error, TypeError, ValueError):
                axis.number_pattern = None
                problems.append(f"{name}: numbers rule unreadable")
        axes[axis.name] = axis
    for extra, source in extras or []:
        _merge_ladders(axes, extra, sets, source, problems)
    sliders: List[SliderWeights] = []
    for match, spec in (data.get("sliders") or {}).items():
        spec = spec or {}
        axis_name = str(spec.get("axis") or "")
        weights = {}
        for phrase, weight in (spec.get("weights") or {}).items():
            try:
                weights[str(phrase).strip().lower()] = float(weight)
            except (TypeError, ValueError):
                problems.append(f"sliders/{match}: weight for '{phrase}' is not a number")
        axis = axes.get(axis_name)
        if axis is None:
            problems.append(f"sliders/{match}: axis '{axis_name}' is not defined")
        else:
            labels = {c.label for c in axis.cues}
            for phrase in weights:
                if phrase not in labels:
                    problems.append(f"sliders/{match}: '{phrase}' is not a cue on axis {axis_name} (add it to the ladder)")
        sliders.append(SliderWeights(str(match).strip().lower(), axis_name, weights))
    return AxesFile(axes, sliders, problems)


def parse_axes(data: dict) -> Dict[str, Axis]:
    return parse_axes_file(data).axes


_cache: Dict[str, tuple] = {}


def _stamp(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _read_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except OSError:
        return {}


def load_axes_file(path: str, with_draft: bool = False) -> AxesFile:
    """axes.yaml plus the accepted ladders beside it (axes_ladders.yaml); with_draft also merges
    axes_ladders_draft.yaml (for the review page only). Empty when axes.yaml is missing. Re-read on change."""
    if not os.path.exists(path):
        return AxesFile({}, [], [f"axes file not found: {path}"])
    folder = os.path.dirname(path)
    live = os.path.join(folder, "axes_ladders.yaml")
    draft = os.path.join(folder, "axes_ladders_draft.yaml")
    key = f"{path}|{with_draft}"
    stamp = (_stamp(path), _stamp(live), _stamp(draft) if with_draft else 0.0)
    cached = _cache.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    extras = [(_read_yaml(live), "ladders")]
    if with_draft:
        extras.append((_read_yaml(draft), "draft"))
    parsed = parse_axes_file(_read_yaml(path), extras)
    _cache[key] = (stamp, parsed)
    return parsed


def load_axes(path: str) -> Dict[str, Axis]:
    return load_axes_file(path).axes


def exact_weights_for(sliders: List[SliderWeights], name: str, file_name: str, axis: str) -> Optional[Dict[str, float]]:
    """Your calibrated weights for this slider on this axis, or None."""
    haystack = f"{name} {file_name}".lower()
    for entry in sliders:
        if entry.axis == axis and entry.match and entry.match in haystack:
            return entry.weights
    return None


def _number_position(axis: Axis, value: float) -> float:
    """The position of the nearest map point (85 -> +1.0 with the age map)."""
    if not axis.number_map:
        return 0.0
    return min(axis.number_map, key=lambda point: abs(point[0] - value))[1]


def _modified(text: str, start: int, position: float) -> float:
    before = text[max(0, start - 16):start]
    if any(re.search(rf"\b{re.escape(word)}\s+$", before) for word in UP):
        lifted = max(-1.0, min(1.0, position * 1.25))
        if abs(position) < EVERYDAY and abs(lifted) >= EVERYDAY:
            lifted = (EVERYDAY - 0.01) * (1 if position > 0 else -1)
        return lifted
    if any(re.search(rf"\b{re.escape(word)}\s+$", before) for word in DOWN):
        return position * 0.6
    return position


def _near(tokens: List[Tuple[int, int, str]], start: int, end: int, context: FrozenSet[str]) -> bool:
    """A context word within NEAR words before `start` or after `end` (char offsets of the cue)."""
    before = [t for t in tokens if t[1] <= start][-NEAR:]
    after = [t for t in tokens if t[0] >= end][:NEAR]
    for _s, _e, word in before + after:
        if word in context or (word.endswith("s") and word[:-1] in context) or (word + "s") in context:
            return True
    return False


def read_axes(axes: Dict[str, Axis], text: str) -> Dict[str, AxisValue]:
    lowered = " " + re.sub(r"[_/,;:()\[\]]+", " ", (text or "").lower()) + " "
    tokens = [(m.start(), m.end(), m.group(0)) for m in WORD.finditer(lowered)]
    found: Dict[str, AxisValue] = {}
    for axis in axes.values():
        hits: List[Tuple[float, str]] = []
        claimed = [False] * len(lowered)
        if axis.number_pattern:
            for match in axis.number_pattern.finditer(lowered):
                position = _number_position(axis, float(match.group(1)))
                if position:
                    hits.append((position, match.group(0).strip()))
                for i in range(*match.span()):
                    claimed[i] = True
        for cue in axis.cues:
            for match in re.finditer(r"(?<![a-z0-9-])" + re.escape(cue.phrase) + r"(?![a-z0-9-])", lowered):
                start, end = match.span()
                if any(claimed[start:end]):
                    continue
                if cue.context is not None and not _near(tokens, start, end, cue.context):
                    continue          # 'old' without a person nearby: an old town, an old car
                for i in range(start, end):
                    claimed[i] = True
                hits.append((_modified(lowered, start, cue.position), cue.label))
        hits = [h for h in hits if h[0]]
        if not hits:
            continue
        signs = {1 if p > 0 else -1 for p, _ in hits}
        if len(signs) > 1:
            found[axis.name] = AxisValue(0.0, " / ".join(p for _, p in hits[:3]), "cues point both ways")
            continue
        position, phrase = max(hits, key=lambda h: abs(h[0]))
        found[axis.name] = AxisValue(round(position, 3), phrase)
    return found
