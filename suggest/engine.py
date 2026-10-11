"""
LoRA Suggester engine: read a prompt (Composer sections or plain text) in the catalog's own pick-list words, score
every LoRA card of the platform for the role it plays, and build a stack by role with per-stage weights.

    result = suggest(SuggestRequest(platform="krea2", sections=..., text=...))
    result.picks      [Pick]   (in stack order)
    result.report     str      why each pick won, the runner-up it beat, what was excluded and why

Scoring (no LLM, no embeddings - instant):
    raw = sum over the role's facets of  weight x  tree match (prompt terms vs card terms; narrower/broader terms
          score part)  +  text similarity (idf-weighted words of the role's text vs the card's name/effect/triggers)
          +  exact hits (artist name, the person it depicts, a distinctive trigger word)
    score = 1 - exp(-raw / 3)   (0..1, the apply threshold works on this)
    then multipliers: low card confidence, LoRA Manager favourite, reviewed by you, your star rating
    (RATING_FACTOR), side effects that clash; for UNRATED cards only, the Civitai community score (LoRA Catalog
    `run.py community`) nudges by at most +-COMMUNITY_NUDGE - a tie-break, smaller than one star step.

NSFW (spec sections 8-9): the node's gate (off | suggestive | nudity | explicit) caps what may be used; NSFW LoRAs
are used only when the prompt itself asks for that content (effective level = min(prompt level, gate)); suggestive
LoRAs are allowed whenever the gate allows suggestive. Hard rule: a LoRA flagged youthful_look is never stacked
with a nudity/explicit LoRA. Config switch block_real_person_nsfw (default on): a real-person likeness LoRA is never
stacked with a nudity/explicit LoRA.

Sliders (2026-10-08): a slider LoRA is a dial, never scored on text. The prompt is read for positions on the axes
in axes.yaml ("wasp waist" -> waist -1.0, "thick waist" -> +0.43); a slider whose card names that axis is set to
position x the useful end on that side (your editor weights, else the slider card, else the node's slider_range),
or to your exact weight for that phrase when axes.yaml `sliders:` calibrates that slider. A pin may carry a value:
`Age Slider=+3`, or name an axis: `axis:age=+3`.

exclude / pin boxes (2026-10-08): whole-word matching with folder: / role: / axis: prefixes - see filters.py.

Style families (2026-10-08): a second style is allowed only when it shares a family with the first (or the two
families are friends in config); it runs as a blend at blend_ratio x its weight.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import filters
from .axes import AxisValue, exact_weights_for, load_axes_file, read_axes
from .catalog import ROLES, LoraCard, load_cards, words_of
from .settings import SuggestSettings, load_settings
from .vocab import PickLists, load_pick_lists

NSFW_LEVELS = ["off", "suggestive", "nudity", "explicit"]
NSFW_TERMS = ["sfw", "suggestive", "nudity", "explicit"]

# Which Composer sections feed which role. None = the whole prompt.
SECTION_ROLES: Dict[str, Optional[Tuple[str, ...]]] = {
    # A Prompt line (a whole prompt from _prompts) can say anything, so it feeds every role.
    "style": ("style", "format", "prompt"),
    "transform": None,
    "treatment": ("style", "format", "prompt"),
    "detailer": None,
    "environment": ("setting", "prompt"),
    "lighting": ("lighting", "prompt"),
    "subject": ("subject", "action", "prompt"),
    "layout": ("composition", "camera", "format", "prompt"),
    "utility": None,
    "slider": None,
}

# How much each pick-list facet counts for each role.
FACET_WEIGHTS: Dict[str, Dict[str, float]] = {
    # genre (gothic, cyberpunk, noir ...) and mood (eerie, cozy ...) added 2026-10-10; both read from the whole prompt
    "style": {"period": 3.0, "medium": 2.5, "use": 1.5, "rendering": 1.0, "palette": 0.6, "line": 0.8,
              "photo_style": 2.0, "cinematic": 2.0, "film_stock": 2.0, "dimension": 0.4, "subjects": 0.5,
              "genre": 2.5, "mood": 1.2},
    "transform": {"period": 2.5, "use": 1.0, "palette": 0.6, "material": 1.5, "pattern": 1.0, "cinematic": 1.0,
                  "lighting": 0.6, "subjects": 0.5, "genre": 2.0, "mood": 1.5},
    "treatment": {"medium": 2.0, "line": 2.0, "palette": 1.0, "pattern": 2.0, "material": 2.5, "film_stock": 2.0,
                  "cinematic": 1.0, "rendering": 0.8, "mood": 0.6},
    "detailer": {"detail": 3.0, "photo_style": 0.5, "mature_content": 1.0},
    "environment": {"subjects": 1.5, "period": 1.0, "cinematic": 0.5, "lighting": 0.5, "genre": 2.0, "mood": 1.5},
    "lighting": {"lighting": 3.0, "palette": 1.0, "cinematic": 1.0, "mood": 2.0},
    "subject": {"figure": 2.0, "appearance": 1.2, "costume": 2.0, "subjects": 1.0, "mature_content": 2.5,
                "material": 1.0, "pattern": 0.5, "genre": 1.0},
    "layout": {"composition": 2.5, "use": 1.5},
    "utility": {"mature_content": 2.0},
}
TEXT_WEIGHT = {"style": 2.0, "transform": 2.5, "treatment": 1.5, "detailer": 1.0, "environment": 3.0, "lighting": 2.0, "subject": 3.0,
               "layout": 2.0, "utility": 1.5}
WHOLE_PROMPT_FACETS = {"subjects", "mature_content", "nsfw", "detail", "genre", "mood"}     # read from the whole prompt for every role
SCALE = 3.0

# Fraction of the card's typical weight per stage (S1 composition, S2 refine, S3 detail).
STAGE_PROFILE = {
    "style": (1.0, 1.0, 0.8), "transform": (1.0, 0.8, 0.5), "treatment": (0.5, 1.0, 1.0), "detailer": (0.0, 0.6, 1.0),
    "environment": (1.0, 0.7, 0.4), "lighting": (0.8, 1.0, 0.8), "subject": (1.0, 1.0, 0.8),
    "layout": (1.0, 0.8, 0.5), "utility": (1.0, 1.0, 0.6),
}
# Two picks that both take over one of these clash (the weaker is dropped), unless one is a treatment/detailer.
CLASHING_OVERRIDES = {"faces", "composition", "anatomy", "background", "camera"}
# Sharing one of these is allowed, but the later pick is turned down (cards list them often and loosely).
SOFT_OVERRIDES = {"lighting", "colour"}
# Words too common in prompts to say anything about a LoRA.
PROMPT_STOP = set("""between near over under behind front side small large big little new old young tall short
long wide high low dark light bright soft hard full half top bottom left right inside outside around across along
against among above below beside during while very really quite scene background foreground detailed detail
beautiful pretty nice good great best high-quality quality masterpiece shot view standing sitting looking holding
wearing woman man person people girl boy figure""".split())
# Cards that ARE a particular character or person: only offered when the prompt names them.
LIKENESS = {"real_person", "fictional_character", "original_character"}
LAYERS = {"transform", "treatment", "detailer", "utility"}     # stack on top of a style without clashing

# Roles that set the look rather than the content: a style that can paint nudes (Vallejo, Frazetta) stays usable
# for a SFW prompt, turned down a little; it is never excluded for that alone.
LOOK_ROLES = {"style", "transform", "treatment", "lighting", "environment", "layout"}

# Your 1-5 stars from the card editor; unrated = 1.0.
RATING_FACTOR = {1: 0.5, 2: 0.8, 3: 1.0, 4: 1.08, 5: 1.15}
# Unrated cards only: the Civitai community score (0-100, percentile within base model + role) scales the score by
# 1 +- COMMUNITY_NUDGE (top of its group +3%, bottom -3%). A popularity prior, never a verdict: any star you give wins.
COMMUNITY_NUDGE = 0.03


def community_factor(card: LoraCard) -> float:
    if card.rating or card.community is None:
        return 1.0
    return 1.0 + COMMUNITY_NUDGE * (card.community - 50.0) / 50.0

MATURE_LEVEL = {"explicit_sexual": 3, "fetish": 2, "artistic_nude": 2, "boudoir": 2, "lingerie_swimwear_focus": 1}

# slider_min_strength (node widget) -> the smallest |position| that dials a slider. 0.35 keeps the old strength-1
# cues (position 0.33: everyday words) off by default.
MIN_POSITION = {1: 0.01, 2: 0.35, 3: 0.65}

# Slider stage shapes (fraction of the dial weight at S1 / S2 / S3), by the axis or slider card's `stages`.
SLIDER_STAGES = {"early": (1.0, 1.0, 0.6), "late": (0.3, 1.0, 1.0), "all": (1.0, 1.0, 1.0)}

# Style families worked out from a card's pick-list terms when it has no `family` facet of its own:
# (family, facet, terms) - a card term counts when it IS one of the terms or sits below one in the tree.
FAMILY_RULES = [
    ("anime", "use", ("anime", "manga")),
    ("comic_illustration", "use", ("comics", "illustration", "animation_look", "cover_art", "card_art",
                                   "character_sheet", "coloring_book", "concept_art")),
    ("comic_illustration", "medium", ("pen_and_ink", "brush_ink", "cel", "line_art", "marker")),
    ("print_graphic", "use", ("poster", "logo", "typography", "vector_flat", "icon_sticker", "pattern", "packaging",
                              "infographic", "collage")),
    ("print_graphic", "medium", ("print", "graphic")),
    ("painterly", "medium", ("paint",)),
    ("painterly", "rendering", ("painterly",)),
    ("photographic", "use", ("photography",)),
    ("photographic", "medium", ("photo_process",)),
    ("three_d", "use", ("three_d_render",)),
    ("three_d", "dimension", ("three_d",)),
]


@dataclass
class SuggestRequest:
    platform: str = "krea2"
    sections: Dict[str, str] = field(default_factory=dict)
    text: str = ""
    max_loras: int = 4
    role_max: Dict[str, int] = field(default_factory=lambda: {
        "style": 1, "transform": 1, "treatment": 2, "detailer": 1, "environment": 1, "lighting": 1, "subject": 1, "layout": 1,
        "utility": 1, "slider": 2})
    threshold: float = 0.35
    weight_budget: float = 2.5
    stage_profile: str = "role"          # role | flat
    variety: float = 0.2
    seed: int = 0
    nsfw: str = "off"
    exclude: str = ""
    pin: str = ""
    prompt_nsfw: Optional[int] = None    # the Composer's nsfw level (0..3) from PROMPT_SECTIONS; None = not given
    slider_range: float = 3.0            # a slider with no range on its card dials within -range..+range
    slider_min_strength: int = 2         # 1 = any cue, 2 = position >= 0.35, 3 = >= 0.65 (MIN_POSITION)


@dataclass
class Pick:
    card: LoraCard
    role: str
    score: float
    reasons: List[str]
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    pinned: bool = False
    soft: float = 1.0          # weight factor from a shared lighting/colour takeover
    dial: Optional[float] = None   # a slider's (or a pinned-with-value LoRA's) signed weight before stage shaping

    def entry(self) -> Dict[str, Any]:
        """One LORA_SUGGESTIONS entry (what a loader reads)."""
        return {"path": self.card.path, "lora_name": self.card.lora_name, "name": self.card.name, "role": self.role,
                "score": round(self.score, 3), "weight_s1": self.weights[0], "weight_s2": self.weights[1],
                "weight_s3": self.weights[2], "triggers": list(self.card.triggers[:2]), "nsfw": self.card.nsfw,
                "reason": "; ".join(self.reasons[:4])}


@dataclass
class SuggestResult:
    picks: List[Pick]
    report: str
    triggers: List[str]
    prompt: str


# ----------------------------------------------------------------------------------------------- reading the prompt
@dataclass
class PromptView:
    whole: str
    role_text: Dict[str, str]
    role_terms: Dict[str, Dict[str, List[str]]]
    whole_terms: Dict[str, List[str]]
    level: int                      # 0..3 what the prompt asks for
    kinds: List[str]                # mature_content terms asked for
    photo: bool
    wants_text: bool


def _join_sections(sections: Dict[str, str], names: Optional[Tuple[str, ...]]) -> str:
    if names is None:
        return ", ".join(v for v in sections.values() if v)
    return ", ".join(sections.get(name, "") for name in names if sections.get(name))


def read_prompt(lists: PickLists, sections: Dict[str, str], text: str) -> PromptView:
    sections = {k: str(v) for k, v in (sections or {}).items() if str(v or "").strip()}
    whole = text.strip() or _join_sections(sections, None)
    role_text = {}
    for role, names in SECTION_ROLES.items():
        own = _join_sections(sections, names) if sections else ""
        role_text[role] = own if own.strip() else ""
    whole_terms = {facet: lists.match(facet, whole) for facet in lists.facets}
    role_terms = {role: ({facet: lists.match(facet, body) for facet in lists.facets} if body else {})
                  for role, body in role_text.items()}
    level = 0
    for term in whole_terms.get("nsfw", []):
        if term in NSFW_TERMS:
            level = max(level, NSFW_TERMS.index(term))
    kinds = whole_terms.get("mature_content", [])
    for kind in kinds:
        level = max(level, MATURE_LEVEL.get(kind, 0))
    lowered = whole.lower()
    photo = bool(whole_terms.get("photo_style") or whole_terms.get("film_stock")) or any(
        word in lowered for word in ("photo", "photograph", "35mm", "dslr", "film still"))
    wants_text = bool(sections.get("text")) or bool(set(whole_terms.get("use", [])) & {"typography", "poster", "logo"})
    return PromptView(whole, role_text, role_terms, whole_terms, level, kinds, photo, wants_text)


# ----------------------------------------------------------------------------------------------- scoring
def _idf(cards: List[LoraCard]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for card in cards:
        for word in card.words:
            counts[word] = counts.get(word, 0) + 1
    total = max(1, len(cards))
    return {word: math.log(1 + total / count) for word, count in counts.items()}


def _text_similarity(prompt_words: List[str], card: LoraCard, idf: Dict[str, float]) -> Tuple[float, List[str]]:
    asked = {w for w in prompt_words if w in idf and w not in PROMPT_STOP}
    if not asked or not card.words:
        return 0.0, []
    shared = [w for w in asked if w in card.words]
    if not shared:
        return 0.0, []
    top = sum(idf[w] ** 2 for w in shared)
    norm = math.sqrt(sum(idf[w] ** 2 for w in asked) * sum(idf.get(w, 0) ** 2 for w in card.words))
    shared.sort(key=lambda w: -idf[w])
    return (top / norm if norm else 0.0), shared[:4]


def _facet_match(lists: PickLists, facet: str, asked: List[str], card_terms: List[str]) -> Tuple[float, List[str]]:
    total, hits = 0.0, []
    for term in asked:
        best, best_term = 0.0, ""
        for card_term in card_terms:
            value = lists.similarity(facet, term, card_term)
            if value > best:
                best, best_term = value, card_term
        if best > 0:
            total += best
            hits.append(best_term if best_term == term else f"{term}~{best_term}")
    return min(total, 2.0), hits


def _flat(text: str) -> str:
    return " " + " ".join(re.sub(r"[^\w']+", " ", (text or "").lower()).split()) + " "


def _contains_name(text_lower: str, name: str) -> bool:
    """A name or phrase as whole words, ignoring punctuation and underscores ('Boris Vallejo,' counts)."""
    flat_name = _flat(name.replace("_", " ")).strip()
    return len(flat_name) >= 4 and f" {flat_name} " in _flat(text_lower.replace("_", " "))


def score_card(card: LoraCard, view: PromptView, lists: PickLists, idf: Dict[str, float],
               distinctive_cut: float) -> Tuple[float, List[str]]:
    role = card.role
    reasons: List[str] = []
    raw = 0.0
    own = view.role_terms.get(role) or {}
    factor = 1.0 if view.role_text.get(role) else 0.6        # no section for this role: read the whole prompt, softer
    for facet, weight in FACET_WEIGHTS.get(role, {}).items():
        asked = view.whole_terms.get(facet, []) if (facet in WHOLE_PROMPT_FACETS or not own) else own.get(facet, [])
        if not asked:
            continue
        value, hits = _facet_match(lists, facet, asked, card.facets.get(facet, []))
        if value:
            raw += weight * value * (1.0 if facet in WHOLE_PROMPT_FACETS else factor)
            reasons.append(f"{facet}: {', '.join(hits)}")
    facet_hit = bool(reasons)
    body = view.role_text.get(role) or view.whole
    similarity, shared = _text_similarity(words_of(body), card, idf)
    if similarity and (len(shared) >= 2 or facet_hit):     # one shared word alone is too thin
        raw += TEXT_WEIGHT.get(role, 1.5) * similarity * 3.0
        if similarity > 0.05:
            reasons.append("words: " + ", ".join(shared))
    lowered = " ".join(words_of(view.whole)) + " " + view.whole.lower()
    for artist in card.artists:
        if _contains_name(view.whole.lower(), artist):
            raw += 6.0 if role in ("style", "treatment") else 2.0
            reasons.insert(0, f"artist named: {artist}")
        else:
            surname = artist.split()[-1].lower() if artist.split() else ""
            if len(surname) >= 5 and surname in _flat(view.whole).split():
                raw += 2.5
                reasons.insert(0, f"artist surname: {artist}")
    for person in card.depicts:
        if _contains_name(view.whole.lower(), person):
            raw += 8.0
            reasons.insert(0, f"depicts: {person}")
    for trigger in card.triggers + card.triggers_off:      # a switched-off trigger still names the LoRA
        trigger_words = words_of(trigger)
        if len(trigger_words) == 1 and len(trigger.strip()) <= 5:
            continue          # a short everyday word ('fish') used as a trigger does not name the LoRA
        if trigger_words and max(idf.get(w, 0) for w in trigger_words) >= distinctive_cut \
                and _contains_name(view.whole.lower(), trigger):
            raw += 4.0
            reasons.insert(0, f"trigger: {trigger}")
    named = any(r.startswith(("depicts:", "trigger:", "artist named:")) for r in reasons)
    score = 1.0 - math.exp(-raw / SCALE)
    # multipliers
    figure = set(card.facets.get("figure", []))
    individual = bool(LIKENESS & figure) or (card.kind in ("character", "person") and "generic_type" not in figure)
    if individual and not named:
        score *= 0.3
        reasons.append("a particular character, not named in the prompt")
    side = set(card.facets.get("side_effects", []))
    kinds = {r.split(":")[0] for r in reasons if not r.startswith("words")}
    if len(shared) >= 2 and similarity > 0.05:
        kinds.add("words")
    if (role in LAYERS or role == "subject") and not named and len(kinds) < 2:
        score *= 0.7          # layers and subjects need two kinds of evidence, not one facet or one shared word
        reasons.append("thin evidence")
    asked_periods = (own or view.whole_terms).get("period", []) if role == "style" else []
    card_periods = card.facets.get("period", [])
    if asked_periods and card_periods and not any(lists.similarity("period", a, c) for a in asked_periods for c in card_periods):
        score *= 0.75         # the prompt names a period and this style belongs to another one
        reasons.append("different period")
    if card.confidence and card.confidence < 0.5 and not card.reviewed:
        score *= 0.6
        reasons.append("low card confidence")
    if card.favorite:
        score = min(1.0, score * 1.05)
    if card.reviewed:
        score = min(1.0, score * 1.05)
    if card.rating:
        factor = RATING_FACTOR.get(card.rating, 1.0)
        score = min(1.0, score * factor)
        if factor != 1.0:
            reasons.append(f"you rated it {card.rating}/5")
    elif card.community is not None:
        score = min(1.0, score * community_factor(card))
        if card.community >= 80 or card.community <= 20:
            top = f"top {max(1, round(100 - card.community))}%" if card.community >= 50 else f"bottom {max(1, round(card.community))}%"
            reasons.append(f"Civitai {top}" + (f" of {card.community_group}" if card.community_group else ""))
    if "adds_text" in side and not view.wants_text and role in ("style", "treatment"):
        score *= 0.7
        reasons.append("adds text, prompt has none")
    if "photo_conflict" in side and view.photo:
        score *= 0.5
        reasons.append("breaks photorealism")
    return score, reasons


# ----------------------------------------------------------------------------------------------- rules
def _real_person(card: LoraCard) -> bool:
    return "real_person" in card.facets.get("figure", [])


def _youthful(card: LoraCard) -> bool:
    return "youthful_look" in card.facets.get("side_effects", [])


def gate_reason(card: LoraCard, view: PromptView, gate: int, effective: int, settings: SuggestSettings,
                exclude_terms: List[filters.Term]) -> str:
    """Why this card may not be used at all for this prompt ('' = it may)."""
    term = filters.first_match(exclude_terms, card)
    if term is not None:
        return f"excluded ({term.raw})"
    if card.exclude:
        return "excluded in LoRA Manager"
    allowed = max(effective, 1)
    if card.nsfw > allowed and not (card.nsfw == 2 and card.role in LOOK_ROLES):
        return f"nsfw {NSFW_TERMS[card.nsfw]} above {'the prompt' if effective < gate else 'the gate'}"
    side = set(card.facets.get("side_effects", []))
    if effective == 0 and "nsfw_bias" in side:
        return "pushes NSFW, prompt is SFW"
    if effective >= 2 and _youthful(card):
        return "looks young: never with nudity/explicit"
    if effective >= 2 and settings.block_real_person_nsfw and _real_person(card):
        return "real person: not with nudity/explicit (config block_real_person_nsfw)"
    if effective >= 2 and "censors_nudity" in side and card.role in ("style", "treatment"):
        return ""      # down-ranked in compose, not excluded
    return ""


def clash(new: Pick, chosen: List[Pick], settings: SuggestSettings, same_family=None) -> str:
    for other in chosen:
        if other.card.path == new.card.path:
            return "same file"
        if new.card.nsfw >= 2 and _youthful(other.card) or other.card.nsfw >= 2 and _youthful(new.card):
            return f"looks young + NSFW with {other.card.name}"
        if settings.block_real_person_nsfw and (
                new.card.nsfw >= 2 and _real_person(other.card) or other.card.nsfw >= 2 and _real_person(new.card)):
            return f"real person + NSFW with {other.card.name}"
        if new.role in LAYERS or other.role in LAYERS or new.role == "slider" or other.role == "slider":
            continue
        if same_family and new.role == other.role == "style" and same_family(new.card, other.card):
            continue          # a blend of two styles from one family: they are meant to share the look
        shared = CLASHING_OVERRIDES & set(new.card.overrides) & set(other.card.overrides)
        if shared:
            return f"both take over {', '.join(sorted(shared))} ({other.card.name})"
    return ""


def soft_overlap(new: Pick, chosen: List[Pick]) -> List[str]:
    """Looser takeovers shared with picks already made (lighting, colour): allowed, the new pick is turned down."""
    if new.role in LAYERS:
        return []
    shared = set()
    for other in chosen:
        if other.role not in LAYERS:
            shared |= SOFT_OVERRIDES & set(new.card.overrides) & set(other.card.overrides)
    return sorted(shared)


def stage_weights(pick: Pick, profile: str) -> Tuple[float, float, float]:
    weight = pick.card.weight or 1.0
    # sliders can run negative (-10..+10): keep the sign; ordinary LoRAs stay within 0.1..2.0
    typical = max(-10.0, min(-0.1, weight)) if weight < 0 else min(2.0, max(0.1, weight))
    if "overpowers" in pick.card.facets.get("side_effects", []):
        typical *= 0.8
    typical *= pick.soft
    if profile == "flat":
        return (round(typical, 2),) * 3
    shape = STAGE_PROFILE.get(pick.role, (1.0, 1.0, 1.0))
    return tuple(round(typical * part, 2) for part in shape)


def apply_budget(picks: List[Pick], budget: float) -> str:
    """Scale every pick but the strongest so the summed S1 weight fits the budget (accents shrink first)."""
    if not picks or budget <= 0:
        return ""
    total = sum(abs(p.weights[0]) for p in picks)
    if total <= budget:
        return ""
    lead = max(picks, key=lambda p: p.score)
    rest = sum(abs(p.weights[0]) for p in picks if p is not lead)
    if rest <= 0:
        return ""
    factor = max(0.35, (budget - abs(lead.weights[0])) / rest)
    for pick in picks:
        if pick is not lead:
            pick.weights = tuple(round(w * factor, 2) for w in pick.weights)
    return f"weight budget {budget:g}: S1 total was {total:.2f}, accents scaled x{factor:.2f}"


# ----------------------------------------------------------------------------------------------- families
def card_families(card: LoraCard, lists: PickLists) -> List[str]:
    """The card's own `family` terms when it has them, else the families its use / medium / rendering terms imply."""
    own = [str(f) for f in card.facets.get("family") or []]
    if own:
        return own
    found: List[str] = []
    for family, facet, terms in FAMILY_RULES:
        tree = lists.facets.get(facet)
        for term in card.facets.get(facet) or []:
            chain = {term} | set(tree.ancestors(term) if tree else [])
            if chain & set(terms) and family not in found:
                found.append(family)
    if card.role == "transform" and "surreal_transform" not in found:
        found.append("surreal_transform")
    return found


def families_compatible(a: List[str], b: List[str], friends: List[List[str]]) -> bool:
    if set(a) & set(b):
        return True
    pairs = {frozenset(pair) for pair in friends if len(pair) == 2}
    return any(frozenset((x, y)) in pairs for x in a for y in b)


# ----------------------------------------------------------------------------------------------- sliders
def _signed_number(text: str) -> Optional[float]:
    try:
        return float(text.strip().replace("+", ""))
    except ValueError:
        return None


def parse_pins(text: str) -> List[Tuple[str, Optional[float]]]:
    """pin lines -> (normalised text, value or None); kept for scripts. The engine uses filters.parse."""
    return [(term.text, term.value) for term in filters.parse(text)]


def _best_first(card: LoraCard) -> tuple:
    """Order for choosing among several matches: your stars, favourite, Civitai score (unrated only), card
    confidence, shorter name."""
    return (-(RATING_FACTOR.get(card.rating, 1.0) if card.rating else 1.0), -int(card.favorite),
            -community_factor(card), -float(card.confidence or 0), len(card.name))


def slider_useful(card: LoraCard, default_range: float) -> Tuple[float, float, str]:
    """(low, high, where from). The card's useful range (your editor weights or the slider pass) when it is believable,
    else the node's slider_range."""
    info = card.slider or {}
    useful = info.get("useful") or []
    trusted = info.get("source") == "your_edit" or float(info.get("confidence") or 0) >= 0.5
    if trusted and len(useful) == 2 and useful[0] is not None and useful[1] is not None:
        low, high = sorted((float(useful[0]), float(useful[1])))
        if high - low > 0.05:
            return low, high, "your card" if info.get("source") == "your_edit" else "slider card"
    one_sided = not str(info.get("negative_pole") or "").strip()
    return (0.0 if one_sided else -abs(default_range)), abs(default_range), "node slider_range"


def slider_weight(card: LoraCard, axis_ends: Tuple[str, str], position: float, default_range: float,
                  exact: Optional[Dict[str, float]] = None, phrase: str = "") -> Tuple[Optional[float], str]:
    """Signed dial weight for an axis position (-1..1), or (None, why not). `exact` = your calibrated weights for this
    slider (axes.yaml sliders:); a phrase found there gives exactly that weight."""
    if exact and phrase in exact:
        return round(exact[phrase], 2), "your exact weight"
    info = card.slider or {}
    toward = str(info.get("positive_toward") or "").lower()
    flip = bool(toward) and toward == axis_ends[0].lower() and toward != axis_ends[1].lower()
    signed = -position if flip else position
    low, high, source = slider_useful(card, default_range)
    if signed > 0:
        weight = signed * high
    else:
        if low >= 0:
            return None, "one-sided slider, the prompt asks for the other end"
        weight = abs(signed) * low
    note = f"{source} {low:g}..{high:g}" + (", trained backwards" if flip else "")
    return round(weight, 2), note


def slider_stage_weights(dial: float, stages: str) -> Tuple[float, float, float]:
    shape = SLIDER_STAGES.get(stages or "all", SLIDER_STAGES["all"])
    return tuple(round(dial * part, 2) for part in shape)


# ----------------------------------------------------------------------------------------------- top level
def _lines(text: str) -> List[str]:
    return [line.strip().lower() for chunk in (text or "").splitlines() for line in chunk.split(",") if line.strip()]


def suggest(request: SuggestRequest, settings: Optional[SuggestSettings] = None) -> SuggestResult:
    settings = settings or load_settings()
    lists = load_pick_lists(settings.vocab)
    if lists is None:
        return SuggestResult([], f"pick lists not found at {settings.vocab} (set lora_suggest.vocab in config.json)", [], "")
    shelves, bases = settings.platform_filter(request.platform)
    if not shelves and not bases:
        return SuggestResult([], f"no LoRA shelf is known for platform '{request.platform}' "
                                 "(add it under lora_suggest.platforms in config.json)", [], "")
    cards, note = load_cards(settings.catalog, shelves, bases, settings.known_shelves())
    view = read_prompt(lists, request.sections, request.text)
    lines: List[str] = []
    gate = NSFW_LEVELS.index(request.nsfw) if request.nsfw in NSFW_LEVELS else 0
    effective = min(view.level, gate)
    composer = ""
    if request.prompt_nsfw is not None:
        composer_level = max(0, min(3, int(request.prompt_nsfw)))
        effective = min(effective, composer_level)
        composer = f", Composer {NSFW_LEVELS[composer_level]}"
    lines.append(f"platform {request.platform}: {len(cards)} carded LoRAs | prompt asks nsfw "
                 f"{NSFW_TERMS[view.level]}, gate {request.nsfw}{composer} -> using {NSFW_TERMS[effective]}")
    if note:
        lines.append("note: " + note)
    if not cards:
        return SuggestResult([], "\n".join(lines), [], view.whole)

    idf = _idf(cards)
    distinctive_cut = math.log(1 + max(1, len(cards)) / 3)
    excluded_words = filters.parse(request.exclude)       # filters.Term list (name kept for the call sites)
    rng = random.Random(int(request.seed))
    excluded: Dict[str, int] = {}
    candidates: List[Tuple[float, Pick]] = []
    for card in cards:
        reason = gate_reason(card, view, gate, effective, settings, excluded_words)
        if reason:
            key = reason if reason.startswith("excluded (") else reason.split(":")[0].split("(")[0].strip()
            excluded[key] = excluded.get(key, 0) + 1
            continue
        best = None                      # a card with several roles competes in the slot where it scores best
        for role in (card.roles or [card.role]):
            if role == "slider" or request.role_max.get(role, 0) <= 0:
                continue                 # sliders are dialled from the prompt's axes, never scored on text
            variant = card if role == card.role else dataclasses.replace(card, role=role)
            role_score, role_reasons = score_card(variant, view, lists, idf, distinctive_cut)
            if best is None or role_score > best[0]:
                best = (role_score, role_reasons, variant)
        if best is None:
            continue
        score, reasons, card = best
        if card.nsfw == 2 and effective < 2:
            score *= 0.8
            reasons.append("may add nudity")
        if effective >= 2 and "censors_nudity" in card.facets.get("side_effects", []):
            score *= 0.5
            reasons.append("covers up nudity")
        if score <= 0:
            continue
        # variety 0.2 lets picks within ~0.1 of each other trade places from seed to seed; 1.0 within ~0.5
        jitter = rng.random() * max(0.0, request.variety) * 0.5
        candidates.append((score + jitter, Pick(card, card.role, score, reasons)))
    candidates.sort(key=lambda item: -item[0])

    family_cache: Dict[str, List[str]] = {}

    def families_of(card: LoraCard) -> List[str]:
        if card.path not in family_cache:
            family_cache[card.path] = card_families(card, lists)
        return family_cache[card.path]

    def same_family(a: LoraCard, b: LoraCard) -> bool:
        return families_compatible(families_of(a), families_of(b), settings.family_friends)

    chosen: List[Pick] = []
    per_role: Dict[str, int] = {}
    pin_lines: List[str] = []
    for term in filters.parse(request.pin):             # pins first, whatever their score
        if term.bad_value:
            pin_lines.append(f"  pin '{term.raw}': the part after = is not a number - ignored")
            continue
        if term.kind in ("folder", "role"):
            pin_lines.append(f"  pin '{term.raw}': {term.kind}: works in exclude only (a pin needs one LoRA: use its name "
                             "or axis:<axis>=<value>) - ignored")
            continue
        if term.kind == "axis" and term.value is None:
            pin_lines.append(f"  pin '{term.raw}': an axis pin needs a value, e.g. {term.raw}=+2 - ignored")
            continue
        found = sorted([c for c in cards if filters.matches(term, c) and all(p.card.path != c.path for p in chosen)],
                       key=_best_first)
        if not found:
            pin_lines.append(f"  pin '{term.raw}': no {request.platform} LoRA matches it (whole words)")
            continue
        card = found[0]
        role = "slider" if card.slider else card.role
        label = f"pinned ({term.raw})"
        chosen.append(Pick(card, role, 1.0, [label], pinned=True, dial=term.value))
        per_role[role] = per_role.get(role, 0) + 1
        if len(found) > 1:
            others = ", ".join(c.name for c in found[1:5]) + (" ..." if len(found) > 5 else "")
            pin_lines.append(f"  pin '{term.raw}' took {card.name}; it also matched {len(found) - 1}: {others}")
        if card.slider and term.value is None:
            pin_lines.append(f"  pin '{term.raw}': a slider pinned without a value runs at its typical weight "
                             f"({card.weight:g}); add =<weight>")
    runners: Dict[str, Tuple[Pick, str]] = {}
    for _order, pick in candidates:
        if len([p for p in chosen if p.role != "slider"]) >= request.max_loras:
            break
        if pick.score < request.threshold:
            continue
        if per_role.get(pick.role, 0) >= request.role_max.get(pick.role, 0):
            runners.setdefault(pick.role, (pick, "role full"))
            continue
        if pick.role == "style":
            lead = [p for p in chosen if p.role == "style"]
            if lead and not all(same_family(pick.card, p.card) for p in lead):
                runners.setdefault("style", (pick, f"different family from {lead[0].card.name} "
                                                   f"({'/'.join(families_of(pick.card)) or 'none'} vs "
                                                   f"{'/'.join(families_of(lead[0].card)) or 'none'})"))
                continue
            if lead:
                pick.soft *= settings.blend_ratio
                pick.reasons.append(f"blend with {lead[0].card.name} ({'/'.join(families_of(pick.card))}): "
                                    f"weight x{settings.blend_ratio:g}")
        why = clash(pick, chosen, settings, same_family)
        if why:
            runners.setdefault(pick.role, (pick, why))
            continue
        overlap = soft_overlap(pick, chosen)
        if overlap:
            pick.soft = 0.8
            pick.reasons.append(f"shares {', '.join(overlap)} with an earlier pick: weight x0.8")
        chosen.append(pick)
        per_role[pick.role] = per_role.get(pick.role, 0) + 1

    # Sliders: dial every axis the prompt sets (strong enough), with the best slider card for that axis.
    axes_file = load_axes_file(settings.axes)
    axes = axes_file.axes
    axis_values = read_axes(axes, view.whole) if axes else {}
    slider_lines: List[str] = [f"  axes.yaml: {problem}" for problem in axes_file.problems[:5]]
    min_position = MIN_POSITION.get(int(request.slider_min_strength), 0.35)
    dialled = {p.card.slider.get("axis") for p in chosen if p.role == "slider" and p.card.slider}
    slider_cards = [c for c in cards if c.role == "slider" and c.slider]
    for axis_name, value in sorted(axis_values.items()):
        if value.note:
            slider_lines.append(f"  axis {axis_name}: left alone, {value.note} ({value.phrase})")
            continue
        if axis_name in dialled or per_role.get("slider", 0) >= request.role_max.get("slider", 0):
            continue
        if abs(value.position) < min_position:
            slider_lines.append(f"  axis {axis_name} {value.position:+.2f} ('{value.phrase}'): below slider_min_strength, "
                                "left to the base model")
            continue
        if axis_name == "age" and value.position < 0 and effective >= 2:
            slider_lines.append(f"  axis age {value.position:+.2f}: no younger-age slider with nudity/explicit (hard rule)")
            continue
        offered = [c for c in slider_cards if c.slider.get("axis") == axis_name
                   and not gate_reason(c, view, gate, effective, settings, excluded_words)]
        if not offered:
            blocked = [(c, gate_reason(c, view, gate, effective, settings, excluded_words))
                       for c in slider_cards if c.slider.get("axis") == axis_name]
            if blocked:
                why = "; ".join(f"{c.name}: {reason}" for c, reason in blocked[:3])
                slider_lines.append(f"  axis {axis_name} {value.position:+.2f} ('{value.phrase}'): its sliders are held back - {why}")
            else:
                slider_lines.append(f"  axis {axis_name} {value.position:+.2f} ('{value.phrase}'): no slider for it on this platform")
            continue
        def calibrated(card: LoraCard) -> Optional[Dict[str, float]]:
            return exact_weights_for(axes_file.sliders, card.name, card.file_name, axis_name)

        # a slider you calibrated for this phrase first, then your stars, card confidence, favourite
        offered.sort(key=lambda c: (-int(bool((calibrated(c) or {}).get(value.phrase) is not None)),
                                    -(RATING_FACTOR.get(c.rating, 1.0) if c.rating else 1.0),
                                    -float(c.slider.get("confidence") or 0), -int(c.favorite),
                                    -community_factor(c)))
        axis = axes[axis_name]
        for card in offered:
            dial, note = slider_weight(card, axis.ends, value.position, request.slider_range, calibrated(card), value.phrase)
            if dial is None:
                continue
            pick = Pick(card, "slider", 1.0, [f"axis {axis_name} {value.position:+.2f} from '{value.phrase}'", note], dial=dial)
            if clash(pick, chosen, settings):
                continue
            chosen.append(pick)
            per_role["slider"] = per_role.get("slider", 0) + 1
            dialled.add(axis_name)
            break
        else:
            slider_lines.append(f"  axis {axis_name} {value.position:+.2f}: its sliders cannot go that way")

    order = {role: index for index, role in enumerate(("subject", "style", "transform", "environment", "lighting",
                                                       "layout", "treatment", "utility", "detailer", "slider"))}
    chosen.sort(key=lambda p: (order.get(p.role, 9), -p.score))
    for pick in chosen:
        if pick.dial is not None:
            stages = (pick.card.slider or {}).get("stages") or (axes.get((pick.card.slider or {}).get("axis", "")).stages
                                                                 if (pick.card.slider or {}).get("axis") in axes else "all")
            pick.weights = slider_stage_weights(pick.dial, stages) if pick.role == "slider" else (pick.dial,) * 3
        else:
            pick.weights = stage_weights(pick, request.stage_profile)
    budget_note = apply_budget([p for p in chosen if p.dial is None], request.weight_budget)

    triggers: List[str] = []
    for pick in chosen:
        if pick.role == "slider":
            continue          # a dial works by weight; its 'trigger' is usually just its name
        for trigger in pick.card.triggers[:2]:
            if trigger.lower() not in {t.lower() for t in triggers}:
                triggers.append(trigger)

    for pick in chosen:
        w = pick.weights
        lines.append(f"+ {pick.role:<11} {pick.card.name}  [{pick.score:.2f}]  S1 {w[0]} / S2 {w[1]} / S3 {w[2]}")
        lines.append(f"      {'; '.join(pick.reasons[:5]) or '-'}")
        lines.append(f"      {pick.card.lora_name}" + (f"   triggers: {', '.join(pick.card.triggers[:2])}" if pick.card.triggers else ""))
    if not chosen:
        lines.append(f"no LoRA reached the threshold {request.threshold:g}")
    if axis_values:
        lines.append("axes read: " + ", ".join(f"{name} {v.position:+.2f} ('{v.phrase}')" if not v.note else f"{name} ?"
                                              for name, v in sorted(axis_values.items())))
    lines.extend(slider_lines)
    lines.extend(pin_lines)
    best_per_role: Dict[str, Pick] = {}
    for _order, pick in candidates:
        best_per_role.setdefault(pick.role, pick)
    for role in ROLES:
        if request.role_max.get(role, 0) <= 0 or any(p.role == role for p in chosen):
            continue
        best = best_per_role.get(role)
        if best:
            lines.append(f"- {role:<11} best was {best.card.name} [{best.score:.2f}]"
                         + (" below threshold" if best.score < request.threshold else ""))
    for role, (pick, why) in runners.items():
        lines.append(f"  runner-up {role}: {pick.card.name} [{pick.score:.2f}] - {why}")
    if budget_note:
        lines.append(budget_note)
    if excluded:
        lines.append("excluded: " + ", ".join(f"{k} {v}" for k, v in sorted(excluded.items(), key=lambda kv: -kv[1])))
    prompt = view.whole
    return SuggestResult(chosen, "\n".join(lines), triggers, prompt)
