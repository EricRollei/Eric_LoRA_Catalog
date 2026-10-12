"""
Phase 3: the card writer. One vision-LLM call per LoRA turns its evidence into a card - what it DOES:

    kind          style_medium | photo_look | character | person | concept_object | clothing | pose_action | setting |
                  lighting_colour | effect_detail | quality_realism | tool_edit | motion | speed_distill | other
    scope         global (changes the whole look) | local (adds or changes one element)
    effect        one or two sentences in plain visual terms
    artists       [{name, basis}] the artist, studio or illustrator it emulates. basis = named_in_evidence (the name or
                  description says so), name_match (a candidate from the wildcard artist lists that the model
                  confirmed against the description and images), recognised_style (the model recognised the style
                  from the images alone - the weakest, shown as such on the review page)
    medium        what the pictures look made with (oil paint, ink and wash, gouache, digital painting, 3D render ...)
    movement_era  the movement, period or tradition (1970s fantasy paperback art, ukiyo-e, art nouveau ...)
    palette       the colour scheme in a few words
    line_quality  how edges and lines are drawn (bold even outlines, scratchy hatching, no lines - painterly)
    suited_subjects  [what it is good at] (barbarian warriors, landscapes, pin-up portraits ...)
    triggers      [{text, use: required | optional | name_token, placement: prefix | inline}]
    weights       typical, low, high
    overrides     what it tends to take over: lighting, colour, composition, faces, anatomy, text, background, texture,
                  style, camera
    nsfw_level    0 none, 1 suggestive, 2 nudity, 3 explicit
    confidence    0-1, calibrated against how much evidence agrees

Evidence given to the model, in this order of trust: your training config and captions > gallery prompts and the
weights they used > Civitai name, tags, trained words and description > artist names matched from the wildcard
artist lists > header dataset tags > the preview images. Trigger words are only ever copied from the evidence, never
invented; an artist with basis name_match must be one of the candidates.

Card pass v2 (2026-10-10, LORA_STUDIO_SPEC.md 5.2): when `run.py community` has collected the LoRA, a separate
WHAT CIVITAI USERS MAKE WITH IT block is added (the creator's examples are intent, this is reality): the content
levels of its newest community images, the share of mature / explicit ones not asked for in the prompt, the weights
people use, and their most-liked prompts. It is kept whole (the rest of the evidence is cut to fit around it). With 10
or more community weights, those replace the model's weight_typical / low / high (median, 25th, 75th percentile).
`run.py cards --show-evidence --only <folder> --limit 3` prints the evidence without calling the model.

LM Studio's structured output (a JSON schema) keeps the answer parseable.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .artists import fold, get_index
from .config import Settings
from .llm import LLMTimeout, backend, ollama_check, timeout_hint
from .local_sources import read_json
from .store import Catalog
from .vocab import load_vocab


def _labels(facet: str) -> str:
    """The pick-list words of one vocab.yaml facet, for the instructions (genre, mood)."""
    try:
        return ", ".join(t.label for t in load_vocab().facets[facet].terms.values())
    except Exception:
        return ""

KINDS = ["style_medium", "photo_look", "character", "person", "concept_object", "clothing", "pose_action", "setting",
         "lighting_colour", "effect_detail", "quality_realism", "tool_edit", "motion", "speed_distill", "other"]
OVERRIDES = ["lighting", "colour", "composition", "faces", "anatomy", "text", "background", "texture", "style", "camera"]
ARTIST_BASIS = ["named_in_evidence", "name_match", "recognised_style"]
IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp")
MAX_EVIDENCE_CHARS = 7000      # with 2 images this stays inside an 8192-token context
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
VIDEO_EXTENSIONS = (".mp4", ".webm")         # video LoRAs (Wan, LTX, MiniMax) often have only video previews
IMAGE_MAX_SIDE = 640

SYSTEM = f"""You catalogue LoRA adapters for an image model, so that a tool can later pick the right one for a prompt.
You are given everything known about one LoRA: its Civitai name, tags, trigger words and description, possible
artists matched from its name, the dataset tags in its file header, example prompts people used it with (and at what
weight), and up to two example images it made. Work out what the LoRA DOES to a picture.

Return:
- kind: the one that fits best. style_medium = an art style, artist or medium (watercolour, comic, ukiyo-e);
  photo_look = a photographic look (film stock, lens, era of photography); character = a named fictional character;
  person = a real person's likeness; concept_object = a thing, creature or concept it adds; clothing; pose_action;
  setting = a place or environment; lighting_colour = a lighting or colour treatment; effect_detail = an effect or
  detail booster (glow, glitch, sharper detail); quality_realism = general quality or realism; tool_edit = a utility
  (relight, remove, next scene, upscale-like); motion = video motion; speed_distill = fewer steps; other.
- scope: global if it changes the look of the whole picture, local if it adds or changes one element.
- effect: one or two sentences describing what it does, in plain visual terms (palette, line, texture, subject).
  No marketing words.
- artists: the artist, illustrator, photographer or studio whose work it emulates, if any. LoRAs are often named
  with a misspelled or shortened artist name ("VALEJO" is Boris Vallejo, "KreaDarrow" is Geof Darrow), so read the
  name with the possible artists you are given and with what you know of illustrators and painters. basis =
  named_in_evidence when the name, description or tags name the artist; name_match when you confirm one of the
  POSSIBLE ARTISTS given (copy that name exactly); recognised_style when you recognise the artist from the images
  and description alone. Reject a possible artist whose work does not fit the description and images. [] when it
  is a generic style that no single artist owns.
- medium: what the pictures look made with, in a few words (oil paint, ink and wash, gouache, airbrush, pencil,
  digital painting, cel shading, 3D render, collage, photograph ...).
- movement_era: the movement, period or tradition it belongs to, in a few words (1970s fantasy paperback cover art,
  ukiyo-e, art nouveau, Franco-Belgian bande dessinee, 1990s anime ...). "" if none fits.
- palette: the colour scheme in a few words (earthy warm with deep shadows, pastel, limited red and black ...).
- line_quality: how lines and edges are drawn (bold even outlines, fine scratchy hatching, no outlines - painterly
  edges ...).
- suited_subjects: two to five things it is good at depicting, from the examples and the description.
- triggers: the trigger words or phrases EXACTLY as they appear in the evidence (trained words first). use = required
  if the description or the examples show it is needed, optional if it only strengthens, name_token if it is a coined
  token like "YFGRene style". placement = prefix if the examples put it first. Never invent a trigger; [] if none.
- weight_typical, weight_low, weight_high: from the example weights and the description; 1.0, 0.6, 1.2 if unknown.
- overrides: what it tends to take over in a picture: {", ".join(OVERRIDES)}. [] if it is subtle.
- nsfw_level: 0 none, 1 suggestive, 2 nudity, 3 explicit - judged from its purpose, not from one example. When
  WHAT CIVITAI USERS MAKE WITH IT is given, use it as a check: nude or explicit images whose prompts ASKED for nudity
  say what people chose to make, not what the LoRA does; a LoRA that is not about nudity stays 0-1 even if many of its
  community images are explicit. Rate it 2-3 when its purpose (name, description, trained words) is nudity or sex, or
  when a large share of its mature / explicit images were NOT asked for in the prompt (it adds nudity by itself).
- When WHAT CIVITAI USERS MAKE WITH IT is given, its prompts show the subjects and uses people really choose - use them
  for suited_subjects and effect, alongside the creator's own description and examples.
- confidence 0-1: 0.9 or more only when the name, the description and the images all agree; about 0.7 when one of
  them is missing or vague; 0.5 or less when you are mostly inferring from the images.
- evidence_used: a few words naming what you relied on."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "card", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "scope": {"type": "string", "enum": ["global", "local"]},
            "effect": {"type": "string"},
            "artists": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "basis": {"type": "string", "enum": ARTIST_BASIS}},
                "required": ["name", "basis"]}},
            "medium": {"type": "string"},
            "movement_era": {"type": "string"},
            "palette": {"type": "string"},
            "line_quality": {"type": "string"},
            "suited_subjects": {"type": "array", "items": {"type": "string"}},
            "triggers": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "use": {"type": "string", "enum": ["required", "optional", "name_token"]},
                "placement": {"type": "string", "enum": ["prefix", "inline"]}}, "required": ["text", "use", "placement"]}},
            "weight_typical": {"type": "number"}, "weight_low": {"type": "number"}, "weight_high": {"type": "number"},
            "overrides": {"type": "array", "items": {"type": "string", "enum": OVERRIDES}},
            "nsfw_level": {"type": "integer", "minimum": 0, "maximum": 3},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_used": {"type": "string"},
        },
        "required": ["kind", "scope", "effect", "artists", "medium", "movement_era", "palette", "line_quality",
                     "suited_subjects", "triggers", "weight_typical", "weight_low", "weight_high", "overrides",
                     "nsfw_level", "confidence", "evidence_used"],
    }},
}


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def preview_images(path: str, count: int) -> List[str]:
    """Up to `count` example images for the LoRA, the least NSFW first (Civitai nsfwLevel from the sidecar; LoRA
    Manager's <stem>.preview.* counts as safe). Paths only."""
    folder = os.path.dirname(path)
    stem = os.path.splitext(path)[0]
    own_prefix = os.path.basename(stem).lower()
    # A downloader folder holds one LoRA and its gallery; a flat folder holds many LoRAs, and then only images
    # named after this LoRA are its own.
    shared = (sum(1 for name in os.listdir(folder) if name.lower().endswith(MODEL_EXTENSIONS)) > 1
              and not re.match(r"^\d+-", os.path.basename(folder)))     # a downloader version folder is one model

    def mine(candidate: str) -> bool:
        return not shared or os.path.basename(candidate).lower().startswith(own_prefix)

    found = []
    for candidate in glob.glob(glob.escape(stem) + ".preview.*"):
        if candidate.lower().endswith(IMAGE_EXTENSIONS):
            found.append((0, candidate))
    for pattern in ("*", os.path.join("images", "*")):
        for candidate in glob.glob(os.path.join(glob.escape(folder), pattern)):
            if not candidate.lower().endswith(IMAGE_EXTENSIONS) or ".preview." in candidate or not mine(candidate):
                continue
            meta = read_json(candidate + ".json")
            level = meta.get("nsfwLevel", 1) if isinstance(meta, dict) else 1
            found.append((level if isinstance(level, int) else 1, candidate))
    found.sort(key=lambda item: (item[0], item[1]))
    chosen = [candidate for _level, candidate in found[:count]]
    if len(chosen) < count:      # no stills: a frame from a preview video stands in (see _data_uri)
        videos = []
        for pattern in ("*", os.path.join("images", "*")):
            for candidate in glob.glob(os.path.join(glob.escape(folder), pattern)):
                if candidate.lower().endswith(VIDEO_EXTENSIONS) and mine(candidate):
                    meta = read_json(candidate + ".json")
                    level = meta.get("nsfwLevel", 1) if isinstance(meta, dict) else 1
                    videos.append((level if isinstance(level, int) else 1, candidate))
        videos.sort(key=lambda item: (item[0], item[1]))
        chosen += [candidate for _level, candidate in videos[:count - len(chosen)]]
    return chosen


def _video_frame(video_path: str):
    """The middle frame of a preview video as a PIL image, or None (OpenCV ships with ComfyUI)."""
    try:
        import cv2
        from PIL import Image
        capture = cv2.VideoCapture(video_path)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames > 2:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
        ok, frame = capture.read()
        capture.release()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None
    except Exception:
        return None


def _data_uri(image_path: str) -> Optional[str]:
    try:
        from PIL import Image
        if image_path.lower().endswith(VIDEO_EXTENSIONS):
            image = _video_frame(image_path)
            if image is None:
                return None
        else:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:      # an unreadable image is simply not shown
        return None


def artist_candidates(row: Any, settings: Settings) -> List[Dict[str, Any]]:
    index = get_index(settings.artist_names_file)
    return index.match(row["model_name"] or "", row["stem"] or "", " ".join(_json_list(row["trained_words"])))


def evidence_text(row: Any, examples: List[Any], candidates: Optional[List[Dict[str, Any]]] = None) -> str:
    parts = [f"File: {row['file_name']}  (shelf: {row['rel_dir']})"]
    if row["model_name"]:
        parts.append(f"Civitai name: {row['model_name']} - {row['version_name'] or ''}  (base model: {row['base_model'] or '?'})")
    if candidates:
        how = {"full name": "first and last name in the LoRA's name", "surname": "surname in the LoRA's name",
               "near": "the LoRA's name is one letter away from the surname"}
        parts.append("POSSIBLE ARTISTS, matched by name only against a list of known artists - confirm or reject:\n"
                     + "\n".join(f"  - {item['name']}  ({how.get(item['how'], item['how'])}: '{item['token']}')"
                                 for item in candidates))
    tags = _json_list(row["tags"])
    if tags:
        parts.append("Civitai tags: " + ", ".join(tags[:15]))
    words = _json_list(row["trained_words"])
    if words:
        parts.append("Trained words (trigger words): " + " | ".join(words[:10]))
    if row["training_trigger"]:
        parts.append(f"Trigger word in YOUR training config: {row['training_trigger']}  (arch {row['training_arch']})")
    captions = _json_list(row["caption_samples"])
    if captions:
        parts.append("Training captions (samples):\n" + "\n".join(f"  - {caption[:200]}" for caption in captions[:6]))
    if row["description"]:
        parts.append("Description:\n" + row["description"][:1500])
    ss_tags = _json_list(row["ss_tags"])
    if ss_tags:
        tags = ", ".join(str(tag)[:60] for tag in ss_tags[:15])      # some headers hold whole captions as tags
        parts.append("Dataset tags in the file header (most frequent first): " + tags[:1200])
    if examples:
        lines = []
        for example in examples[:4]:
            weight = f"weight {example['weight']:g}" if example["weight"] is not None else "weight unknown"
            lines.append(f"  - ({weight}) {example['prompt'][:300]}")
        parts.append("Example prompts used with it:\n" + "\n".join(lines))
    return "\n\n".join(parts)[:MAX_EVIDENCE_CHARS]


def community_block(catalog: Any, version_id: Optional[int]) -> Dict[str, Any]:
    """{'text': the WHAT CIVITAI USERS MAKE WITH IT evidence ('' when not collected), 'weights': (p25, median, p75) when
    10+ community images say, else None}."""
    out: Dict[str, Any] = {"text": "", "weights": None}
    if not version_id:
        return out
    try:
        stats = catalog.query("select * from community_stats where version_id = ? and status = 'ok'", (version_id,))
        prompts = catalog.query("select cleaned, nsfw_level, weight, reactions from community_prompts where version_id = ? "
                                "and source = 'top' order by reactions desc limit 12", (version_id,))
    except Exception:            # the community tables do not exist yet
        return out
    if not stats:
        return out
    s = stats[0]
    lines = [f"WHAT CIVITAI USERS MAKE WITH IT ({s['distinct_posters']} different people in its {s['images_sampled']} "
             f"newest community images; {s['downloads']} downloads):"]
    from .community import peer_text
    keys = s.keys()
    facts = [f"{s['thumbs_up']} likes"]
    if "images_per_day" in keys and s["images_per_day"] is not None:
        facts.append(f"{s['images_exact']} community images in all" if s["images_exact"] is not None else
                     "200+ community images")
        facts.append(f"{s['images_per_day']:g} posted per day lately")
    if "top_reactions_median" in keys and s["top_reactions_median"] is not None:
        facts.append(f"median {s['top_reactions_median']:g} reactions on its {s['top_sampled']} most-liked images")
    lines.append("  popularity (context only - it does not say what the LoRA does): " + ", ".join(facts))
    peers = peer_text(s) if "peer" in keys else ""
    if peers:
        lines.append(f"  compared with the others: {peers}")
    try:
        share = json.loads(s["nsfw_share"] or "{}")
    except ValueError:
        share = {}
    if share:
        lines.append("  content levels: " + ", ".join(f"{level} {round(value * 100)}%" for level, value in share.items() if value))
    if s["explicit_unasked"] is not None:
        lines.append(f"  of its mature / explicit images, {round(s['explicit_unasked'] * 100)}% had no nudity or sex words in "
                     "the prompt")
    if s["weight_n"]:
        lines.append(f"  weights people use: median {s['weight_median']:g}, middle half {s['weight_p25']:g} to "
                     f"{s['weight_p75']:g} ({s['weight_n']} images)")
        if s["weight_n"] >= 10:
            out["weights"] = (s["weight_p25"], s["weight_median"], s["weight_p75"])
    from .community import _similar
    distinct: List[Any] = []
    for p in prompts:                    # the same prompt posted twice is one prompt
        if not any(_similar(p["cleaned"], q["cleaned"]) for q in distinct):
            distinct.append(p)
    if distinct:
        lines.append("  their most-liked prompts:")
        for p in distinct[:4]:
            weight = f", weight {p['weight']:g}" if p["weight"] is not None else ""
            lines.append(f"  - ({p['nsfw_level']}{weight}) {str(p['cleaned'] or '').replace(chr(10), ' ')[:220]}")
    out["text"] = "\n".join(lines)
    return out


def apply_community_weights(card: Dict[str, Any], weights: tuple) -> None:
    """typical = what users use most (median). low / high = their 25th / 75th percentile - unless users all sit on one
    value (0.9 / 1.0 / 1.0), then that side keeps the model's guess, so the card still has a range to work in."""
    p25, median, p75 = weights
    card["weight_typical"] = median
    if p25 is not None and p25 < median:
        card["weight_low"] = p25
    elif card.get("weight_low") is None or card["weight_low"] >= median:
        card["weight_low"] = round(median * 0.7, 2)
    if p75 is not None and p75 > median:
        card["weight_high"] = p75
    elif card.get("weight_high") is None or card["weight_high"] <= median:
        card["weight_high"] = round(median * 1.25, 2)


def call_llm(settings: Settings, text: str, images: List[str]) -> Dict[str, Any]:
    from .llm import chat_json
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        uri = _data_uri(image)
        if uri:
            content.append({"type": "image_url", "image_url": {"url": uri}})
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    return chat_json(settings, settings.card_model, messages, SCHEMA, temperature=0.1, max_tokens=1300, timeout=300)


def check_artists(artists: List[Dict[str, Any]], evidence: str, candidates: List[Dict[str, Any]]) -> tuple:
    """(kept, dropped). named_in_evidence needs the surname in the evidence; name_match must be a candidate
    (matched by its folded name, and the candidate's spelling is used); recognised_style is kept as the model gave it."""
    folded_evidence = fold(evidence)
    by_folded = {fold(item["name"]): item["name"] for item in candidates}
    kept, dropped = [], []
    for artist in artists or []:
        name = str(artist.get("name") or "").strip()
        basis = artist.get("basis")
        if not name:
            continue
        surname = fold(name).split()[-1] if fold(name) else ""
        if basis == "name_match":
            if fold(name) in by_folded:
                kept.append({"name": by_folded[fold(name)], "basis": basis})
            elif surname and any(fold(candidate).split()[-1] == surname for candidate in by_folded):
                kept.append({"name": next(c for f, c in by_folded.items() if f.split()[-1] == surname), "basis": basis})
            else:
                dropped.append(name)
        elif basis == "named_in_evidence":
            if surname and surname in folded_evidence:
                kept.append({"name": name, "basis": basis})
            else:
                kept.append({"name": name, "basis": "recognised_style"})    # said to be named, but it is not
        else:
            kept.append({"name": name, "basis": "recognised_style"})
    return kept, dropped


def write_cards(settings: Settings, only: str = "", limit: int = 0, force: bool = False, workers: int = 1,
                paths_file: str = "", show_evidence: bool = False) -> Dict[str, int]:
    """workers > 1 sends that many LLM requests at once (LM Studio serves them in parallel); results are written by
    this thread only, so the catalog sees one writer."""
    catalog = Catalog(settings.catalog_path)
    where, params = "1=1", []
    if only:
        where, params = "rel_dir like ?", [only.replace("\\", "/") + "%"]
    rows = catalog.query(f"select * from loras where {where} order by rel_dir, file_name", tuple(params))
    done = {row["path"] for row in catalog.query("select path from cards")}
    todo = [row for row in rows if force or row["path"] not in done]
    if paths_file:                       # only these LoRAs (one path per line)
        with open(paths_file, "r", encoding="utf-8") as handle:
            wanted = {line.strip() for line in handle if line.strip()}
        todo = [row for row in todo if row["path"] in wanted]
    print(f"[cards] {len(todo)} LoRAs to card with {settings.card_model} ({len(rows) - len(todo)} already done)", flush=True)
    counts = {"written": 0, "failed": 0}
    started = time.time()
    if limit:
        todo = todo[:limit]
    community_weights: Dict[str, Any] = {}

    def prepare(row):
        examples = catalog.query("select * from examples where path=? order by (weight is null), media_file", (row["path"],))
        images = preview_images(row["path"], settings.card_images)
        candidates = artist_candidates(row, settings)
        block = community_block(catalog, row["civitai_version_id"])
        community_weights[row["path"]] = block["weights"]
        text = evidence_text(row, examples, candidates)
        if block["text"]:                # kept whole: the rest of the evidence is cut to fit around it
            text = text[:max(1500, MAX_EVIDENCE_CHARS - len(block["text"]) - 2)] + "\n\n" + block["text"]
        return row, images, candidates, text

    if show_evidence:                    # print what the model would get, call nothing
        for row in todo:
            _row, images, _candidates, text = prepare(row)
            print(f"\n===== {row['path']}\n images: {images}\n weights from community: "
                  f"{community_weights.get(row['path'])}\n\n{text}")
        catalog.close()
        return {"shown": len(todo)}

    if todo and backend(settings) == "ollama":       # warn before the first card, not after 300 s of silence
        for note in ollama_check(settings, settings.card_model):
            print(f"[cards] warning: {note}", flush=True)
    hinted = False

    def results():
        """(row, images, candidates, text, card or error) in completion order, keeping `workers` requests in flight."""
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        pending = iter(todo)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            flying = {}
            for _ in range(max(1, workers)):
                row = next(pending, None)
                if row is None:
                    break
                job = prepare(row)
                flying[pool.submit(call_llm, settings, job[3], job[1])] = job
            while flying:
                finished, _ = wait(flying, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = flying.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as error:
                        outcome = error
                    row = next(pending, None)
                    if row is not None:
                        nxt = prepare(row)
                        flying[pool.submit(call_llm, settings, nxt[3], nxt[1])] = nxt
                    yield job + (outcome,)

    for number, (row, images, candidates, text, card) in enumerate(results(), 1):
        if isinstance(card, Exception):
            counts["failed"] += 1
            print(f"[cards] failed {row['file_name']}: {str(card)[:120]}", flush=True)
            if isinstance(card, LLMTimeout) and not hinted:
                print(timeout_hint(settings, workers), flush=True)
                hinted = True
            continue
        # A trigger the model made up is dropped: it must appear somewhere in the evidence.
        lowered = text.lower()
        triggers = [trigger for trigger in card.get("triggers") or [] if str(trigger.get("text", "")).strip().lower() in lowered]
        artists, dropped_artists = check_artists(card.get("artists") or [], text, candidates)
        weights_from = "model"
        if community_weights.get(row["path"]):       # 10+ community images say: they replace the model's guess
            apply_community_weights(card, community_weights[row["path"]])
            weights_from = "community"
        catalog.connection.execute(
            """insert or replace into cards (path, kind, scope, effect, triggers, weight_typical, weight_low, weight_high,
                                             overrides, nsfw_level, confidence, evidence, refined_by, refined_at,
                                             artists, medium, movement_era, palette, line_quality, suited_subjects,
                                             genre, mood)
               values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row["path"], card.get("kind"), card.get("scope"), card.get("effect"), json.dumps(triggers, ensure_ascii=False),
             card.get("weight_typical"), card.get("weight_low"), card.get("weight_high"),
             json.dumps(card.get("overrides") or []), card.get("nsfw_level"), card.get("confidence"),
             json.dumps({"used": card.get("evidence_used"),
                         "images": [os.path.relpath(image, os.path.dirname(row["path"])) for image in images],
                         "artist_candidates": candidates, "dropped_artists": dropped_artists,
                         "dropped_triggers": [t.get("text") for t in card.get("triggers") or [] if t not in triggers],
                         "weights_from": weights_from, "community": "WHAT CIVITAI USERS" in text},
                        ensure_ascii=False),
             settings.card_model, time.strftime("%Y-%m-%d %H:%M"),
             json.dumps(artists, ensure_ascii=False), card.get("medium"), card.get("movement_era"), card.get("palette"),
             card.get("line_quality"), json.dumps(card.get("suited_subjects") or [], ensure_ascii=False),
             None, None))      # genre / mood: NULL = not classified yet - run.py genre-mood does them (27B model)
        counts["written"] += 1
        if number % 10 == 0:
            catalog.commit()
            rate = (time.time() - started) / number
            print(f"[cards] {number}/{len(todo)}  {rate:.1f} s each  eta {(len(todo) - number) * rate / 60:.0f} min", flush=True)
    catalog.close()
    counts["seconds"] = int(time.time() - started)
    return counts
