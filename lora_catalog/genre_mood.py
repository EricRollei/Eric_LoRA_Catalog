"""
Genre and mood for each card, as a step of its own (2026-10-10). Replaces asking the card writer for them: the 8B card
model put "dark fantasy" on ~60% of LoRAs (Ghibli, street photos, poster prints) - a small model given a long list of
words grabs one rather than leave the field empty.

    python run.py genre-mood                       every card not classified yet (genre IS NULL)
    python run.py genre-mood --force               every card again
    python run.py genre-mood --only Krea2 --limit 30 --dry-run      print, write nothing

How:
- text only (no images): the LoRA's name, Civitai tags, trained words, a slice of its description and the card's own
  effect / medium / era / subjects - enough to say what world and feeling it belongs to, and fast;
- the answer is constrained by a JSON schema to the term ids of vocab.yaml (no free words like "anime" or "porn"),
  at most 2 genres and 2 moods, and EMPTY is the expected answer for most LoRAs - the instructions say so plainly and
  give examples of both;
- the term lists are shuffled per LoRA (seeded by its path, so a re-run is repeatable), so no word wins by being first;
- model: settings.genre_model (default the 27B already loaded in LM Studio), not the 8B card model;
- after a run it prints how often each term was given and warns when one term covers more than 20% of the LoRAs it
  classified (the failure that started this).

Stored in cards.genre / cards.mood as comma-separated labels ("" = classified, none fits; NULL = not classified yet).
vocab.prefill turns them into pick-list terms; your edits in the card editor always win.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import collections
import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings
from .store import Catalog
from .vocab import load_vocab

DEFAULT_MODEL = ""          # settings.genre_model, else the card model (config.Settings.finish)
MAX_PICKS = 2
WARN_SHARE = 0.20

SYSTEM = """You label LoRA adapters for an image model with GENRE and MOOD, so a tool can match them to prompts.

GENRE = the fictional world or aesthetic subculture the LoRA's images belong to: gothic, cyberpunk, steampunk,
noir, western, horror, dark fantasy, high fantasy, sci-fi, post-apocalyptic, cottagecore, vaporwave and so on.
- A way of DRAWING or PHOTOGRAPHING is not a genre: anime, manga, comic, watercolour, oil painting, an artist's look,
  a film stock, a poster print, realism, a camera style -> genre [].
- A subject or body type, clothing, a pose, a person, NSFW content, a detail or quality booster, a slider, a lighting
  tool -> genre [].
- Pick a genre ONLY when the name, tags, description or effect clearly place the LoRA's pictures in that world
  (castles and dragons with grim armour -> dark_fantasy; neon megacity, implants -> cyberpunk; brass gears and
  airships -> steampunk). A single dark or fantastical picture is not enough.
- Most LoRAs have NO genre. [] is the normal answer.

MOOD = a feeling the LoRA pushes onto every picture even when the prompt is neutral (eerie, cozy, melancholic,
whimsical, serene, dreamy ...).
- Most LoRAs leave the mood to the prompt -> mood [].
- Do not read mood from lighting alone: "dramatic lighting" or "dark shadows" is lighting, not a mood.

Examples:
- "Studio Ghibli style" (soft anime landscapes)            genre []                 mood [whimsical, nostalgic]
- "Street photo Hong Kong night" (photographic look)       genre []                 mood []
- "Detail tweaker" / "Age slider" / "Hands fix"            genre []                 mood []
- "Lovecraftian horror art" (tentacles, cultists)          genre [cosmic_horror]    mood [eerie]
- "Neon Tokyo 2099" (cyborgs, holograms)                   genre [cyberpunk]        mood []
- "Cottage kitchen mornings" (warm rustic interiors)       genre [cottagecore]      mood [cozy]
- "Moebius style" (clean-line sci-fi comics)               genre [science_fiction]  mood []
- "Pin-up girl 1950s"                                      genre []                 mood []

Return at most two of each, from the lists given (use the ids exactly), most certain first."""


def _terms(facet: str) -> List[Tuple[str, str]]:
    vocab = load_vocab()
    return [(term_id, term.label) for term_id, term in vocab.facets[facet].terms.items()]


def schema(genres: List[str], moods: List[str]) -> Dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "genre_mood", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "genre": {"type": "array", "maxItems": MAX_PICKS, "items": {"type": "string", "enum": genres}},
            "mood": {"type": "array", "maxItems": MAX_PICKS, "items": {"type": "string", "enum": moods}},
        },
        "required": ["genre", "mood"]}}}


def _json_list(value: Any) -> List[Any]:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def evidence(row: Any) -> str:
    parts = [f"Name: {row['model_name'] or row['stem']} {('- ' + row['version_name']) if row['version_name'] else ''}",
             f"Folder: {row['rel_dir']}   Base model: {row['base_model'] or '?'}"]
    tags = _json_list(row["tags"])
    if tags:
        parts.append("Civitai tags: " + ", ".join(map(str, tags[:15])))
    words = _json_list(row["trained_words"])
    if words:
        parts.append("Trained words: " + " | ".join(map(str, words[:6])))
    for label, key in (("What it does", "effect"), ("Medium", "medium"), ("Era / movement", "movement_era"),
                       ("Palette", "palette")):
        if row[key]:
            parts.append(f"{label}: {row[key]}")
    subjects = _json_list(row["suited_subjects"])
    if subjects:
        parts.append("Good at: " + ", ".join(map(str, subjects)))
    if row["description"]:
        parts.append("Description (start): " + " ".join(str(row["description"]).split())[:700])
    return "\n".join(parts)


def classify(settings: Settings, model: str, row: Any, genre_terms: List[Tuple[str, str]],
             mood_terms: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    from .llm import chat_json
    rng = random.Random(row["path"])                  # shuffled per LoRA, same order on a re-run
    genres, moods = genre_terms[:], mood_terms[:]
    rng.shuffle(genres)
    rng.shuffle(moods)
    lists = ("GENRE ids: " + ", ".join(t for t, _l in genres) + "\nMOOD ids: " + ", ".join(t for t, _l in moods))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": evidence(row) + "\n\n" + lists}]
    answer = chat_json(settings, model, messages, schema([t for t, _l in genres], [t for t, _l in moods]),
                       temperature=0.0, max_tokens=1500, timeout=180)   # a thinking model uses ~200-600 tokens first
    known_g, known_m = {t for t, _l in genres}, {t for t, _l in moods}
    pick_g = [t for t in answer.get("genre") or [] if t in known_g][:MAX_PICKS]
    pick_m = [t for t in answer.get("mood") or [] if t in known_m][:MAX_PICKS]
    return list(dict.fromkeys(pick_g)), list(dict.fromkeys(pick_m))


def run(settings: Settings, only: str = "", limit: int = 0, force: bool = False, dry_run: bool = False,
        model: str = "", sample: bool = False, genre_like: str = "") -> Dict[str, Any]:
    model = model or getattr(settings, "genre_model", "") or DEFAULT_MODEL or settings.card_model
    catalog = Catalog(settings.catalog_path)
    where, params = ["1=1"], []
    if only:
        where.append("replace(l.rel_dir, '\\', '/') like ?")
        params.append(only.replace("\\", "/").strip("/") + "%")
    if not force and not dry_run:
        where.append("c.genre is null")
    if genre_like:                                    # a check on cards that already carry a genre word
        where.append("lower(c.genre) like ?")
        params.append(f"%{genre_like.lower()}%")
    order = "random()" if sample else "l.rel_dir, l.file_name"
    rows = catalog.query(f"""select l.path, l.stem, l.model_name, l.version_name, l.rel_dir, l.base_model, l.tags,
                                    l.trained_words, l.description, c.effect, c.medium, c.movement_era, c.palette,
                                    c.suited_subjects, c.genre as old_genre, c.mood as old_mood
                             from loras l join cards c on c.path = l.path where {' and '.join(where)}
                             order by {order}""", tuple(params))
    if limit:
        rows = rows[:limit]
    genre_terms, mood_terms = _terms("genre"), _terms("mood")
    labels = {t: l for t, l in genre_terms + mood_terms}
    print(f"[genre-mood] {len(rows)} LoRAs with {model}{' (dry run)' if dry_run else ''}", flush=True)
    counts: Dict[str, collections.Counter] = {"genre": collections.Counter(), "mood": collections.Counter()}
    done = failed = with_genre = with_mood = 0
    started = time.time()
    for number, row in enumerate(rows, 1):
        try:
            genres, moods = classify(settings, model, row, genre_terms, mood_terms)
        except Exception as error:
            failed += 1
            print(f"[genre-mood] failed {row['stem']}: {str(error)[:120]}", flush=True)
            continue
        counts["genre"].update(genres)
        counts["mood"].update(moods)
        with_genre += bool(genres)
        with_mood += bool(moods)
        done += 1
        genre_text = ", ".join(labels[t] for t in genres)
        mood_text = ", ".join(labels[t] for t in moods)
        if dry_run:
            print(f"  {str(row['model_name'] or row['stem'])[:40]:40s} genre [{genre_text}]  mood [{mood_text}]"
                  f"     (was: {row['old_genre'] or '-'} / {row['old_mood'] or '-'})", flush=True)
        else:
            catalog.connection.execute("update cards set genre = ?, mood = ? where path = ?",
                                       (genre_text, mood_text, row["path"]))
            if number % 25 == 0:
                catalog.commit()
        if number % 50 == 0:
            rate = (time.time() - started) / number
            print(f"[genre-mood] {number}/{len(rows)}  {rate:.1f} s each  eta {(len(rows) - number) * rate / 60:.0f} min",
                  flush=True)
    catalog.commit()
    catalog.close()
    report = {"classified": done, "failed": failed, "with_genre": with_genre, "with_mood": with_mood,
              "genres": counts["genre"].most_common(12), "moods": counts["mood"].most_common(12)}
    for facet, have in (("genre", with_genre), ("mood", with_mood)):
        for term, n in counts[facet].most_common(3):
            if done >= 30 and n / done > WARN_SHARE:
                print(f"[genre-mood] WARNING: {facet} '{term}' on {n} of {done} LoRAs ({n / done:.0%}) - "
                      f"check a sample before trusting it", flush=True)
    return report
