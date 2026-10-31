# Prompt library: export the catalog's prompts as a searchable, updatable database

2026-10-29 - status: **BUILT 2026-10-29** (lora_catalog/prompt_library.py, run.py prompts build|stats|export, refresh step 11c, community hook, community_prompts.nsfw_num; tests dev/test_prompt_library.py). Changes from this plan while building: triggers that are ordinary prompt words (in 0.2% or more of all prompts) are kept; word options and LoRA names are not stripped (they are usually descriptive words). Civitai's images API returns browsingLevel (1-16): stored from now on as nsfw_num, so new community rows have the exact X / XXX split.

## Goal

Turn the prompts the catalog already collects into one deduplicated **prompt library** you can filter (base model,
rating, NSFW level, style, the look of the LoRA it came with) and export - to text files the Eric_Krea2 Sweep Plan
reads directly (`random: D:\prompts\x.txt x20`), to JSONL/CSV for LLM tools and spreadsheets, or as a standalone
SQLite file. It updates itself as the catalog collects more.

## What exists (read from the code and your catalog, 2026-10-29)

| Source | Table | What one row is | Size |
|---|---|---|---|
| Card preview prompts ("Prompts people used") | `examples` | the generation data of one preview image of one of your LoRAs: prompt, negative, LoRA weight, checkpoint, sampler, steps, CFG, NSFW level, the image file | 3,019 rows, 2,435 different prompts (Krea2 785) |
| Community prompts ("Test prompts") | `community_prompts` | one Civitai community image made with a LoRA version you own: raw prompt, **cleaned** (LoRA tags removed), negative, weight, base model, style (natural / tags), NSFW level (None / Soft / Mature / X), reactions, picked as a test prompt | 121,139 rows, 70,264 different cleaned prompts; NSFW None 57,846 / Soft 11,165 / Mature 14,243 / X 37,885; natural 68,754 / tags 52,385; 28,076 rows have no base model recorded |
| Stack harvest images | `harvest_images` | which LoRAs one community image used, base, NSFW, reactions - **no prompt text** | 46,495 (used for "which LoRAs were in it" only) |

## The library

A separate file, **`L:\Models\prompts\_prompt_library.sqlite`** (setting `prompt_library_dir`; exports default to the
same folder), rebuildable, never edited by hand:

```
prompts      id (hash of the normalised text), text (triggers stripped), raw (one original wording), negative,
             style (natural | tags), words, tokens_est, nsfw (worst level seen), nsfw_words (0/1: the text itself
             mentions nudity etc.), reactions_max, reactions_sum, occurrences, first_seen, last_seen
occurrences  prompt_id, source (preview | community-top | community-newest | creator), image_id or media_file,
             version_id, lora_path (yours), base_model, checkpoint, sampler, steps, cfg, weight, nsfw, reactions
loras_used   prompt_id, lora (name or version), weight, owned (0/1)      - every <lora:...> tag / LoRA in that image
triggers     prompt_id, lora_path, trigger                                - what was stripped, per LoRA
facets       prompt_id, facet, value                                      - copied from the card of the LoRA it came
                                                                           with: kind, medium, period, genre, use ...
```

- **One prompt, many occurrences.** The same prompt text used with 6 LoRAs is one row with 6 occurrences, so
  counts, reactions and facets accumulate instead of duplicating.
- **Base model:** taken from the occurrence; when Civitai left it empty (28k rows), from the LoRA version's own base.
- **Ratings kept for filtering:** community reactions (max and sum), and **your** rating of the LoRA it came with
  (card stars and sweep quality) - joined at export time so they stay current.

## Trigger words: strip, and remember

Your sweeps show triggers leaking into images as text or signatures, so the library's `text` has them removed while
`triggers` records exactly what was removed (and `raw` keeps one original):

- removed: the LoRA's card triggers, its trigger-recipe words and word options, `<lora:...>` / `<lyco:...>` tags,
  and the LoRA's own name when it appears as a word; whole-word, case-insensitive; the punctuation left behind is
  tidied (no ", ," or leading commas);
- not removed: ordinary words that happen to be part of a trigger phrase elsewhere in the sentence (only the whole
  phrase is matched);
- `loras_used` keeps which LoRAs (and weights) the original image used - useful to rebuild the original, to find
  "prompts that worked with LoRA X", and to avoid testing a LoRA on prompts written for a different one.

So the default export is generic prompts that do not depend on any LoRA, and you can still ask for the originals.

## Updating over time

- `run.py prompts build` reads the catalog tables and **merges**: new prompts are added, known ones get new
  occurrences, `last_seen` moves; nothing you exported changes meaning. Re-running it with nothing new is cheap
  (hash lookups).
- New material arrives through the existing collectors - `run.py community` (community prompts for your LoRAs, which
  refreshes as Civitai gets new images), the crawl (new LoRAs' previews). `run.py prompts build` can run after them
  automatically - as the last step of `run.py refresh` and after `run.py community` (a `prompt_library_auto: true` setting) - or by hand.
- LoRAs you delete: their occurrences stay (the prompts are still good prompts) but are marked `lora_path` gone.

## Selecting and exporting

```
run.py prompts export --base "Krea 2" --max-nsfw Soft --style natural --min-reactions 20 --out D:\prompts\krea2.txt
run.py prompts export --facet period="art nouveau" --lora-rating 4 --format jsonl --out art_nouveau.jsonl
run.py prompts export --base "Krea 2" --split-by medium --out D:\prompts\krea2\         one .txt per medium
run.py prompts stats                                                                         counts by base, NSFW, style
```

| Filter | Meaning |
|---|---|
| `--base` | base model (part of the name; "Krea" matches Krea 2 and Krea 2 Turbo) |
| `--max-nsfw PG / PG-13 / R / X / XXX` | Civitai's current rating scale, the worst image level seen for that prompt (default **X**, see NSFW levels); `--clean-text` also drops prompts whose text itself mentions nudity |
| `--min-reactions N` | community likes on the best image made with it |
| `--lora-rating N` | only prompts that came with LoRAs you rated N stars or more (card stars or sweep quality) |
| `--style natural / tags` | natural-language sentences or tag lists (1girl, masterpiece ...) |
| `--facet k=v` | the look of the LoRA it came with (medium=watercolour, period="art nouveau", genre=noir ...) |
| `--words / --max-words` | length; the Krea 2 prompt budget is about 500 tokens |
| `--with-triggers` | export `raw` (triggers kept) instead of the stripped text |
| `--source preview / community` | only card-preview prompts or only community prompts |

Formats: `txt` (one prompt per line - ready for the Sweep Plan), `jsonl` / `csv` (every field), `sqlite` (a copy of the
library, filtered). A `--sample N --seed S` option draws a random subset.

Later (not stage 1): a Prompts page in the card editor with the same filters and a copy / export button, and on each
card the preview prompts shown next to their image.

## Tests

- Dedup: the same text with different spacing / capitals / LoRA tags is one prompt with several occurrences.
- Trigger stripping: whole-phrase, case-insensitive; "rust" is not removed from "rusted"; the leftover punctuation is
  clean; `triggers` lists what was removed; `--with-triggers` gives the raw text back.
- NSFW: the worst level wins; `--clean-text` drops a None-level image whose prompt says "nude".
- Incremental build: a second build with one new community row adds one occurrence and changes nothing else.
- Every filter on a fixture DB.

## NSFW levels: Civitai's current scale

Civitai rates images **PG, PG-13, R, X, XXX** (numbers 1, 2, 4, 8, 16; 32 = blocked). The library stores that scale:

| Stored in the catalog | Maps to |
|---|---|
| numbers (card previews, `examples`): 1 / 2 / 4 / 8 / 16 / 32 | PG / PG-13 / R / X / XXX / blocked (never exported) |
| older text levels (community prompts, harvest): None / Soft / Mature / X | PG / PG-13 / R / **X-or-XXX** |

The older text "X" does not separate X from XXX, so those count as X: with the default `--max-nsfw X` some XXX images
from that source can come through. When `run.py community` next runs it should store Civitai's number as well
(stage 1 checks whether the API returns it; if so, new rows get the exact level and old ones are refreshed over time).

## Rewriting prompts during export (optional)

Community prompts are uneven: tag lists, quality-tag junk (`masterpiece, best quality`), `(word:1.2)` weights, or a
few words. `--rewrite MODE` passes the selected prompts through a local LLM - the same OpenAI-compatible server the
catalog already uses (LM Studio / Ollama, `llm_base_url`), text model from a new `prompt_model` setting (blank:
`genre_model`, then `card_model`):

| Mode | What it does |
|---|---|
| `tidy` | Same content, cleaner: tag lists become sentences, weights and quality tags go, grammar fixed. Adds nothing. |
| `krea2` | Restructures into a Krea 2 prompt with the **Eric_Krea2 Magic Prompt** expander's system prompt (read from the installed Eric_Krea2 node, or its `system_prompt_path` file - e.g. Krea's official expander), kept within the Krea 2 prompt budget. |
| `expand` | Only prompts shorter than `--expand-under N` words (default 25) are expanded with that expander; the rest pass unchanged. |

- Rewrites are **cached** (table `rewrites`: prompt id, mode, model, a hash of the system prompt, text, date), so a
  second export reuses them and a big run is resumable. JSONL / CSV exports carry both the original and the rewrite.
- It is slow by nature (a few seconds per prompt with an 8B model): the export prints an estimate first and asks
  before more than 200 new rewrites (`--yes` to skip the question). Trigger stripping happens before the rewrite, so
  the LLM never sees them.

## Note

Community prompts are other people's public posts. The library is for your own use; it is not shipped with the
repo, and an export to share should say where the prompts came from.

## Decisions (2026-10-29, Eric)

1. NSFW: Civitai's current scale PG / PG-13 / R / X / XXX; default export maximum **X**.
2. Text: **stripped** by default (`--with-triggers` for the original).
3. Rebuild: **automatic** (end of `run.py refresh`, after `run.py community`).
4. Location: **`L:\Models\prompts\`** (library file and default export folder).
5. Optional LLM rewrite during export (tidy / krea2 / expand) - added to the plan above.
