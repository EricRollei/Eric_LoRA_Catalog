# LoRA Catalog: user guide

The reference for everything the catalog does. Start with the [README](../README.md) for install and first run.

- [The catalog database](#the-catalog-database)
- [Card editor](#card-editor)
- [Search and folders](#search-and-folders)
- [Civitai audience data](#civitai-audience-data)
- [Genre and mood](#genre-and-mood)
- [Sweep verdicts (ComfyUI + Eric_Krea2)](#sweep-verdicts-comfyui--eric_krea2)
- [Sliders and axes](#sliders-and-axes)
- [Forge / AUTOMATIC1111](#forge--automatic1111)
- [Every command](#every-command)

## The catalog database

One sqlite file (`settings.json` `catalog_path`; default `<data folder>/lora_catalog.sqlite`).

| table | what |
|---|---|
| `loras` | one row per file: identity (sha256, Civitai ids), Civitai fields (name, base model, tags, trained words, description, licence), header facts (network dim, dataset tags), your own training (trigger, arch, captions), LoRA Manager flags (favourite, exclude, notes) |
| `examples` | gallery prompts that used the LoRA, with the weight it was used at when that can be told |
| `cards` | what each LoRA does, written by the LLM: role, kind, scope, effect, facets, triggers, weights, NSFW level, genre, mood |
| `card_edits` | your corrections, keyed by sha256 (they follow the file when it moves); they always win over `cards` |
| `card_facets` | the merged view (card + your edit) the LoRA Suggester reads; `run.py facets` rebuilds it |
| `community_stats`, `community_prompts` | Civitai audience data (below) |
| `stacks` | your good / bad marks on sweep cells |

**Sources, cheapest first**: LoRA Manager's `<stem>.metadata.json` (sha256, flags, Civitai block), LoRA Manager's
offline Civitai database, the Civitai downloader's folder layout and sidecars (`<file>.json`), the safetensors header
(kohya `ss_*`, `modelspec.*`), ai-toolkit job folders (trigger word, arch, captions) for LoRAs you trained.
LoRA Manager is GPL-3: this tool reads its data files and contains none of its code.

`index` skips files whose size and date are unchanged (`--force` re-reads them). A library of ~5,500 files takes
about two minutes: only headers and sidecars are read, nothing is hashed, nothing goes to the network.

## Card editor

    launch.bat / launch.sh, or: python run.py serve      -> http://127.0.0.1:8765
    (inside ComfyUI: the LoRA Catalog sidebar tab -> Open card editor)

- **Pick lists** come from `vocab.yaml` in your data folder (role, use, medium, period, genre, mood, rendering,
  depth, palette, line, subjects, side effects). Your copy starts as the shipped one; edit it freely, the editor
  re-reads it on page load. `python run.py vocab-gaps` lists words you typed that are not in it yet.
- **Edits** go to `card_edits`, keyed by sha256. Generated cards are never touched; your edit wins wherever it exists.
- **Chips**: filled = chosen, dashed orange = the model picked it and you removed it, purple = your own word. Click
  a purple word's **＋** to make it a button for every card (written into `vocab.yaml`, a dated backup first).
- **Keys**: Enter save and next, U unsure, J/K next/prev, 1-4 enlarge an image (←/→ step through them),
  Shift+1-5 rate, Shift+0 clear the rating, `/` search. Leaving a card with unsaved changes warns once.
- **Look**: a one-line summary from period, medium, rendering, depth, line, palette and lighting (display only).
- **My rating** (stars by the name): the LoRA Suggester ranks ★4-5 higher and ★1-2 lower; unrated is neutral. Use
  ★1-2 and the rating filter to find LoRAs to cull.
- **Trigger words**: click cycles normal → ★ priority (put first) → off (kept on the card, never added to prompts);
  double-click edits; × removes.
- **Weights**: low, typical and high, each −10 to +10 (sliders often run negative). The bar shows your range (solid)
  over the model's (dashed), your typical weight (tick) and, when collected, the weights Civitai users use (green).
- **copy as prompt tag**: `<lora:name:weight>` plus the active trigger words, for Forge / A1111 prompts.
- **Example images**: "Get images from Civitai" under the images on a card with fewer than four, or
  `python run.py images` for the whole library (`--fewer-than 4`, `--only`, `--max 12`, `--dry-run`). Saved next to
  the LoRA in `images/` the way the Civitai downloader does it. Least NSFW first; images Civitai flags as showing a
  minor are never downloaded.
- Queue "most useful first" = low confidence, sparse cards and missing artists first.

## Search and folders

- **Search box** (top left, or `/`): name, trigger, artist, creator, notes, Civitai tags, pick-list words, card
  text, folder, description. Every word must match and each is a prefix (`photo` finds photography).
  `"film noir"` = exact phrase, `-anime` = leave out, `name:` `trigger:` `artist:` `creator:` `notes:` `tag:` `card:`
  `folder:` `description:` = one field only. With no LoRA matching every word it falls back to any word and says so.
- **Folders** button: a folder panel and checkboxes on the list. Tick LoRAs (Shift-click = a range), hover a folder
  and press **here** to move them (nothing ticked = the open card). **New folder**, **Rename / move**, **Undo...**
- Every change shows its plan first and does nothing until **Apply**. Never deleted, never overwritten: a name taken
  at the target skips that LoRA, a file ComfyUI has loaded skips that LoRA, moves stay inside one library root.
- What moves with a LoRA: a Civitai-downloader LoRA moves as its whole version folder; a LoRA in a flat folder moves
  with its `<stem>.*` files and `images/<stem>*`.
- Kept in step: every catalog table that stores the path, your edits, LoRA Manager's `.metadata.json`; LoRA Manager
  is asked to rescan when ComfyUI is running. `path_history` keeps old → new paths; each batch writes
  `<data folder>/moves/<date_time>.json` (what Undo replays).
- Saved workflows still name LoRAs by their old relative path. After a move, "Find saved workflows / notes that
  still name them" lists them (folders in `settings.json` `reference_dirs`; ComfyUI's saved workflows are found by
  themselves); nothing is rewritten.
- **Delete…** moves the ticked LoRAs, with everything that travels with them, to `_lora_trash/<date_time>/` beside
  the library root: out of every UI's lists, same plan / Apply / Undo as a move. Empty the trash folder yourself.
- **Duplicates**: every LoRA that exists more than once (same sha256), copies side by side; tick the copy to remove,
  Delete…; the plan warns if you ticked every copy of something.
- Don't move while a card, slider or harvest run is going (they write by path); the plan warns when one is.

## Civitai audience data

How the wider Civitai audience rates each LoRA, what weights people really use, what they make with it, and prompts
to test it with. Only metadata from the public API (no images downloaded), with your Civitai key.

- `python run.py community` (`--only <folder>`, `--limit N`): per Civitai version, the model stats, its 50
  most-reacted and 200 newest community images (metadata only), each person counted at most 5 times. Skips versions
  fetched in the last 30 days (`--refresh-days`, `--force`). `--score` recomputes scores only; `--show NAME` prints one
  LoRA's numbers and prompts.
- **Score** = mean of five percentiles (approval, likes per download, downloads per day, people posting with it,
  reactions), ranked within the same base model + role. No score when the base model has fewer than 5 collected. A
  popularity prior: your stars always win. Shown as "Civitai top N%".
- **Weights people use** (p25 / median / p75 from image metadata); "use these" fills your low / typical / high. With
  10+ community weights the card pass uses them instead of the model's guess.
- **Content levels** of its newest images (None / Soft / Mature / X), and a warning when a third or more of its
  mature images had no nudity words in the prompt (the LoRA may push it by itself; a heuristic).
- **Test prompts**: from the creator's examples, the most-liked and the newest, cleaned of LoRA tags, each labelled
  with its level and style (natural / tags), filterable by level.
- **Compared with its platform**: ranked against the other LoRAs of the same base model you collected, on downloads,
  likes per download, images per day and median reactions. Context only; the score is unchanged.
- **newer** badge when Civitai has a newer version of the same model for the same base model (only when the names say
  it is an update, not a sibling in a collection).
- Usernames are never stored in clear (a salted hash, enough to count distinct people).
- `python run.py identify` finds the Civitai version of files that came without one, by sha256 (`run.py hash` first).
- Wishlist: `python run.py pairs --wishlist [--base Krea] [--top 40]` - LoRAs you do not own that Civitai users
  stack with yours (after `run.py harvest` + `run.py pairs`).

## Genre and mood

Two pick lists beside period: **genre** = the world or aesthetic (gothic, horror, dark fantasy, cyberpunk,
steampunk, noir, western, cottagecore, vaporwave ...) and **mood** = the feeling (dark / moody, serene, cozy,
whimsical, dreamy, nostalgic, dramatic, romantic, mysterious, energetic).

- `run.py genre-mood` classifies them as its own step, text only, with `genre_model` (blank = the card model; a
  larger text model does better: the 8B card model alone put "dark fantasy" on most LoRAs). Answers are limited to
  the vocab ids, at most 2 each, empty is the expected answer, the lists are shuffled per LoRA, and it warns when one
  term covers more than 20% of the LoRAs. `--dry-run --sample --limit 30` to check; `--force` to redo all.
- Cards not classified yet use a guess from name, tags, era and effect.
- The LoRA Suggester reads both from the whole prompt: genre counts most for styles, transforms and environments,
  mood most for lighting.

## Sweep verdicts (ComfyUI + Eric_Krea2)

Which LoRAs work together, judged by eye on your own renders. Needs the Eric_Krea2 nodes.

- `python run.py sweep-plan --folders Krea2/graphic-art Krea2/art-style` picks LoRA pairs and saves a preset in
  Eric_Krea2's `sweep_presets.json` (backup first): choose it in the SweepPlan node's `sweep_preset` list. `--dry-run`
  prints without writing. `--single "name"`: one LoRA's characterization sheet (its own test prompts x three weights +
  no-LoRA). `--prompts from-loras`: the LoRAs' own test prompts. More: `python run.py sweep-plan --help`.
- **Sweeps** page (link in the card editor header, or `/sweeps`): press **G** good, **B** bad, **U** unsure on each
  cell; **Space** opens the cell beside its parts. Each mark stores the cell's exact stack in the `stacks` table.
- `python run.py stacks --pairs`: every stack and LoRA pair with its good / bad / unsure counts.

## Sliders and axes

Slider LoRAs (age, waist, shot size, saturation ...) are dialled by the LoRA Suggester from phrases in the prompt.
`axes.yaml`, `axes_ladders.yaml` and how to calibrate: [design/SLIDERS_AND_AXES.md](design/SLIDERS_AND_AXES.md).

    python run.py sliders            slider cards (axis, direction, range) - text only
    python run.py ladders --draft    LLM-drafted phrase ladders -> axes_ladders_draft.yaml (then --tidy, --accept)

## Forge / AUTOMATIC1111

- **copy as prompt tag** on any card.
- `python run.py export-sidecars` lists what it would write; `--write` writes `<lora name>.json` next to each LoRA
  (activation text = trigger words, preferred weight = typical weight, description = what it does). Forge and A1111
  show these in their LoRA browser. A sidecar that this tool did not write is never touched; `--overwrite-ours`
  refreshes the ones it did. Run `run.py facets` first.

## Every command

`python run.py --help` lists them; `python run.py <command> --help` explains one. The refresh steps, in order:
index, hash, identify, crawl, creators, images, community, cards, genre-mood, review-all, facets.

Not in the refresh (run by hand when wanted): `harvest` + `pairs` (which LoRAs people stack together; slow),
`sliders`, `ladders`, `cards --force` (re-card everything), `export-sidecars`.

Tests (no network, nothing written to your catalog): `python dev/verify_release.py`, `dev/verify_lora_suggest.py`,
`dev/verify_community.py`, `dev/verify_organise.py`, `dev/verify_sweepplan.py`, `dev/verify_panel_routes.py`.
