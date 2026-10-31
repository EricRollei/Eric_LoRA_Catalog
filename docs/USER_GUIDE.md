# LoRA Catalog: user guide

The reference for everything the catalog does. Start with the [README](../README.md) for install and first run.

- [The catalog database](#the-catalog-database)
- [Card editor](#card-editor)
- [Search and folders](#search-and-folders)
- [Civitai audience data](#civitai-audience-data)
- [Genre and mood](#genre-and-mood)
- [Style vectors (CSD): search by look](#style-vectors-csd-search-by-look)
- [Trigger recipes](#trigger-recipes)
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
| `stacks` | your marks on sweep cells: verdict, stars, tags, notes, plus the cell's stack, checkpoint, subject, words |
| `style_vectors` | one CSD style vector per LoRA from its preview images, plus its render verdict (below) |
| `style_renders` | one row per checked sweep cell: how close the render came to the LoRA's own look |
| `trigger_recipes`, `trigger_tests` | prompt recipes per LoRA and the trigger-test scores ([TRIGGERS.md](TRIGGERS.md)) |

**Sources, cheapest first**: LoRA Manager's `<stem>.metadata.json` (sha256, flags, Civitai block), LoRA Manager's
offline Civitai database, the Civitai downloader's folder layout and sidecars (`<file>.json`), the safetensors header
(kohya `ss_*`, `modelspec.*`), ai-toolkit job folders (trigger word, arch, captions) for LoRAs you trained.
LoRA Manager is GPL-3: this tool reads its data files and contains none of its code.

`index` skips files whose size and date are unchanged (`--force` re-reads them). A library of ~5,500 files takes
about two minutes: only headers and sidecars are read, nothing is hashed, nothing goes to the network.

## Card editor

    launch.bat / launch.sh, or: python run.py serve      -> http://127.0.0.1:8765
    (inside ComfyUI: the LoRA Catalog sidebar tab -> Open card editor)

- **The editor's switch in the sidebar (2026-10-17).** Under *Open card editor* the panel shows whether the editor is
  running (since when, its pid) with a **Stop editor / Start editor** button. An editor started from the panel stops by
  itself within ~10 s of ComfyUI closing; one started with launch.bat runs until you stop it. When the running editor
  is older than the code on disk (after an update), the line turns orange and **Open card editor** restarts it.
  Catalog jobs (refresh, cards ...) are separate: they keep running through a ComfyUI restart and have their own Stop.

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

## Style vectors (CSD): search by look

Words describe what a LoRA is *about*; they are poor at what it *looks like*. The catalog can also describe every
LoRA by its look, with **CSD** (Contrastive Style Descriptors; Somepalli et al., ECCV 2024): a CLIP ViT-L/14 image
model fine-tuned so that two images in the same style land close together *whatever they show*. Each image becomes
768 numbers (a unit vector); the cosine between two vectors says how alike their styles are (1 = same, ~0.1 =
unrelated).

- **Per LoRA:** up to 12 of its preview images are embedded and averaged into one *style signature*. Its previews
  typically agree at a cosine of ~0.5 with each other and ~0.1 with other LoRAs' previews, so no rendering is needed.
  Image vectors are cached (`<data folder>/cache/style_images.sqlite`), so a refresh only embeds new images.
- **What it is good at:** look-type LoRAs (cards of kind style_medium, photo_look, lighting_colour). Two Mignola
  LoRAs match at 0.87; Frazetta sits with Vallejo and Brom.
- **What it cannot do:** tell photoreal LoRAs apart. Concept, subject and most NSFW LoRAs all look like "a realistic
  photo" to it (0.90+ with each other), so style search and the checks below only use look-type cards.

Setup (optional - nothing else needs it):

    python run.py style-setup         download (2.4 GB) and convert the CSD weights to a 1.2 GB safetensors file
    settings.json: "style_enabled": true   ("style_device": auto | cuda:N | cpu; "style_model": your own copy)
    python run.py style               vectors for every LoRA (also a refresh step once enabled)

On a GPU it embeds about 80 images a second (fp16); on CPU about 3 a second, an evening for a big library, once.
Torch is only loaded for this; the editor hides the style controls when it is off.

What uses it:

- **Editor - similar looks:** each card shows a strip of the LoRAs with the closest look.
- **Editor - image search:** click **By image**, or drop or paste an image onto the search box. The image is embedded
  with the same model and the list becomes your LoRAs ordered by closeness of look, at most 200, inside the current
  filters (base-model shelf, status, rating) - choose the shelf first. The first search loads the model (about 15 s);
  it unloads again when idle. Tips:
  - It matches style, not subject. For "LoRAs like this one", use its card's Similar look strip instead.
  - **made with:** when the image carries generation metadata (ComfyUI workflow, A1111 / Forge parameters, Civitai),
    a line above the list names the LoRAs it was made with and their weights - yours open in the editor, the others
    link to their Civitai page. Screenshots and most social-media copies have no metadata.
  - A new text search (or the search box's clear button) ends the image search.
- **Editor - Similar look strip** (under each card's images): the closest looks to this LoRA. *previews agree*
  (high / mixed / low) says how consistent its own previews are - low means its signature is a blur of different
  looks and the matches are less reliable. Toggles: **all bases** (off = same base model only), **all kinds** (off =
  style, photo-look and lighting LoRAs only), **NSFW**, and **list all** (the whole ranked list on the left). A
  render-check verdict shows here too (shows its look / needs its style words / weak).
- **Editor - Look-alikes:** pairs of style LoRAs on the current shelf whose previews look nearly the same, side by
  side - for culling near-copies of one look.
- **LoRA Suggester (Advanced) `style_image`:** a reference image's look counts as evidence for the style,
  transform and treatment picks, next to the prompt words (`style_image_weight`).
- **Render check** - `python run.py style --renders <sweep folder>`: does each LoRA cell actually show the
  LoRA's preview look? A cell **shows** when its render ranks among the 10 closest looks on its platform and sits at
  least +0.05 closer to the LoRA than the no-LoRA render. Each LoRA gets a verdict: `shows`, `needs_words` (shows
  only with its creator's prompt), `weak` or `unclear`. The Suggester (Advanced, `add_style_words`) gives
  `needs_words` LoRAs their card's style words with the triggers.
- **Second pass** - `python run.py style --second-pass`: a sweep preset that re-renders the `unclear` LoRAs with
  their creator's own prompt, so the next render check can tell `needs_words` from `weak`.
- **Trigger test** - `run.py triggers --check` scores trigger variants the same way (next section).
- `python run.py style --show <name>` prints the closest looks to a LoRA from the command line.

## Trigger recipes

What a LoRA needs in the prompt - trigger token, official words, a style phrase - and how that depends on the
checkpoint: [TRIGGERS.md](TRIGGERS.md).

**Prompt words** (2026-10-15): the card editor's *Prompt words* panel (under Trigger words) lists the recipe's
variants with their trigger-test scores; **Use** picks one for all checkpoints or for one checkpoint, saved at once.
A trigger test's own suggestion counts automatically for the checkpoint it ran on (`--check`; `--sync-picks` for
older checks). Eric_Krea2's `recipe` trigger mode then puts those words in the prompt. The card shows them as
`prompt_text` / `trigger_status` / `prompt_by_checkpoint` in `card_facets` (table `trigger_picks` holds the picks).

**Word options** (2026-10-20): for LoRAs trained on alternatives rather than one trigger - one keyword per colour
grade, per style, per age or cohort - the card keeps several named sets of words. *Prompt words > Word options*, one
per line: `name = words` (a line without `=` is both), a leading `*` makes that option the LoRA's default words (after
your own pick for all checkpoints; per-checkpoint picks still win). Saved at once; an empty box saves "no options";
**Back to automatic** forgets yours.

- **Automatic options:** when a LoRA's Civitai trained words look like alternatives they are its options with no
  setup and no default (they change nothing until you pick one): 3+ entries of 6 words or fewer, all different, and
  either one word shared by most of them (60% and at least 3: `cohort_fi`, `cohort_se` ...; `... style`) or all of
  them single made-up tokens (`18yo`, `19yo` ...). A lone single-word entry without the shared word (the LoRA's own
  trigger) goes in front of every option. About 60 LoRAs qualify today; a few lists of words meant to be used
  together slip through (they are harmless; clear them with an empty box).
- **Eric_Krea2 Multi-LoRA Stack:** the `word_options` field, one line per slot: `3: teal orange` (an option's name,
  number, or the start of its name). **Pick words** fills it from a list per slot.
- **Eric_Krea2 Sweep Plan** (bakeoff and stacks lines): `name #option @0.8`, and `name #each` = one cell per option
  (several `#each` in a stack line = every combination). The cell's caption keeps the `#option`.
- The chosen option's words replace that LoRA's usual words in every trigger mode except none. Stored in
  `word_options` (yours) and `card_facets.word_options` (what is in force).

## Sweep verdicts (ComfyUI + Eric_Krea2)

Which LoRAs work together, judged by eye on your own renders. Needs the Eric_Krea2 nodes.

- `python run.py sweep-plan --folders Krea2/graphic-art Krea2/art-style` picks LoRA pairs and saves a preset in
  Eric_Krea2's `sweep_presets.json` (backup first): choose it in the SweepPlan node's `sweep_preset` list. `--dry-run`
  prints without writing. `--single "name"`: one LoRA's characterization sheet (its own test prompts x three weights +
  no-LoRA). `--prompts from-loras`: the LoRAs' own test prompts. More: `python run.py sweep-plan --help`.
- **Sweeps** page (link in the card editor header, or `/sweeps`): rate each cell **1-5** stars (since 2026-10-17 the
  stars are the rating; G / B / U are gone); **Space** opens the cell beside its parts and the baseline. Each mark
  stores the cell's exact stack in the `stacks` table.
  On a trigger test, **use words** on a cell makes that cell's words the LoRA's prompt words for that checkpoint
  (Shift-click: every checkpoint). Sweep folders named `sweep_manifest.json` (older) or
  `sweep_manifest_<sweep name>.json` (since 2026-10-15) both show up.
- `python run.py stacks --pairs`: every stack and LoRA pair with its 4-5 / 1-2 / 3 star counts.
- **Rating a cell (2026-10-17).** The stars say how well the LoRA did its job on this prompt, judged against the
  baseline cell: a lovely image where the LoRA didn't show is a 2 with *style didn't show*.
  - **1-5** stars (click the same star again to take them off), **0** clears the whole mark.
  - **+** / **−**: *try stronger* / *try weaker* next time - the strength push.
  - Tags say why. **Flaws** (the LoRA's fault): signature or watermark (**S**), unwanted text (**T**), added frame,
    changes the character or face (**C**), changes the subject, changes pose, ignores the prompt (**I**), same face
    every time, faces or hands broken (**F**), anatomy wrong, artifacts (**A**), overcooked, too weak, style didn't
    show (**W**), colour cast, oversaturated, washed out, soft, unwanted NSFW (**N**). **Praise**: great colour /
    detail / composition / lighting / texture, true to its style, keeps the subject, follows the prompt, strong mood,
    clean and coherent. The letters work in the grid and the zoom view; all tags are buttons in the zoom view.
  - Rate the **baseline** cells too: they say how hard the prompt is on that checkpoint, which the analysis takes out.
  - The old good / bad / unsure is kept, filled from the stars (4-5 / 3 / 1-2), so stack counts, pairs, Follow-up and
    share sheets work as before. Marks made earlier with G / B / U and no stars stay (score 4 / 3 / 2, counted at half
    weight); the *old marks* filter shows them for re-rating.
  - Every mark records the prompt's subject and **type** (portrait / figure / creature / scene / object / abstract),
    the prompt without the LoRA words, the words added and the trigger mode, sampler and steps, and the **prompt fit**:
    how close the prompt is to the LoRA's own prompts (creator examples and Civitai prompts: same subject, same type,
    similar words; 0-1, shown on the cell). **Import workbook ratings** brings `sweep_review.xlsx` ratings in as stars.
  - Marking the same cell again updates that cell's mark; other cells and sweeps are separate observations.
- **The analysis (markmodel.py).** A star mixes four things: how good the LoRA is, how well the prompt suits it, the
  checkpoint, and how hard the prompt is anyway. Sweeps cross LoRAs x prompts x checkpoints at one seed, so these are
  separated by an additive model with shrinkage (an estimate is pulled toward average until the marks earn it - three
  ratings never make a verdict). It runs by itself a few seconds after you stop marking, from **Update the analysis**
  on the Results page, from `python run.py stacks --update` and as the refresh's *marks* step. Results go to the
  `lora_observed` table.
- **Sweep record** on each card: how sure (early / some / solid, by marks, checkpoints and prompt types); the
  LoRA's quality with prompt and checkpoint taken out; its range (what its own prompts are, prompts like its own vs
  outside its range, and a warning when your marks favour another kind of prompt than it was made for); flaws and
  praise as *k of n* against how often LoRAs get them overall (bold when at least twice as often), each flaw with its
  likely cause (how it was made / prompt range / weight); the strength range per checkpoint from your pushes and
  too weak / overcooked tags; a checkpoint x prompt-type table; and the tables by checkpoint, prompt type, subject,
  weight, words, alone / in a stack and partners. When the marks clearly point somewhere it offers a card weight
  (*fill in*, then Save) or a checkpoint's words pick (*use these words*). Nothing changes on the card until you click.
- **By prompt (2026-10-18)** on the Sweep record: one row per prompt the LoRA was rated on (per checkpoint): its
  score, the mean of the *other* LoRAs rated on the same prompt and checkpoint, and the difference - in a sweep every
  LoRA gets the same prompt, trigger mode and seed, so the difference is about this LoRA on this prompt, not about how
  hard the prompt is. Also the baseline's stars (when rated) and the prompt fit; hover a row for the whole prompt, your
  notes and tags. Best first. Once 2+ prompts are clearly above (+0.75★ or more) or below the other LoRAs, **Good
  for / Weak for** says what those prompts share (prompt type or subject when the others are not mostly the same kind,
  and words they share that the other prompts lack), with **add to notes** to put it in the card's notes (Save keeps
  it). It is worded from the prompts themselves, no LLM; with sweeps of near-identical prompts (all pin-up figures) the
  shared words are the best it can say - more varied prompts make it sharper.
- **The LoRA Suggester** reads the same analysis: a LoRA that rates above or below average (with 2+ marks' weight) is
  turned up or down by up to 12%, using its number for the checkpoint in your workflow when it has one (the node finds
  the checkpoint in the graph). A flaw you tagged in a third or more of its marks (2+ times) turns it down when it
  matters: signature, text (unless the prompt wants text), frame, unwanted NSFW, artifacts, broken faces / hands; and
  *changes the character* / *same face* when the prompt has a person. The report says so (*your sweeps: ...*).
- **Stack follow-ups (2026-10-16).** **Follow-up** on the Sweeps page lists the stacks you marked good (or 5 stars,
  or anything not bad) as Sweep Plan lines with their weights: paste them into `lora_files` with `stack_parts` =
  build up or leave one out (Eric_Krea2's Start 13). **Space** on a stack now shows the reference and every smaller
  stack of its own LoRAs the sweep made - each alone, the build-up steps or the leave-one-out pairs - fewest first.
- **Share sheets (2026-10-16).** The Sweeps page's **Share sheets** button makes pictures to post (Civitai, Reddit
  ...) from the open sweep: one labelled strip per prompt and/or each image labelled on its own, black or white,
  with each LoRA's short name, weight, creator and Civitai model number - and, if you choose, your stars (or
  verdict) and notes. Files go to the sweep's `share` folder and show as thumbnails you can open full size. Uses
  Eric_Krea2's `_sweep_sheets.py`, so Eric_Krea2 must be installed next to the catalog. Details:
  Eric_Krea2's [docs/SWEEPS.md](../../Eric_Krea2/docs/SWEEPS.md).

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
index, hash, identify, crawl, creators, images, community, pairs (optional), cards, genre-mood, triggers (optional,
shelves in `trigger_arches`), review-all, facets, style (optional, with `style_enabled`).

Not in the refresh (run by hand when wanted): `harvest` (a bigger fetch of top community images per LoRA for the
pair data; slow - run `pairs` again after it),
`sliders`, `ladders`, `cards --force` (re-card everything), `export-sidecars`, `prune-orphans` (after deleting or
moving LoRA files by hand: lists the cards and other rows left for files no longer indexed; `--apply` backs up the
catalog and removes them - your card edits, marked stacks and the move log stay).

Tests (no network, nothing written to your catalog): `python dev/verify_release.py`, `dev/verify_lora_suggest.py`,
`dev/verify_community.py`, `dev/verify_organise.py`, `dev/verify_sweepplan.py`, `dev/verify_panel_routes.py`.
