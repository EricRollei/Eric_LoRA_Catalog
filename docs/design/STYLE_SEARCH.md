# Style search (CSD) - spec for parts 1 and 2

Status 2026-10-12: parts 1 and 2a-2d are built and tested on Eric's library: 4,829 LoRAs have vectors, and
dev/verify_style.py passes. 2e (widened text search) is not built. The feature is off by default (`style_enabled`);
since 2026-10-14 it is documented for users in USER_GUIDE.md ("Style vectors (CSD)"). Part 3 (Suggester
`style_image` input) has since been built in the Advanced Suggester; part 4 (style blends) is only sketched at the end.

## Why

The catalog finds LoRAs by words: card, tags, triggers, names. Nobody can search by what a LoRA *looks like*, and
Civitai can't either. CSD (Contrastive Style Descriptors, Somepalli et al., ECCV 2024, arXiv 2404.01292) is a CLIP
ViT-L/14 fine-tuned to describe style while ignoring subject. The pilot (A:\LoRA_Catalog\pilot_effect_vectors) showed:

- **Previews alone are enough.** One LoRA's previews agree at a mean cosine of 0.50; previews of different LoRAs
  agree at 0.12. No rendering is needed.
- **The neighbours are right.** Across 579 Krea2 LoRAs, both Mignola LoRAs match (0.87), both Klimt LoRAs (0.84),
  Frazetta sits with Vallejo and Brom, Beksinski with Zawadzki, and Rackham with Rackham.
- **It's fast.** About 5,700 images took 70 s on one GPU, after a 12 s model load.
- **The limit:** photoreal concept, subject and NSFW LoRAs all collapse into one "realistic photo" look (0.90+ with
  each other). CSD is right about that, but it means style search only helps look-type LoRAs (cards with kind
  style_medium, photo_look or lighting_colour).

## Part 1 - storing style vectors

### Model files (optional feature)

- `lora_catalog/style.py` is new. It holds:
  - the CSD model as its own small ViT-L/14 module, with the same layout as OpenAI CLIP's visual tower and
    QuickGELU, about 80 lines of plain torch;
  - the CSD preprocessing (resize 224 bicubic, centre crop, CLIP mean/std), done in PIL so torchvision isn't needed.
  - **No open_clip, no transformers.** The pack's dependency list doesn't change.
- **Optional.** When torch or the weights are missing:
  - `style.available()` returns the reason;
  - the refresh step logs one line and is skipped;
  - the editor hides the style controls.
  - torch is imported only inside style.py.
- **Weights.** `python run.py style-setup` does the download and conversion:
  - it downloads `pytorch_model.bin` (2.4 GB) from huggingface.co/tomg-group-umd/CSD-ViT-L (CC-BY-4.0; the code is
    MIT);
  - it converts the file to `DATA_DIR/models/csd_vit_l.safetensors` (1.2 GB of weights only), then recycles the
    .bin.
  - The conversion uses torch's safe unpickler. It allows only what the checkpoint contains (argparse.Namespace and
    numpy scalars), never `weights_only=False`.
  - The settings key `style_model` (blank = that default path) lets someone point at their own copy. Eric's copy
    stays at H:\semantic_search\models\CSD-ViT-L.
- **Device.** The settings key `style_device` takes auto, cuda:N or cpu (default auto).
  - The model runs in fp16 on GPU and fp32 on CPU.
  - CPU works, at roughly 0.3 s per image: a whole library in 30-60 min, once.
- **Validation.** The hand-written ViT must reproduce the pilot's open_clip vectors (cosine > 0.999 on the pilot
  images). dev/verify_style.py checks this.

### Table

```sql
CREATE TABLE IF NOT EXISTS style_vectors (
    key        TEXT PRIMARY KEY,   -- coalesce(lower(sha256), path): survives moves, like card_edits
    path       TEXT,               -- last known path
    model      TEXT,               -- 'csd-vit-l' (room for a better model later)
    source     TEXT,               -- 'previews' (part 1); 'renders' later from the overnight sweeps
    n_images   INTEGER,
    agreement  REAL,               -- mean pairwise cos of its own previews; NULL for 1 image
    vec        BLOB,               -- 768 x float16, unit length (1.5 KB; whole library < 10 MB)
    images_sig TEXT,               -- hash of the image list + mtimes: re-embed only when previews change
    updated_at TEXT
);
```

Per-image vectors go to a cache file, `DATA_DIR/cache/style_images.npz`, not the catalog. That way a LoRA whose
preview list changes re-embeds only its new images.

### Refresh step `style`

- **Position:** after `facets`, as step 12. It needs the images from step 6.
- **Failure:** a failed step is logged and the refresh carries on, like the internet steps. A missing style model is
  never a reason to stop a refresh.
- **Images:** for every LoRA it calls `cards.preview_images(path, 12)`, the same images, in the same order, that the
  editor shows. That means:
  - least-NSFW first;
  - in flat folders, only images named after the LoRA;
  - preview videos are skipped.
- **Work:** it embeds only images not yet in the cache and writes rows whose `images_sig` changed.
- **Speed:** 8 decoder threads and batches of 32. After the first run, a refresh with 20 new LoRAs takes seconds
  plus the 12 s model load. The model unloads at the end of the step.
- **Command:** `python run.py style [--limit N] [--force]`. The `check` command reports whether the style model is
  set up.

### LoRAs whose previews can't tell them apart

When several LoRAs end up with the same image set (same images_sig), they keep one vector each, but the UI shows
them as one entry: "same previews as X, Y". It doesn't list them as twins. Only renders can separate them.

## Part 2 - the card editor

Every list result keeps the existing filters (shelf / folder / status / rating) and the existing list layout. The
score column shows similarity instead of the search score.

### 2a. "Similar look" on a card

- **What:** a strip under the example images showing up to 8 LoRAs: first preview thumbnail (existing /img), name,
  kind and similarity. Clicking one opens that card.
- **Scope:** same base model by default; a toggle shows all base models. All base models is useful for discovery:
  "you have this look for SDXL - is there a Krea2 one?".
- **Kinds:**
  - For look-type cards, only look-type neighbours by default; a toggle shows every kind.
  - For a non-look card the strip is collapsed, with a note: "previews show a plain photo look - style
    neighbours are not meaningful".
- **Agreement:** shown on the card as "Previews agree: high / mixed / low" (>= 0.55 / 0.35-0.55 / < 0.35). Low
  usually means a concept LoRA, or a gallery made with other LoRAs.
- **Endpoint:** `GET /api/similar?path=&n=8&scope=base|all&kinds=look|all` returns `[{path, name, kind, base, score,
  same_previews}]`.

### 2b. Search by image

- **How:** drop or paste an image onto the search box. A chip appears with its thumbnail and an x to clear it, and
  the list shows the LoRAs whose look is closest to that image.
- **Endpoint:** `POST /api/style_search?<filters>`. The body is the image bytes (Content-Type image/*). It returns
  the same shape as /api/queue with a q: `{items, matched, mode: "image"}`.
  - The query image is embedded once; an image cosine against a LoRA's signature tends to run lower than
    signature-vs-signature (0.4-0.7 for a good match).
- **Model in the server:**
  - It loads on the first image search (about 12 s; the chip shows "loading style model...").
  - It unloads after 10 idle minutes (settings key `style_idle_unload_min`).
  - Without torch: the endpoint returns 409 with the reason, and the chip says so.
  - 2a and 2c never need the model, only the stored vectors.

### 2c. `like:` in the text search

- `like:mignola` finds the best name match, then lists the LoRAs closest to it in look. It can be combined with the
  filters, e.g. `like:frazetta` with the Krea2 shelf. It's the keyboard version of 2a.
- In search.py: `like:` is parsed before FTS; it resolves the name with the existing name column, then ranks by
  vector.

### 2d. Look-alikes queue

- **What:** next to the duplicates queue (`dupes=1`), a "look-alikes" queue lists pairs of look-type LoRAs at or
  above 0.85, highest first, each pair shown side by side. It's for culling: Brom vs Frank Frazetta, the Krea* style
  family, Underwater vs Freedive.
- **Endpoint:** `GET /api/style_twins?min=0.85&base=` returns `[{a, b, score}]`.

### 2e. Typed search widened (optional, small)

- Under normal text hits, a second section "Similar look (not mentioned in the card)". It takes up to 6 neighbours
  of the top 3 hits that the text search didn't find.
- It catches cards that never name the artist or style you typed.

### Maths without numpy

The pack doesn't list numpy. When numpy is present (always in ComfyUI) it does the maths; otherwise a plain-Python
dot product over 5,000 x 768 values takes about 1 s, which is fine for 2a, 2c and 2e. The vectors load once per
catalog change, using the same stamp idea as SearchIndex.

## Files touched

| file | change |
|---|---|
| lora_catalog/style.py | new: model, preprocessing, embed, setup/convert, vector store read/write, similar(), twins() |
| lora_catalog/store.py | style_vectors table in the schema + migration |
| lora_catalog/refresh.py | step `style` after `facets`, in the carry-on-on-failure set |
| lora_catalog/__main__.py | commands `style`, `style-setup`; `check` reports the style model |
| lora_catalog/server.py | /api/similar, /api/style_search, /api/style_twins; lazy model + idle unload |
| lora_catalog/search.py | `like:` |
| lora_catalog/editor.html | similar strip, image drop on search, like: help text, look-alikes queue |
| lora_catalog/config.py | settings style_model, style_device, style_idle_unload_min |
| settings.example.json, README, USER_GUIDE | the optional feature, its 1.2 GB download, CC-BY credit |
| dev/verify_style.py | new: vectors match the pilot; Mignola / Klimt pairs rank 1; endpoints answer |

Untouched: the Suggester, cards, card_facets. ComfyUI nodes' widgets are untouched.

## Order and rough size

1. style.py + table + refresh step + commands + verify (half a day). Run it on your library, then check the
   neighbours in the report.
2. /api/similar + the similar strip + agreement (2a) and `like:` (2c).
3. Image search (2b).
4. Look-alikes queue (2d), widened search (2e).

## Later (not in this spec)

- **Render match** from the overnight sweep: `source='renders'` rows plus a per-LoRA "needs prompting" flag.

## Decisions (Eric, 2026-10-12)

1. "Similar look" defaults to the **same base model** (Krea2 with Krea2, Flux2 with Flux2), with an "all bases" toggle.
2. **Opt-in.** `style_enabled` is false by default. The refresh step and the editor controls do nothing until it is
   on. `style-setup` downloads only when the user runs it (or with `--from <file>` to convert a copy they already
   have).
3. **Try it first.** It's built in the pack but left out of the README and USER_GUIDE until tried; it ships later.
4. **Non-look LoRAs** show the strip with a warning ("previews are a plain photo look - neighbours mean little").
   The strip has an **NSFW toggle**: NSFW neighbours are hidden by default (card nsfw level >= nudity). A toggle on
   the strip shows them.
## Parts 3 and 4 - the Suggester (spec for review, 2026-10-12)

Not built yet. Both read the stored vectors (part 1); only part 3 needs the model. Nothing changes when style
search is off, or when the new input is left unconnected.

### Measurements behind the numbers

- **One image against LoRA signatures** (4,033 preview images, 367 Krea2 look-type LoRAs, leave-one-out):
  - The image's own LoRA ranks 1st 64% of the time, top 5 86%, top 10 90%.
  - Raw cosines overlap: own LoRA median 0.69 (p25 0.59), best other LoRA median 0.63 (p75 0.69).
  - So a fixed cutoff can't decide; look has to be scored **relative to the other candidates** for that image.
- **Your good / bad marks on stacked pairs:** none yet (the Sweeps page has no marked pairs). Blend thresholds can't
  be fitted to your judgement yet; see 4c.

### Part 3 - `style_image` on the LoRA Suggester

**3a. Node.** Two optional inputs, appended after `slider_min_strength`, so saved workflows keep every value in place:
- `style_image` (IMAGE): one or more reference images. A batch is averaged into one look, which is a way to say "this
  kind of look" with several examples.
- `style_image_weight` (FLOAT 0-1, default 0.5): how much the look decides against the prompt words. 0 = report
  only. 1 = look only.

**3b. Model inside ComfyUI.**
- It runs on ComfyUI's own device (comfy.model_management), not `style_device`. Inside ComfyUI torch is already
  loaded, so nvidia-smi numbering can't be applied.
- It stays loaded between runs, about 0.6 GB in fp16, and unloads after `style_idle_unload_min`.
- The vector is cached per image hash, so re-running the same reference costs nothing.

**3c. Scoring (engine.py).** For candidates in the look roles (style, transform, treatment) that have a vector:

    cos  = image vector . card vector
    look = clamp((cos - median) / (top - median), 0, 1)     median / top over this platform's look-type cards
    score = (1 - w) x score + w x look                     w = style_image_weight

- **The relative scale is deliberate.** The best-matching LoRA gets 1.0 and the median gets 0, whatever the image's
  raw cosines are.
- **The image alone can pick a style.** With no style words in the prompt, w 0.5 gives the best match 0.5, above the
  0.35 threshold.
- **Prompt words still count.** A LoRA the text matches strongly and the image matches moderately can win.
- **Other roles** (detailer, subject, slider, environment, lighting) are untouched.
- **Report:** a line per look pick ("look 0.71 to style image, #2 of 367"), plus the 5 closest looks even when they
  weren't picked, so you can see what the image pointed at.

**3d. Without style search** (off, or no weights): the input is ignored and the report says why. No error.

### Part 4 - style blends with look similarity

**4a. Today.** A second style is allowed only when its card family (anime, painterly, photographic...) matches the
first's, and then it runs at blend_ratio x its weight. Families come from pick-list terms, and many cards have
none, so good blends are refused for "family: none" and loose ones pass.

**4b. With vectors on both styles:**
- **cos >= `style_blend_max` (default 0.88): refused.** "Same look twice" doubles the weight of one look. These are
  the look-alike pairs.
- **`style_blend_min` <= cos < max (default 0.45): allowed as a blend**, whether or not the families match. The
  report says "related look 0.62".
- **cos < min: the family rule decides, as today.** CSD only widens what the family rule allows and adds the
  redundancy refusal; it never blocks a same-family pair that isn't redundant.
- **Either side has no vector: the family rule, as today.**
- The settings keys `style_blend_min` and `style_blend_max` go in the settings.json "suggest" section.

**4c. Calibrating 0.45 / 0.88.**
- Both are first guesses from the Krea2 neighbour lists: above about 0.88 the pairs are the same artist or one
  creator's family; around 0.45-0.6 they are related schools.
- To fit them to your eye: a stack sweep with 12 Krea2 style pairs spread over cos bands (0.3 / 0.5 / 0.7 / 0.9),
  on 2 prompts. That's about 25 cells, roughly 30 min. You mark them on the Sweeps page, and the cut-offs come from
  your good / bad marks.
- `run.py sweep-plan` would get a `--by-look` option to pick the pairs.

### Files

| file | change |
|---|---|
| nodes/lora_suggest_node.py | 2 appended optional inputs; embed the image (style.py) and pass the vector on |
| suggest/engine.py | SuggestRequest.style_vector / style_weight; look score in candidates; blend rule; report lines |
| suggest/catalog.py | LoraCard.style_vec, read from style_vectors (no torch) |
| suggest/settings.py | style_blend_min / style_blend_max |
| lora_catalog/style.py | device override for use inside ComfyUI; per-image-hash cache |
| lora_catalog/sweepplan.py | `--by-look` pair picking (4c) |
| dev/verify_lora_suggest.py | cases: image only, image + prompt, w=0, style off, blend refused / allowed |

### Questions

1. Is `style_image_weight` default **0.5** right, or should the prompt dominate by default (0.3)?
2. A batch of images = **one averaged look**: OK? (The other reading is "best match to any image".)
3. Run the **calibration sweep** (4c) before trusting 0.45 / 0.88, or ship the guesses and adjust later?
4. Should the image also steer the **lighting** role? CSD sees palette and light to a degree, but it's a style model.
   I'd leave lighting to the words.


### Decisions on parts 3 and 4 (Eric, 2026-10-12)

1. `style_image_weight` defaults to **0.4**. An image alone still gives its best match 0.4, above the 0.35 threshold.
2. **Several images: both ways, as an option** (`image_mode`), from a leave-one-out test on 348 Krea2 style LoRAs:
   - **Same look (k images of one LoRA): average wins.** Own LoRA ranked 1st: 1 image 65%, average of 2 77%, average
     of 4 84% (best-match: 70% / 73%).
   - **Two different looks (one image each from two LoRAs): best match wins.** Neither look in the top 10: average 47
     trials, best match 12. Both in the top 10: average 259, best match 352.
   - So `image_mode`: **one look (average)**, the default, for several examples of a single style; **each image
     (best match)** for references that differ.
3. **Run the calibration sweep first.** Preset "Calibration - style blends by look (CSD)": 12 Krea2 style pairs, 3 per
   band (far 0.32-0.34 / related 0.47-0.59 / close 0.67-0.74 / near-twin 0.86-0.89), 2 prompts, 74 cells, about
   1.5 h. Pairs and cosines are in A:\LoRA_Catalog\pilot_effect_vectors\blend_calibration_pairs.json.
4. **Lighting stays with the prompt words**; the image steers style, transform and treatment only.
5. **A new node, "LoRA Suggester (Advanced)"**, instead of changing the existing one:
   - The existing node, its inputs and its behaviour stay exactly as they are.
   - The engine gains only fields that default to off, so the old node calls it unchanged.
   - The look-based blend rule (part 4) is used only by the Advanced node.

### Part 3 built (2026-10-12): LoRA Suggester (Advanced)

- **Files:**
  - `nodes/lora_suggest_advanced_node.py` (new node "EricLoRASuggesterAdvanced");
  - `suggest/engine.py` (look_scores, score_card look evidence, request fields);
  - `suggest/catalog.py` (load_style_vectors);
  - `lora_catalog/style.py` (device override, multi-image embedding with a content-hash cache);
  - `__init__.py` (registration);
  - `dev/verify_suggest_advanced.py`.
  - Backups: `docs/backup/*_pre_advanced_2026-10-12`.
- **Change from 3c: the look counts as evidence, not as a blend of final scores.** A blend let the engine's own
  penalties push an image-only pick under the threshold. Klimt's image: best look 0.4, x0.8 "may add nudity" = 0.32,
  so no pick.
  - The look is now added to the card's raw evidence: raw += 12 x weight x look, where look is (cos - median) /
    (best - median), clamped to -0.5..1. A look below the median counts against.
  - Then score = 1 - exp(-raw/3), and the usual multipliers apply.
  - At weight 0.4 the best look alone scores 0.80, about the same as a strong word match.
- **Plain node unchanged:** the same picks on 24 fixed prompt/seed runs before and after. Report text differs only in
  the order of equal-rarity words, which already varied between Python processes.
- **Tests (verify_suggest_advanced.py, all ok):**
  - Inputs: the Advanced widgets come first, then the plain node's inputs in order.
  - Image only: the reference's own LoRA is the closest look 4/4, and picked 2/2 for cards below nudity.
  - Weight 0 gives the plain node's picks; so does no image.
  - Image + prompt: the look shows in the reasons.
  - Two different looks with each image (best match): both appear in the closest 5.

### Open points for Eric (part 3)

1. **The "may add nudity" rule against a style image.** A nudity-level style card (Klimt, Araminta Frazetta) is turned
   down x0.8 when the prompt doesn't ask for nudity. With its own image as the reference it then loses to the next
   closest look (Hiroshige, Brom). That's the existing NSFW rule. Should a style image the card matches lift it, or
   leave it as is?
2. **An image alone fills every look slot.** With no prompt words, the image picks 2 styles plus a transform and a
   treatment in the same look (Mignola image: Mike Mignola Style + //MIKE MIGNOLA// + KreaComBook + Arcana Press).
   Part 4's near-twin refusal (cos >= ~0.88) would drop the second Mignola. Lower style_max, transform_max and
   treatment_max limit the rest.

**Decided (Eric, 2026-10-12): a style image lifts the "may add nudity" turn-down.**
- A nudity-rated (2) look card whose relative look match is >= 0.5 (engine LOOK_LIFTS_NUDITY) isn't turned down
  x0.8. The report says "may add nudity - not turned down: it matches the style image".
- The gates are unchanged: explicit (3) cards stay out unless the node's nsfw setting allows them, and the plain node
  is unaffected.
- Tests (variety 0): own LoRA picked from its own image 4/4 (Mignola, Klimt, Araminta Frazetta, Semenov). At the
  default variety 0.2, picks within about 0.1 trade places by seed, as designed: Araminta 0.79 vs Brom 0.77.

## Render check: does a LoRA show its look in our own renders? (built 2026-10-13)

The preview signature says what a LoRA's creator showed. The render check says what it does on Eric's setup.

- **Command:** `python run.py style --renders <sweep folder> [--kind auto|neutral|creator]`.
  - Reads `sweep_manifest.json` and embeds each render (cached in `style_images.sqlite`).
  - Single-LoRA cells only; stack cells are skipped. `auto` = creator for combos sweeps, else neutral.
- **Per LoRA cell (table `style_renders`, one row per LoRA, sweep, prompt and strength):**
  - `match` = render vs the LoRA's own preview signature; `base_match` = the no-LoRA render of the same prompt vs it.
  - `lift` = match - base_match.
  - `rank` = 1 + the same-platform look LoRAs (one per preview set) closer to the render than its own LoRA.
  - **shows** = rank <= 10 (SHOWS_RANK) and lift >= +0.05 (SHOWS_LIFT).
- **Verdict per LoRA** (`style_vectors.render_verdict` + `render_note`, from all its rows):
  - `shows` - shows on a plain (neutral) prompt;
  - `needs_words` - shows only with its creator's prompt;
  - `weak` - has creator rows but never shows;
  - `unclear` - only neutral rows, none show: candidates for the second pass.
  - The preview step now upserts `style_vectors`, so new previews don't erase a verdict.
- **Second pass:** `python run.py style --second-pass [--arch Krea2] [--prompts 1] [--limit N] [--nsfw] [--dry-run]`.
  - A combos preset for the `unclear` LoRAs: each LoRA's creator prompt as a no-LoRA cell plus a LoRA cell at its
    card weight. Triggers: catalog.
  - Creator prompt: the Civitai example images first (lowest NSFW level), then `community_prompts` source creator.
    `<lora:...>` tags are removed; prompts over 1,200 characters are cut at a sentence. R/X prompts and NSFW cards
    only with `--nsfw`.
  - Eric_Krea2 SweepPlan now applies LoRA cells in combos (lora_mode none + lora_file/lora_stack in a combo:
    bakeoff/stacks).
- **Card editor:** a badge in the "Similar look" header (shows its look / needs its style words / look weak in renders
  / render check: unclear); the hover text has the note.
- **LoRA Suggester (Advanced) only:** a picked style/transform/treatment LoRA with verdict `needs_words` gets up to 3
  words from its card's medium, rendering and line facets (not already in the prompt), added with the triggers. The
  report has a "style words for ..." line. The plain node is unchanged (same picks on 24 runs, old vs new code,
  same data).
- **First runs:**
  - Pilot A (neutral, 10 LoRAs x 4 prompts): shows 5 (Ballpoint Pen, ink lore, He Who Burns, Vintage Film Photo,
    Dali); unclear 5 (Chalk Markers, Cyanotype, graphite sketch, mikkoph, Esoterica).
  - Chalk diag (creator): Liquid Chalk Markers is needs_words. Its creator prompt puts its own look rank 1, but the
    no-LoRA render of that prompt is already chalky (lift +0.05).
  - Second-pass dry run: 3 LoRAs, 6 cells. mikkoph was left out because its only creator prompt is Mature.
- **Known limit:** card facet words are coarse. For Chalk Markers they are "marker, drawing", while the words that
  work are "streaky chalk drawing on a black chalkboard". A better source would be the leading style phrase of the
  creator prompt.

## LoRAs named in an image's metadata (built 2026-10-13)

Images often record which LoRAs made them. The reader is `lora_catalog/image_meta.py`:
- **What it reads:**
  - ComfyUI PNG graphs: every LoRA-loader input, including stack / multi-slot / rgthree / LoRA Manager loaders.
    Bypassed or switched-off slots are skipped.
  - A1111 / Forge text: `<lora:..>`, "Lora hashes", "Hashes", "Civitai resources".
  - Civitai JPEG / WebP EXIF comments.
  - Eric_Krea2 sweep-cell PNGs.
- **Matching to the library,** best first: sha256 or AutoV2 hash, then Civitai version id, then file name. LoRAs you
  don't own are listed with their Civitai link.
- **Suggester (Advanced) only** (Eric's option b):
  - The node follows the `style_image` wire back to the Load Image file(s), through batch nodes, and reads their
    metadata. The IMAGE input itself carries no metadata.
  - Owned LoRAs get strong evidence for their role: engine `USED_RAW = 8`, about as much as naming the person a LoRA
    depicts.
  - They count as named, so no "thin evidence" or "particular character" turn-down. They also lift the "may add
    nudity" x0.8.
  - Slot limits, NSFW gates, exclude, and card weights all still apply. The image's weight is shown in the report, not
    used.
  - Utility helpers (turbo LoRAs, filter bypasses) are reported but never boosted.
  - The report lists every LoRA found, with yours / not yours.
  - The plain node is unchanged: same picks on 24 runs, old vs new code, same data.
- **Card editor:** an image search (drop, paste or file) also shows "made with (from the image's metadata)". Click
  one of yours to open it; LoRAs you don't own link to Civitai. This works with style search off too.
- **Tests:** `dev/verify_image_loras.py` (all ok):
  - an Ultra render's stack, a sweep cell, and a Civitai JPEG (hash match plus not-owned);
  - a graph walk through a batch node;
  - evidence picks the image's style LoRA on an unrelated prompt; utility LoRAs are not boosted.
- **Side finding (scan of the downloaded showcase images):**
  - Stock Krea 2 Turbo (fp8 / mxfp8 / int8) is the most common base.
  - Kreamania, Kreation and other fine-tunes together are about as common.
  - Raw plus a turbo LoRA is rare.
  - The TextFusion Refusal Reduction LoRA rides along in about 300 images, and detailers are common.
