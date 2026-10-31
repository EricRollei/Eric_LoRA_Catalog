# Discover, stage 1: LoRAs you don't own, from what pairs with your best ones

2026-10-29 - status: **BUILT 2026-10-29** (lora_catalog/discover.py, /api/discover, editor Discover button, run.py pairs --wishlist taste; tests dev/test_discover.py). Added beyond the spec: a LoRA you own in another version is labelled and listed last.

## Goal

A **Discover** view in the card editor listing LoRAs you do not own, ranked by how often Civitai users stack them
with the LoRAs **you rate highly**, with the reasons shown and a link to get them. Stage 1 uses only data that is
already collected or one cached Civitai call away; stages 2 (popularity done properly) and 3 (taste match by style
vectors) come later and are out of scope here.

## What already exists (read from the code and your catalog, 2026-10-29)

| Piece | Where | State in your catalog |
|---|---|---|
| Stack harvest: top community images of each owned LoRA, the LoRAs each image used | `harvest.py`, tables `harvest_seeds`, `harvest_images` | 853 owned LoRAs harvested, 46,495 images |
| Pair counts by distinct users, lift, `score = log2(lift) x log(1 + n)`, owned flags | `lora_pairs` (per base model) | Krea 2: 1,958 pairs, Krea 2 Turbo: 1,180 |
| Unowned candidates (pairs with exactly one owned side, score > 0) | derivable | **443 unowned LoRAs**, paired with 262 of yours |
| `run.py pairs --wishlist`: sum of pair scores x (0.5 + Civitai score / 100) | `harvest.wishlist()` | CLI only, text output |
| Civitai community score for unowned versions (approval, likes per download, downloads per day; percentile among your LoRAs of the same base) | `community.quick_scores()`, cached version records | used by the wishlist |
| Your per-LoRA quality (prompt / checkpoint taken out, shrunk) | `lora_observed.quality` | 101 of those 262 partners have one |
| Your stack synergy | `lora_synergy` | 12 pairs so far |

So stage 1 is mostly: **add your taste to the wishlist ranking, and give it a screen.**

## Ranking

For an unowned LoRA X, over every owned partner P it pairs with (one `lora_pairs` row, best base model match):

```
pair_term(P, X) = pair.score x taste(P)
taste(P)        = clamp(1 + 0.5 x quality(P), 0.25, 2.0)      quality = lora_observed.quality (stars vs average)
                  1.0 when P has no quality yet (unrated: neutral, not penalised)
                  0.0 when P is one you rated poorly enough that quality(P) <= -1.5 (you would not stack with it)
rank(X)         = sum_P pair_term(P, X) x (0.5 + civitai_score(X) / 100)        civitai_score: quick_scores, 50 if unknown
```

- A LoRA used with three of your 4-star LoRAs outranks one used with three you rated 2 stars - the existing wishlist
  treats them the same.
- Civitai popularity stays a 0.5x-1.5x modifier (as now), never the main signal: your taste and co-use lead.
- Unpublished on-site training outputs stay excluded (existing `TRAINING` rule).
- Base model: default **the shelf you are on** (Krea2 shelf = pairs counted for Krea 2 and Krea 2 Turbo); a toggle for
  all bases.

## Screen

A **Discover** button in the editor header, next to Duplicates / Look-alikes. The left list becomes the ranked
unowned LoRAs (name, rank, base). Selecting one shows a **discover card** in the main pane instead of the editor:

- name, version, base model, Civitai link (opens the model page), creator if known;
- Civitai numbers from `quick_scores`: downloads, up / down, community score;
- **why it is suggested**: the owned partners it is used with, each with your stars and the pair numbers (users,
  lift), best first - e.g. "used with 4 of your LoRAs: Frazetta Style 4.5, KreaFortiche 4 ...";
- typical weights other users give it (from `lora_pairs.weights_*`);
- buttons: **Open on Civitai**, **Copy version id** (for the Civitai downloader), **Not interested** (hides it).

The usual review keys keep working where they make sense: J/K or arrows next / previous; Esc leaves Discover.

## Data and API

- `GET /api/discover?base=<shelf>&all_bases=0&top=100` -> `[{key, version_id, name, base_model, rank, civitai:{score,
  downloads, up, down}, partners:[{path, name, stars, quality, n_users, lift, score}], weights:[...]}]`
- New table `discover_dismissed (key TEXT PRIMARY KEY, dismissed_at TEXT)`; `POST /api/discover/dismiss {key}` and
  undo from the card. Dismissed LoRAs are left out of the list and the CLI wishlist.
- `quick_scores` is called only for the top 3 x `top` by pair sum (as the wishlist does), cached per version, so a
  normal open makes no Civitai calls after the first.
- CLI: `run.py pairs --wishlist` gains the taste factor (and `--no-taste` for the old ranking) so both agree.

## Freshness

The list is only as new as the last harvest. The Discover header shows "harvested N LoRAs, last <date>", and the
README tells how to refresh (`run.py harvest --only Krea2`, then `run.py pairs`). Harvesting from the UI is not in
stage 1.

## Tests

- Ranking on a small fixture DB: a candidate paired with highly rated partners outranks one with equal pair scores
  but poorly rated partners; unrated partners are neutral; quality <= -1.5 partners contribute 0.
- `--no-taste` reproduces the current wishlist order exactly.
- Dismissed keys are excluded from API and CLI; undo restores them.
- The API makes no Civitai calls when every needed version record is cached.

## Out of scope (later stages)

- Stage 2: Civitai stats for candidates not reached by co-use, downloads per day since release, NSFW-skew correction,
  browsing by base model beyond your harvest.
- Stage 3: embed candidates' preview images with CSD and rank by closeness to the looks you rate highest.
- Downloading from the screen (the Civitai downloader integration).

## Open decisions

1. Taste weights: `1 + 0.5 x quality` clamped to 0.25-2.0, and the -1.5 cut-off - fine as a start?
2. Default scope: the current shelf (proposed) or always all bases?
3. Should your positive **synergy** with a partner add to `taste(P)` (stage 1) or wait until there is more of it
   (12 pairs today)? Proposed: wait.
4. NSFW: the community score is NSFW-skewed (your earlier concern). Stage 1 shows the NSFW level on the card but does
   not filter; a toggle like the Similar-look strip's is easy to add - include it?

## Decisions (2026-10-29, Eric)

- **Wording:** "partner" = one of YOUR LoRAs that Civitai users put in the same image as the suggested LoRA. The
  screen says "used together with your ..." rather than "partner".
- **1. Taste weights:** OK as proposed (1 + 0.5 x quality, clamped 0.25-2.0; quality <= -1.5 counts 0; unrated = 1).
- **2. Scope = the base model selected in the shelf dropdown.** Krea2 selected -> only Krea2 suggestions; another base
  selected -> suggestions for that base. Needs a shelf -> Civitai base-model mapping (Civitai's names differ: the Krea2
  folder = "Krea 2" + "Krea 2 Turbo"): derived from the base_model values the harvest recorded for LoRAs in that
  folder, overridable in settings.json ("discover_bases": {"Krea2": ["Krea 2", "Krea 2 Turbo"]}). A base with no
  harvested LoRAs shows an empty Discover list with how to fill it (run.py harvest --only <folder>, run.py pairs).
  "All folders" / all-bases toggle = every base.
- **3. Synergy in the ranking:** wait - not in stage 1.
- **4. NSFW:** not answered yet; stage 1 default = show the level on the card, no filter.
