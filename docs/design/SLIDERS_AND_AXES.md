# Sliders and axes: how the LoRA Suggester dials slider LoRAs

A **slider LoRA** is a dial, not a style. A positive weight pushes the picture toward one end of an attribute (older,
heavier, wider shot, more saturated). A negative weight pushes it toward the other end. Weights often run far outside
0..1, for example −14 … +7.

The LoRA Suggester (video_prompter, node **LoRA Suggester**) never picks a slider because its description sounds like
the prompt. It reads the prompt for **phrases that ask for a place on an axis** ("wasp waist", "a very old fisherman",
"extreme close-up"). It then sets the matching slider to a signed weight.

```
prompt ──► phrases found on an axis (axes.yaml + axes_ladders.yaml)   "very old" → age +0.95
                │
                ▼
slider card for that axis (slider_cards → card_facets.slider)   THE age slider: useful −3 … +10
                │
                ▼
weight = position × the useful end on that side            +0.95 × 10 = +9.5
         (or your exact weight if you calibrated that phrase on that slider)
```

Changes to `axes.yaml` and `axes_ladders.yaml` take effect on the **next run of the node**; ComfyUI does not need a
restart. Changes to slider cards need `run.py facets` (see the end of this page).

---

## 1. The files

The pick lists live in your data folder (`python run.py check` shows it; the shipped copies are in `data/`) unless a path says otherwise.

| File | What it is | Edit it? |
|---|---|---|
| `axes.yaml` | **The hand-written master file.** For every axis: its two ends, which sampling stage it acts in, the old coarse cue lists (`neg` / `pos`), hand ladders, and the word **sets** used for context. At the end, **exact per-slider weights** (`sliders:`). | **Yes.** This is your file. |
| `axes_ladders.yaml` | **Live ladders.** Many graded phrases per axis, each with a position −1 … +1. Drafted by an LLM, cleaned up, then accepted. The Suggester reads it. | Yes, for small fixes (move or delete a rung). |
| `axes_ladders_draft.yaml` | Working copy for new or redrafted ladders. **The Suggester never reads it.** It only goes live through `run.py ladders --accept`. | Yes, before accepting. |
| `axes_ladders.yaml.bak_*`, `axes_ladders_draft.yaml.bak_*` | Automatic backups made by `--accept` and by cleanup scripts. | No. Copy one back to undo. |
| `cleanup_ladders_2026-10-09.py` | The one-off cleanup applied to the first LLM draft (a record of what was changed and why). | No need. |
| `reviews/krea2_sliders.html` | **Review page.** Every slider on the platform with its axis, ends, useful range, and every phrase with the weight it would give *on that slider*. | Generated. Re-run to refresh. |
| `reviews/krea2_sliders_draft.html` | The same page with the draft ladders merged in (draft rungs in orange). | Generated. |
| `lora_catalog/sliders.py` | `run.py sliders`: the text-only pass that writes **slider cards** (axis, direction, range) into the catalog. | Code. |
| `lora_catalog/ladders.py` | `run.py ladders`: drafting, tidying and accepting ladders. | Code. |
| catalog table `slider_cards` | One row per slider LoRA: axis, which end a positive weight moves toward, what + and − do, range, useful range, stages, confidence. In the catalog database. | Rarely (see section 6). |
| catalog column `card_facets.slider` | The finished view of the slider card that the Suggester reads. Your card-editor weights replace its useful range. Rebuilt by `run.py facets`. | No. Rebuild it. |

In video_prompter:

| File | What it is |
|---|---|
| `core/lora_suggest/axes.py` | Reads the axes files and finds phrases in a prompt. |
| `core/lora_suggest/filters.py` | The `exclude` / `pin` box syntax (section 8). |
| `core/lora_suggest/engine.py` | Picks the slider for each axis and works out the weight. |
| `dev/slider_cue_table.py` | Builds the review pages. |
| `dev/verify_sliders.py` | Self-checks (run after changing code or the format). |
| `dev/try_sliders.py` | Runs a few real prompts against your catalog → `dev/logs/try_sliders.txt`. |

---

## 2. Positions: the one idea to understand

Every phrase has a **position** on its axis, from **−1 to +1**.

- **−1** = the most extreme ask for the first end (`ends[0]`). **+1** = the most extreme for the second end
  (`ends[1]`). **0** = neutral.
- A position is **not a weight**. Each slider turns it into its own weight:
  `weight = position × that slider's useful end on that side`.
  Example: "tiny waist" is −0.71. On a slider useful from −14 to +7 it gives −0.71 × 14 = **−9.9**; on one useful from
  −2 to +2 it gives **−1.4**. One ladder therefore serves every slider on the axis.

**The firing line.** The node's `slider_min_strength` decides how big a position must be before any slider is used:

| slider_min_strength | fires at \|position\| ≥ | use |
|---|---|---|
| 1 | any | every cue counts |
| **2 (default)** | **0.35** | everyday words stay off |
| 3 | 0.65 | only strong asks |

So a phrase at **0.30** is deliberately "off by default". That is where everyday words go that usually describe a
style rather than a degree: "detailed", "dark", "illustration", "comic", "busy", "golden hour". They are kept, rather
than deleted, so the report can show that they were seen.

**Modifiers.** "very / extremely / super / incredibly / really / highly" in front of a phrase multiply it by 1.25.
"slightly / a little / somewhat / a bit / mildly" multiply it by 0.6. A modifier never lifts an everyday (below-0.35)
word over the line, so "very dark" stays off.

**Conflicts.** If one prompt has phrases on both sides of an axis ("an old man and a teenage girl"), that axis is left
alone, and the report says so.

---

## 3. Writing a phrase (cue)

The same forms work in `axes.yaml` (ladders and `neg`/`pos` lists) and in `axes_ladders.yaml`.

| Form | Meaning | Example |
|---|---|---|
| `elderly` | the phrase as whole words, anywhere in the prompt | "an elderly man" |
| `"old + @people"` | only when a word from the set `people` is within **4 words** of it, before or after | "an old fisherman" ✓, "an old car" ✗ |
| `"wet + skin\|hair\|body"` | the same, with the context words written inline | "wet hair" ✓, "wet street" ✗ |
| `"thin + @people\|@body"` | several sets and words can be mixed | |

Rules:

- **Case does not matter.** Matching is on **whole words**: `old` does not match "bold" or "golden".
- **The longest phrase wins.** "very old" is used instead of "old" when both are present.
- A plural of a context word also counts ("hairs" for "hair", "men" only if "men" is in the set).
- **Quote any phrase that contains `+`, `:`, `#`, or starts with `@`** (YAML needs it): `"old + @people": 0.75`.
  Plain phrases may be quoted or not.
- Use a context word for **any word with a common other meaning**: old (town, car), dark (fantasy), wet (street),
  tall (building), sleek (car), thick (fog).
- Keep **one word on one axis.** If "faded" is on both saturation and contrast, one prompt word moves two sliders.

### Word sets

At the top of `axes.yaml`:

```yaml
sets:
  people: [man, men, woman, women, person, fisherman, ...]
  body:   [body, build, figure, physique, arms, legs, ...]
  light:  [lighting, light, lit, scene, room, shadows, night, alley, ...]
  colour: [colour, colours, color, colors, palette, tones, hues, tone]
  place:  [street, room, city, town, station, square, beach, ...]
```

Add words to a set and every cue that uses `@set` picks them up. Add a new set the same way, then use it as
`@newset`.

---

## 4. Where a phrase's position comes from (precedence)

For each axis, highest wins:

1. **A `ladder:` written in `axes.yaml` itself** (hand ladder). Waist has one. These rungs are never overridden.
2. **The axis's ladder in `axes_ladders.yaml`.** When an axis has a live ladder, it **replaces** that axis's old
   `neg`/`pos` lists completely, so a rung you delete from the ladder does not come back from the old list.
3. **The `neg` / `pos` lists in `axes.yaml`** (strength 1 / 2 / 3 = position 0.33 / 0.67 / 1.0), only for axes with
   no ladder.

Separately, a **calibrated slider** (section 5, `sliders:`) uses your exact weight for the phrases you listed; all
other phrases on that slider still use position × range.

**Which range a slider uses** (the "useful end" in the formula), highest wins:

1. Your **weight low / high in the card editor** for that LoRA.
2. The **slider card's** useful range, when its confidence is 0.5 or more.
3. The node's **`slider_range`** widget (default ±3; 0 … +3 for a one-sided slider).

The review page shows, for every slider, which of these it is using.

---

## 5. How to…

### Add a phrase, or move one

Edit `axes_ladders.yaml` (or the `ladder:` in `axes.yaml` for an axis that has a hand ladder):

```yaml
axes:
  age:
    ladder:
      "in her seventies": +0.82
      "silver-haired": +0.6
```

Save, then refresh the review page: `python dev\slider_cue_table.py krea2` (from the video_prompter folder). The
node picks up the change on its next run.

### Stop a word from firing a slider

Either set its position to **0.3** (it is still read but does nothing by default), give it a context word
(`"dark + @light"`), or delete the line. Prefer 0.3 for style words; delete words that are simply wrong for the axis.

### Calibrate one slider exactly (your own numbers)

At the end of `axes.yaml`:

```yaml
sliders:
  waist slider:                 # part of the LoRA's name or file name, whole words
    axis: waist
    weights:
      wasp waist: -14
      tiny waist: -10
      hourglass waist: -8
      thin waist: -5
      thick waist: 3
```

- Every phrase listed here **must also exist as a cue on that axis** (in a ladder or the `neg`/`pos` lists).
  Otherwise the review page shows a red "axes.yaml" warning at the top.
- A calibrated slider is preferred over other sliders on the same axis for the phrases it calibrates.
- Good practice: when you calibrate, also set the shared ladder positions to match (weight ÷ that slider's end), so
  other sliders on the axis follow your judgement too. For the waist slider that is `wasp waist: -14 / 14 = −1.0`,
  `thick waist: 3 / 7 = +0.43`.

### Add a new axis

1. In `axes.yaml` under `axes:`, add:
   ```yaml
     freckles:
       ends: [no freckles, heavily freckled]    # [negative end, positive end]
       stages: late                             # early = shape / body / framing; late = surface / skin / colour; all = overall light or realism
       ladder:
         "a few freckles": 0.4
         freckled: 0.6
         "heavily freckled": 0.9
   ```
2. Point the slider at it. A slider card's axis comes from the slider pass. Re-run it for that LoRA's folder:
   `python run.py sliders --only Krea2/sliders/<folder> --force`. The model now sees your new axis in its list.
   Then run `python run.py facets`.
3. `python run.py sliders --gaps` lists axes that slider cards use but `axes.yaml` lacks. Those sliders can only be
   pinned until the axis is added.

### Draft a ladder with the LLM

`python run.py ladders --draft --axis freckles` (LM Studio running; model in `settings.slider_model`). It writes to
`axes_ladders_draft.yaml` only. Check it on the draft review page
(`python dev\slider_cue_table.py krea2 --draft`), edit, then:

- `python run.py ladders --tidy` keeps words you marked everyday (strength 1 in `axes.yaml`) at 0.3.
- `python run.py ladders --accept` copies the draft over `axes_ladders.yaml` (the old one is backed up as
  `axes_ladders.yaml.bak_<date>`).

**Note:** `--accept` replaces the whole live file with the draft. Run `--draft` without `--axis` only when you want
to redraft every axis that has no hand ladder. To add one axis to a live file you have edited, copy that axis's block
from the draft into `axes_ladders.yaml` by hand instead.

### Undo

Copy a `.bak_*` file back over `axes_ladders.yaml` (or `axes_ladders_draft.yaml`). The LoRA Catalog's
`backup_*` folders hold earlier versions of `axes.yaml`.

---

## 6. Slider cards (what the catalog knows about each slider)

`python run.py sliders` reads each slider's Civitai description, trigger words and the weights used in its gallery
images. It asks a text model for:

- `axis`: which axis it moves
- `positive_toward`: which end a **positive** weight moves toward. This catches sliders trained backwards; the
  review page marks them "trained backwards".
- `positive_pole` / `negative_pole`: what + and − do. An empty negative pole means one-sided: only positive weights
  are used.
- `range` and `useful`: the extremes, and where results stay clean.
- `stages` and `confidence`.

Fixing a card:

- **Range:** set weight low / high in the card editor (`launch.bat`, or the sidebar panel). This wins over the card. Saving in the
  editor updates the Suggester immediately.
- **Axis or direction wrong:** re-run with `--force` for that LoRA's folder (above). Or edit the row in
  `slider_cards` with a SQLite tool, then `python run.py facets`.
- **Not really a slider:** set `is_slider` to 0 in its `slider_cards` row, then `python run.py facets`.

Commands:

```
python run.py sliders                          every slider LoRA without a slider card
python run.py sliders --only Krea2 --force     redo one shelf or folder
python run.py sliders --gaps                   axes used by cards but missing from axes.yaml
python run.py facets                           rebuild what the Suggester reads (after any slider-card change)
```

---

## 7. Checking your changes

- **Review page:** `python dev\slider_cue_table.py krea2` (video_prompter folder) →
  `<data folder>/reviews/krea2_sliders.html`. Per slider: every phrase, its position (grey), the weight it gives
  (blue), "exact" (green) for calibrated weights, greyed when below the firing line. Problems in `axes.yaml` show in
  red at the top.
- **Real prompts:** `python dev\try_sliders.py` → `dev\logs\try_sliders.txt`. Edit the prompt list at the top of the
  script to try your own.
- **The node's report output:** every run lists `axes read: …`, each slider it set and why, axes it left alone (below
  the line, conflicting cues, its sliders held back by the NSFW gate or an exclude), and which pins matched what.
- **Self-checks:** `python dev\verify_sliders.py`.

Things that commonly surprise:

- A slider whose card is rated nudity or explicit is **held back at NSFW off**, like any other LoRA. The report
  names it. Lower the card's rating in the card editor if the rating is wrong.
- Under a hard rule, a younger-age dial is never used together with nudity or explicit content.
- Sliders do not count toward `max_loras` or the `weight_budget`. `slider_max` limits them separately.
- Slider trigger words are not added to the prompt; a slider works by its weight.

---

## 8. The node's `exclude` and `pin` boxes (quick reference)

One entry per line, or comma-separated. Whole words. Case does not matter.

| Entry | In **exclude** | In **pin** |
|---|---|---|
| `vallejo`, `waist slider` | every LoRA with these words in its name or file name | the best single match (stars, favourite, confidence); the report lists the others it matched |
| `folder:sliders` | everything in folders with these words | not supported |
| `role:treatment` | every LoRA in that role | not supported |
| `axis:age` | every slider on that axis | needs a value |
| `name=weight` | (value ignored) | that LoRA at this weight: `waist slider=-8` |
| `axis:age=+3` | (value ignored) | the best slider on that axis at +3 |

A pin overrides an exclude and the NSFW gate.
