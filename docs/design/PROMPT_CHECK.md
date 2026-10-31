# Prompt check: a vision model scores how well each sweep image follows its prompt

2026-10-30 - status: **PROPOSED**, no code written. Agree on this before any edit.

## Goal

Every sweep cell gets an automatic **prompt score** (0-100) and a list of what is **missing**, from the local vision
model, so reviewing a 100-prompt sweep does not mean reading every prompt against every image. It informs your rating;
it never sets your stars or tags.

## What the first test showed (2026-10-30, qwen/qwen3-vl-8b in LM Studio, 3 cells of the latest sweep)

One call per image, image scaled to 896 px, temperature 0.1: **4-6 s per image** (11 s for the first, model loading).

| Cell | Score | What it said |
|---|---|---|
| Bradhamel bone-carapace figure | 0 | all 6 requirements missed: "a nude, screaming figure with mechanical limbs ... completely failing to match the biomechanical carapace" |
| curved medieval town under the moon | 70 | 5 of 6 met; "the magical blue-white energy spires are missing" |
| sailor and crustacean monster | ~100 | boat, cowering sailor, frayed linen, barnacled carapace, claws, crab legs - all yes |

The instructions used in that test (one step: the model wrote its own list per image):

> You check whether an image matches its prompt. First list the 6 most important visual requirements in the prompt
> (subjects, their attributes, actions, setting, composition, style). For each, look at the image and answer yes,
> partly or no. Ignore artist names and trigger words. Reply as JSON: {"items": [{"requirement": "...", "answer":
> "yes|partly|no"}], "score": 0-100, "missing": "one short sentence"}

Good enough to build on, with two changes below: one checklist per prompt (fair comparison), and a score computed
from the answers instead of the model's own number.

## Method

**Step 1 - one checklist per prompt (text model, cached).** Every cell with the same prompt is judged on the same list,
so "LoRA A 7/8, LoRA B 4/8" is a real comparison.

> You turn an image-generation prompt into a checklist for judging a picture made from it. Write 4 to 8 requirements a
> person could verify by looking at the picture: the main subjects, their key attributes (colour, material, clothing),
> actions or pose, the setting, composition or viewpoint, lighting, and the medium (oil painting, watercolour,
> photograph). One fact per item, in plain words. Leave out: artist names, LoRA trigger words, quality words
> (masterpiece, 8k, detailed), moods that cannot be seen, and anything the prompt does not ask for. Mark each item
> "core" (the picture fails without it) or "detail". Reply as JSON: {"items": [{"requirement": "...", "kind":
> "subject|attribute|action|setting|composition|lighting|colour|medium|text|count", "weight": "core|detail"}]}

The prompt is first stripped of the cell's trigger words (the same stripping as the prompt library), and the checklist
is keyed by a hash of that text, so the same prompt in any sweep reuses it.

**Step 2 - per image (vision model).**

> Look at the image and check each requirement. Answer only from what is visible. "yes" = clearly there, "partly" =
> there but wrong in some way (colour, number, position) or hard to make out, "no" = not there. For every partly or no,
> add a few words on what you see instead. Reply as JSON: {"answers": [{"n": 1, "answer": "yes|partly|no", "note":
> "..."}]}

Temperature 0, the image at 1024 px on its long side, JSON schema enforced where the server supports it (the catalog's
`chat_json` already does this).

**Score** = 100 x (sum of points) / (sum of weights), where yes = 1, partly = 0.5, no = 0, and core items weigh 2,
details 1. **Missing** = the core items answered no, then partly, with the model's notes.

Baseline cells (no swept LoRA) are checked too: they show how hard the prompt is for the checkpoint, so a LoRA can be
judged against its own baseline ("baseline 75, this LoRA 50").

## Flaw detectors (added 2026-10-30)

The same per-image call also answers a fixed set of flaw questions that map to your rating keys. The prompt is taken
into account, so text or nudity the prompt asks for is not a flaw (step 1 records whether the prompt asks for text,
lettering, nudity or a frame).

| Question to the model | Flaw tag (key) | Expected reliability (small vision model) |
|---|---|---|
| Is there a signature, artist mark, logo or watermark? Where? | signature or watermark (S) | good when visible at 1024 px; tiny corner marks can be missed - see corner crops |
| Is there any text or lettering? What does it say? Did the prompt ask for it? | unwanted text or lettering (T) | good; reading the words lets you spot leaked trigger words |
| Is there nudity or sexual content? Did the prompt ask for it? | unwanted NSFW (N) | good |
| Is there a frame, border or matte around the picture? | added frame or border (R) | good |
| Are faces or hands visibly broken (extra / fused fingers, melted features)? | faces or hands broken (H) | fair at best - small models often miss these |
| Is the anatomy visibly wrong (extra limbs, impossible joints)? | anatomy wrong (A) | weak - treat as a hint only |

**Corner crops for signatures (optional):** signatures usually sit in a corner and are small. With
`prompt_check_corners` on, the four corners are also sent at full resolution (about 1-2 s more per image) and only the
signature / text questions look at them.

**Leaked trigger words:** when the model reads text in the image, it is compared with the cell's trigger words; a match
is reported as "trigger word drawn as text: 'tamalemp'" - direct evidence for the comma-vs-period trigger_join test.

**How it shows:** the matching flaw buttons in the zoom get a **suggestion outline** (dashed, with the model's note on
hover); one key press confirms it as a tag, as now. Nothing is ever set for you. The tile shows small flaw initials
(e.g. "S T") next to the prompt score; you can filter by them ("cells the model thinks have a signature").

**Checking the checker** covers these too: agreement with your own S / T / N / R / H / A tags on an already-rated
sweep, per flaw, so weak detectors can be switched off (`prompt_check_flaws`: the list of flaws to ask about).
## Storage

```
prompt_checklists  key (hash of the stripped prompt), prompt, model, items (JSON), created_at
prompt_checks      sweep_id, cell_file, checklist_key, model, answers (JSON), score, missing, seconds, created_at
```

A check is redone only if its checklist or the vision model changes.

## Running it

- **Sweeps page:** a **Check prompts** button runs the check for the open sweep in the background, cell by cell, with
  progress in the header and a Stop button; cells already checked are skipped. About 25 minutes for 300 cells.
- **Command line:** `run.py sweep-check <sweep id or folder>` (and `--all-unchecked`).
- **Automatic (off by default):** a setting `prompt_check_auto` checks a new sweep when it first appears on the Sweeps
  page.
- **GPU:** LM Studio and ComfyUI may share a card; run it after the sweep has finished. One request at a time.

## Showing it

- **Tile:** a small "prompt 75" badge, coloured (green 80+, amber 50-79, red under 50); sort and filter by it.
- **Zoom:** inside the prompt & settings strip (key Q): the checklist with yes / partly / no marks and the model's
  notes, plus the baseline's score for the same prompt.
- **Workbook:** columns "prompt check" and "missing".
- **Catalog:** stored next to the marks as a separate signal; it does not change the rating model unless you decide so
  later.

## Checking the checker (before trusting it)

1. Run it on a sweep you have already rated (stars, and ideally some follows / ignores the prompt tags).
2. Compare: rank correlation between the check score and your stars; how well the score separates the cells you tagged
   "ignores the prompt" from "follows the prompt"; and the 20 biggest disagreements, looked at side by side.
3. Run the same 30 cells twice: answers that flip between runs show where the model is unreliable.
4. If the 8B model is weak, try a larger vision model from LM Studio on the same 30 cells and compare time vs agreement.

Known weak spots of small vision models, so expect disagreements there: counting beyond 3, left / right, text in the
image, and a lean towards "yes".

## Settings

`prompt_check_model` (blank: `card_model`), `prompt_check_side` (1024), `prompt_check_auto` (false).

## Open decisions

1. Which vision model: start with the card model (qwen3-vl-8b), or try a larger one first?
2. Automatic after every sweep, or only by button (proposed: button first, automatic once it has proven itself)?
3. Core items weigh 2, details 1 - fine?
4. Should a low score ever *suggest* the "ignores the prompt" tag (shown as a hint, never set), or stay a number only?
5. Flaw detectors: which to ask about at first (proposed: S, T, N, R; H and A only after the agreement test), and corner crops on or off?
