# Trigger words: what to put in the prompt

A LoRA's Civitai "trigger" is often one made-up token. Many model pages say more in prose ("use this", "add
*streaky*", "works best with ...") that the API does not return, and the trained-words list is sometimes a long
phrase, several alternatives, or truncated. The catalog therefore writes a **prompt recipe** per LoRA and lets you
test which part of it the LoRA actually needs.

## The recipe (`run.py triggers`)

Evidence per LoRA:

- the full Civitai trained words and the card's triggers;
- the model page description (HTML stripped; lines about prompting, weights and settings listed first);
- the creator's example prompts, ranked by how much their image looks like the LoRA's style (CSD style vectors);
- phrases that recur in the prompts people used with the LoRA but are rare in prompts for the same base model.

The local text model (`genre_model` in settings) reads this and writes: the trigger, a style phrase of up to 40 words,
extra phrases the page asks for, setting hints (weight, sampler, CFG ...), words to avoid and the page's advice. A
trigger the model gives that appears nowhere in the evidence is dropped; a made-up token that sits in most of the
LoRA's prompts is used when no trained words are listed.

From the recipe come named prompt variants (duplicates merged):

| variant | text |
|---|---|
| `none` | nothing - the LoRA alone |
| `trigger` | the trigger token |
| `official` | all Civitai trained words (truncation repaired) |
| `trigger+phrase` | trigger + style phrase |
| `full` | official + phrase + extra phrases (max ~90 words) |

Untested LoRAs default to `trigger+phrase`. Recipes go to the `trigger_recipes` table; cards and your edits are not
touched.

```
python run.py triggers --names parkes,ukiyo     write recipes (only new or changed evidence unless --force)
python run.py triggers --show --names parkes     print stored recipes
python run.py triggers --plan-test --names ...   write a trigger-test SweepPlan preset (Eric_Krea2)
python run.py triggers --check <sweep folder>    score a finished trigger test (its pick is saved, see below)
python run.py triggers --sync-picks              older checks -> per-checkpoint picks, cards refreshed
```

## The trigger test

One cell per variant at the card's weight (node triggers off - the words are written into each cell's prompt), one
**words-only** cell per LoRA (the fullest words, no LoRA) and one **plain** cell (subject only, no LoRA). The check
embeds every image with CSD and scores **gain** = similarity to the LoRA's own preview look minus the plain cell's.
The suggested variant is the shortest within 90% of the best gain.

## Picking the words (2026-10-15)

Which variant a prompt actually gets is a **pick**, stored per LoRA and, if you like, per checkpoint
(`trigger_picks` table):

- **A trigger test** saves its suggestion as a *test* pick for the checkpoint that sweep ran on (`--check` does it;
  `--sync-picks` does it for checks you ran before).
- **You** pick in the card editor's **Prompt words** panel: the recipe variants with their text and test scores
  (★ = the test's suggestion), a **Use** button on each, your own words in a box below, and a scope - *all
  checkpoints* or one checkpoint your tests ran on. It saves at once (no Save needed). Or on the **Sweeps page**: a
  trigger-test cell has a **use words** button (that sweep's checkpoint; Shift-click: every checkpoint).

The card (`card_facets`) then carries `prompt_text`, `trigger_status` (picked / tested / recipe / trigger) and
`prompt_by_checkpoint`. Eric_Krea2's **`recipe`** trigger mode (Apply LoRA, Multi-LoRA Stack, Sweep Plan) uses them
for the loaded checkpoint, in this order: your pick for this checkpoint, the test's pick for it, your pick for every
checkpoint, the newest test pick, the recipe's trigger + style phrase, the first trigger. A pick of `none` means
"this LoRA needs no words here" and adds nothing. (Since 2026-10-16 the Krea2 nodes call this mode **saved pick**;
**full** uses the recipe's full variant and ignores picks; **trigger** uses the card's first active trigger.)

## Correcting a recipe (2026-10-16)

The text model sometimes gets a recipe wrong - e.g. Shouji Ink's trigger came out as `woman1 appearance age`, learned
from JSON-style community prompts. In the card editor's **Prompt words** panel, open **Recipe**: the trigger, style
phrase, official words and the Full line are editable (the model's text shows as the placeholder). **Save recipe
changes** rebuilds the variants from your fields at once; a field you leave alone keeps the model's value, and Full,
left alone, is rebuilt from the fields above. *back to the model's* undoes one field. Your fields are kept in the
`recipe_edits` table and applied on top whenever `run.py triggers` rewrites the recipe, so a re-run never loses
them. A LoRA with no recipe yet gets one made from your fields.

## Results, 14 Oct 2026: the checkpoint decides how much the LoRA needs its words

Five Krea 2 LoRAs, one seed, one neutral subject (a woman on a chair by a window), 1 MP, the same panel, on three
checkpoints: **Lox** (loxsutopicworld v3.0 fine-tune), **FinalCut** (krea2finalcut universal v3 fine-tune) and
**base turbo** (krea2_turbo_bf16). Gain is measured against each checkpoint's own plain cell. ★ = suggested variant.

| LoRA / variant | Lox | FinalCut | base turbo |
|---|---|---|---|
| **Michael Parkes** - no words | +0.00 | +0.03 | **+0.44** |
| trigger `mparkes` | +0.01 | +0.02 | +0.24 |
| official | ★ +0.55 | +0.52 | ★ +0.57 |
| trigger+phrase | +0.53 | ★ +0.54 | +0.57 |
| words only, no LoRA | +0.20 | +0.33 | +0.34 |
| **Liquid Chalk** - no words | +0.03 | +0.06 | **+0.60** |
| trigger `n0vuschr0ma style` | +0.02 | ★ +0.46 | +0.60 |
| trigger+phrase | ★ +0.43 | +0.42 | ★ +0.69 |
| words only | +0.11 | +0.02 | +0.30 |
| **Ukiyo-e** - no words | -0.02 | -0.03 | +0.03 |
| trigger `ukiyo-e` | +0.06 | +0.04 | +0.38 |
| official | ★ +0.39 | ★ +0.36 | ★ +0.49 |
| trigger+phrase | +0.40 | +0.21 | +0.41 |
| full | +0.43 | +0.24 | +0.44 |
| words only | +0.41 | +0.23 | +0.39 |
| **Ballpoint** - no words | +0.02 | +0.17 | +0.08 |
| trigger `n0vuschr0ma style` | ★ +0.80 | ★ +0.82 | ★ +0.70 |
| trigger+phrase | +0.82 | +0.81 | +0.72 |
| words only | +0.69 | +0.75 | +0.56 |
| **zy cinematic** - no words | +0.00 | +0.04 | +0.08 |
| trigger `zy_cinematic` | +0.01 | +0.20 | +0.14 |
| trigger+phrase | ★ +0.18 | ★ +0.33 | ★ +0.35 |
| words only | +0.20 | +0.38 | +0.27 |

What it shows:

- **On the base turbo model the LoRAs need far fewer words.** Two of the five show strongly with no words at all
  (Parkes +0.44, Liquid Chalk +0.60 on the plain prompt), and the bare trigger token works much better than on the
  fine-tunes (Ukiyo-e +0.38 vs +0.06 on Lox). The full words still add something, but little.
- **On the fine-tunes they mostly do not.** On Lox every LoRA except Ballpoint is near zero with no words *and with
  the bare trigger token*. What switches them on is the official trained words or trigger + style phrase - the
  descriptive words, not the token. FinalCut sits in between (Chalk and zy answer to the trigger alone; Parkes and
  Ukiyo-e do not).
- **The recipe variants work on all three checkpoints** (+0.4 to +0.7 for every LoRA except zy on Lox), so a recipe
  is a safe default whatever the checkpoint.
- **Sometimes the words do it without the LoRA.** On Lox, Ukiyo-e and zy cinematic reach almost the same score from
  words alone: the fine-tune (or Krea 2 itself) already knows that look. Whether a fine-tune has a style merged in, or
  simply responds to the words more strongly, this test cannot tell apart.
- **The same trigger can do different things.** Liquid Chalk and Ballpoint (same creator) share `n0vuschr0ma style`:
  on Lox it switches Ballpoint fully on and does nothing for Liquid Chalk, which needs the page's words ("liquid chalk
  marker on black chalkboard, streaky").

What it means in practice:

- **Stacks on the base model can be lean**: a trigger token per LoRA (sometimes nothing), which leaves the
  507-token prompt budget for the subject and for more LoRAs.
- **Stacks on a fine-tune need each LoRA's words**: budget roughly 20-40 words per style LoRA, or test which ones
  can go.
- **A trigger is tested per checkpoint family.** One pick per card that holds on your strictest checkpoint is the
  practical choice for a large library; with one or two checkpoints and a few dozen LoRAs, testing per checkpoint is
  cheap and gives the shortest prompts.

Limits of this run: five LoRAs, one seed, one subject, each at its card weight; CSD measures closeness to the
LoRA's preview images, not quality. The 20-LoRA pilot ("Trigger test - pilot 20") repeats it at a larger scale.
