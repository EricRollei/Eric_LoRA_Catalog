# How the catalog gets cards and picks right

A LoRA card is only as good as its evidence, and the LoRA Suggester is only as good as the cards. This page sums up
every layer that feeds them - what is built, what is being tested, and what is planned - so it is clear what each
part contributes and where a wrong pick can come from. Status as of 14 Oct 2026.

## 1. Evidence for the card (built)

- **Local facts, cheapest first:** LoRA Manager data, the Civitai downloader's sidecars, the safetensors header
  (kohya dataset tags, network dim), ai-toolkit job folders for self-trained LoRAs (trigger, captions).
- **Civitai records** by sha256: name, base model, tags, trained words, the model page description.
- **Example images** saved next to each LoRA; the vision model sees two of them when it writes the card.
- **Civitai audience data:** the weights people actually use (the card's weights come from 10+ community weights
  when there are enough), the prompts they write, content levels, how the LoRA ranks against others of its base model.

## 2. Writing the card (built)

- A local **vision model** (e.g. Qwen3-VL 8B) writes the card: role, kind, effect, facets, triggers, weights, NSFW.
- A larger **text model** adds **genre and mood** from the text evidence alone (`run.py genre-mood`), limited to the
  vocabulary, at most two each, with a warning when one term covers too many LoRAs.
- **Your edits always win.** Corrections, trigger states (priority / normal / off), weights and 1-5 star ratings are
  kept in `card_edits` by sha256, so they follow a file when it moves. The editor's queue shows uncertain cards first.

## 3. Checking a card against what the LoRA really does (built, being extended)

Words on a card can be wrong; renders are not. With the **CSD style model** (see the user guide) the catalog
compares what a LoRA actually renders with its own preview images:

- **Render check** (`run.py style --renders`): one verdict per LoRA - `shows`, `needs_words`, `weak`, `unclear`.
  The Advanced Suggester gives `needs_words` LoRAs their card's style words.
- **Second pass** (`run.py style --second-pass`): re-renders the `unclear` ones with their creator's prompt.
- **Trigger recipes** (`run.py triggers`, Oct 2026): a recipe per LoRA from the trained words, the page's prompting
  advice, the creator's and the community's prompts; a trigger test measures which variant each LoRA needs. First
  finding: the checkpoint decides - fine-tunes need the LoRA's descriptive words, the base model needs far less.
  [TRIGGERS.md](TRIGGERS.md). The refresh writes recipes for new LoRAs (step 9b, shelves in `trigger_arches`).

## 4. Choosing LoRAs for a prompt (built)

The Suggester scores each card against the prompt by role (style, transform, treatment, detailer, lighting,
environment, subject), with facet and text weights per role, then applies:

- **your rating** (1 star x0.5 ... 5 stars x1.15); unrated cards get a small **Civitai popularity** nudge (±3%),
  never more than a star;
- **genre and mood** read from the whole prompt;
- **style families:** a second style is only added when it belongs to the same family as the first, and it runs as
  a blend at a reduced weight;
- **clash rules** between roles and side effects, and the **NSFW gate**;
- **a reference image** (Advanced node, `style_image`): its CSD look counts next to the words.

## 5. Learning which LoRAs work together (built, data growing)

- **Civitai stacks:** `run.py harvest` + `run.py pairs` count which LoRAs people combine in their top images
  (and a wishlist of LoRAs you don't own that pair well with yours).
- **Your own sweeps:** star ratings, flaw / praise tags and strength pushes on the Sweeps page store each cell's
  exact stack in `stacks`; `run.py stacks --pairs` lists good and bad pairs, and the marks model (`markmodel.py`,
  2026-10-17) separates each LoRA's quality from the prompt fit and the checkpoint for its card and the Suggester.
- **Style blends by look:** a calibration sweep marked G / B / U is the basis for a blend rule (`sweep-plan
  --by-look`) - analysis pending.

## 6. Next (agreed, not built yet)

- **Tested trigger pick:** a "use this trigger" choice in the editor and on the Sweeps page, pre-selected from the
  trigger test, stored with the card; optionally per checkpoint for people who use only a few. The nodes then use, in
  order: the pick for this checkpoint, the card's tested pick, trigger + phrase, the first trigger.
- **`recipe` trigger mode** on the LoRA / Multi-LoRA Stack nodes (next to `catalog`, which stays one short trigger).
- **Stack prompt merging:** one shared, shortest prompt text for a whole stack - duplicate pieces merged, pieces
  shared by several LoRAs kept first, LoRAs whose test shows the trigger is enough contribute only the trigger, a
  word budget by weight, and a note when two LoRAs' cards contradict (medium, palette).
- **To do - explore text embeddings** (`nomic-embed-text`, see below): a small test first - embed the pilot
  recipes' phrases and a sample of cards, check whether near-synonyms land close and unrelated phrases don't, then
  decide whether to use it for stack merging, card search or the Suggester.

## 7. Ideas under consideration

- **Text embeddings** (e.g. `nomic-embed-text`, which LM Studio and Ollama serve through the same OpenAI-style API
  as the chat models): a small text model that turns a phrase or a whole card into a vector, so that "dramatic
  chiaroscuro lighting" and "deep shadows, strong contrast" land close together even with no word in common. It
  would merge near-synonym phrases when stacks are merged, and could make card search and the Suggester match by
  meaning instead of by vocabulary. Not used yet (exploring it is on the to-do list above).
- **Effect vectors:** the CSD difference between a render with and without the LoRA, to describe what a LoRA
  *changes* rather than what its previews look like.
- **Wildcards by look or genre** ("Art Nouveau") that pick a fitting LoRA from the catalog straight into the
  Multi-LoRA Stack or the Suggester, and catalog-assisted LoRA picking inside the stack node.
