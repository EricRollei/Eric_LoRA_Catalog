# Test queue (started 2026-10-13)

Tick items off as they're done; Claude runs the analysis on each sweep folder you point it at.

## Now running

- [x] **Overnight - style check (portrait)**: done and graded (sweep 20261013_082330). Render check run: 73 LoRAs show their look, 270 unclear. It agrees with Eric's grades: AUC 0.90. Steps:
  - `run.py style --renders <folder>`, which gives each LoRA a verdict (shows its look / unclear);
  - `run.py style --second-pass`, which writes the creator-prompt preset for the unclear LoRAs (item 6).

## 1. LoRA checks first: is the LoRA mechanism sound?

If these find a fault in Ultra's LoRA path, the style-check results must be redone after the fix.

| # | What | Where | Checkpoint | Cells / time |
|---|------|-------|------------|--------------|
| 1a | (native side: 4 of 6 done; the 2 Chalk-with-LoRA cells ran out of VRAM, rerun after a ComfyUI restart) **Native vs Ultra**: Esoterica + 2 Liquid Chalk prompts, the creators' own settings, each with and without its LoRA | workflow `LoRA check - native vs Ultra.json`, then sweep preset `LoRA check - native vs Ultra` | stock `krea2_turbo_bf16` for both | 6 + 6, ~15 min |
| 1b | **Strength ladder**: Ballpoint, Esoterica, Chalk at 0 / 0.5 / 1 / 1.5, no triggers | preset `LoRA check - strength ladder (no triggers)` | run twice: loxsutopicworld v3.0, then stock turbo | 26 x 2, ~35 min each |
| 1c | (done 2026-10-14, sweep 20261013_211514: Ballpoint shows clearly only at lcm_hybrid2/12 steps; Liquid Chalk splotchy at lcm_hybrid2/12 steps (white speckles), clean but faint at euler; lcm_hybrid2/24 steps noisier everywhere) **Sampler**: the same 3 at 1.0, lcm_hybrid2+bong_tangent vs euler+linear, 12 and 24 steps | preset `LoRA check - sampler (no triggers)` | whichever 1b points to (default loxsutopicworld) | 16, ~25 min |
| 1d | (done, sweep 20261013_223042: sampler and size change nothing - Ballpoint strong in all 6, Parkes/Ukiyo-e only peeling-plaster walls, zy and Chalk ~nothing. Same panel + checkpoint gave strong looks with the creators' prompts in 'native vs Ultra', so the neutral prompt is the variable) **Sampler x size** (2026-10-14): Michael Parkes, Ukiyo-e, zy cinematic, Liquid Chalk (bad overnight, loaded cleanly) + Ballpoint (control) + no-LoRA; card weights, catalog triggers; lcm_hybrid2/bong_tangent vs euler/simple vs er_sde/simple at 1.0 and 2.4 MP | preset `LoRA check - sampler x size` | loxsutopicworld v3.0, overnight panel (12 steps, eta 0, white noise, cond_rebalance off) | 36, ~50 min |
| 1e | **Overnight rerun - 6 duplicate-name LoRAs** (2026-10-14): the 6 cells that failed overnight, now named by folder path | preset `Overnight rerun - 6 duplicate-name LoRAs` | overnight panel and seed, so the baseline matches | 7, ~10 min |
| 1f | **LoKr after the fix** (2026-10-14): Klimt and Beksinski should now show their look. Log line to look for: `LoKR: composed decomposed factors into full w1/w2` | any run with `3137507_klimt` and `3164272_zdzislaw_beksinski_krea_2` (bakeoff, card weight) | loxsutopicworld v3.0 | 2-4 cells |
| 1h | **Trigger test - pilot 5** (2026-10-14): Parkes, Ukiyo-e, zy cinematic, Liquid Chalk, Ballpoint - each recipe variant (none / trigger / official / trigger+phrase / full, duplicates merged) at the card weight, plus a words-only (no LoRA) cell each and one plain cell. Node triggers OFF: the words are in each cell's prompt. Score: `run.py triggers --check <folder>` | preset `Trigger test - pilot 5` | loxsutopicworld v3.0, overnight panel and seed; size 1 MP is in the cells | 26, ~30 min |
| 1i | **Trigger test - pilot 20**: the 20 'unclear' overnight LoRAs with the most community prompts, same layout | preset `Trigger test - pilot 20` | same as 1h | 89, ~1.7 h |
| 1g | **DoRA after the fix** (2026-10-14): `3153372_remove_clothing_censors` (the only DoRA on the shelf). Log line: `DoRA direct merge (weight=...): applied=160`. Expect only a modest change from before: on the real turbo weights its magnitudes sit within ~1% of the base column norms | any run with it | loxsutopicworld v3.0 | 1-2 cells |

How to read them:
- **1d:** a LoRA clean and in style at euler / 1 MP but splotchy at lcm_hybrid2 / 2.4 MP means the sampler or the size, not the loader. Splotchy everywhere while Ballpoint stays clean points back at the LoRA (or how it's applied).
- **1a:** native shows the look and Ultra doesn't: the fault is in our pipeline. Both show it: the earlier sweeps muted it. Neither shows it: the creator's own checkpoint matters.
- **1b:** strength 0 must equal no-LoRA, and the change must grow with strength. Weak LoRAs coming alive on stock turbo means the fine-tune mutes them.
- **1c:** weak LoRAs showing more under euler or more steps means the fast sampler mutes them.

## 2. Marking (no GPU)

- [x] **Calibration - style blends by look**: mark the 74 cells G / B / U on the card editor's Sweeps page (sweep 20261013_001804). This unlocks part 4: the blend rule and `sweep-plan --by-look`.

## 3. More sweeps

- [x] **Pilot B - lighting & detail** (sweep 20261013_174232). Analysis still to do.
- [x] **Pilot C - subjects** (sweep 20261013_185308). Analysis still to do. Claude runs the effect analysis on both.
- [ ] **Style check - second pass (creator prompts)**: written after the overnight analysis. Then run `run.py style --renders` on it, which turns unclear into needs-its-words / weak.

## 4. In ComfyUI (quick)

- [ ] **LoRA Suggester (Advanced)**: first real run in a graph: style_image (one image, then a batch), both image modes, weight 0 / 0.4 / 1.
- [ ] **Advanced Suggester with a Civitai download or one of your own PNGs** through Load Image: the report should list
  `LoRAs in <file>:`, and your style LoRA from it should be picked ("used in the style image").
- [ ] **Card editor**: restart it, then drop a Civitai download or one of your PNGs into the search box and check the
  "made with" line. Also check the render-verdict badge in "Similar look" (Ballpoint = shows its look, Chalk = needs its style words).

## Housekeeping

- [ ] `A:\Eric_LoRA_Catalog_staging` is still locked. Close any terminal or editor that has it open, then Claude retries the move to the Recycle Bin.
