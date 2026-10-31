"""
Command line.

    python -m lora_catalog index                      index every library root (unchanged files are skipped)
    python -m lora_catalog index --only Krea2/graphic_art
    python -m lora_catalog index --force              re-read everything
    python -m lora_catalog report [--only Krea2]      coverage per architecture shelf

    python run.py refresh                             every step after adding LoRAs (index ... facets)
    python run.py check                               settings, folders and the LLM server, in plain words

Run it through run.py with Python 3.12 (ComfyUI's embedded Python works:
    ComfyUI_windows_portable\\python_embeded\\python.exe custom_nodes\\Eric_LoRA_Catalog\\run.py index).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import DATA_DIR, SETTINGS_FILE, load_settings
from .indexer import index
from .jobs import job
from .report import coverage
from .store import Catalog


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lora_catalog")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("index", help="walk the libraries and write the catalog")
    run.add_argument("--only", default="", help="one folder below a library root, e.g. Krea2/graphic_art")
    run.add_argument("--force", action="store_true", help="re-read files whose size and date have not changed")
    run.add_argument("--limit", type=int, default=0, help="stop after this many files (testing)")
    show = sub.add_parser("report", help="coverage per architecture shelf")
    show.add_argument("--only", default="")
    fetch = sub.add_parser("crawl", help="fetch Civitai records for LoRAs with a version id but no local record")
    fetch.add_argument("--only", default="")
    fetch.add_argument("--limit", type=int, default=0)
    cards = sub.add_parser("cards", help="write each LoRA's card with the LLM (phase 3)")
    cards.add_argument("--only", default="")
    cards.add_argument("--limit", type=int, default=0)
    cards.add_argument("--force", action="store_true", help="rewrite cards that exist")
    cards.add_argument("--model", default="", help="override settings.card_model")
    cards.add_argument("--workers", type=int, default=1, help="LLM requests in flight at once (LM Studio parallel slots)")
    cards.add_argument("--paths-file", default="", help="only the LoRAs listed in this file (one path per line)")
    cards.add_argument("--show-evidence", action="store_true", help="print what the model would be given; call nothing")
    page = sub.add_parser("review", help="write an HTML review page of the cards in one folder")
    page.add_argument("--only", required=True)
    sub.add_parser("review-all", help="write review pages for every shelf/category folder, plus reviews/index.html")
    editor = sub.add_parser("serve", help="the card editor in your browser (http://127.0.0.1:8765)")
    editor.add_argument("--port", type=int, default=8765)
    editor.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    editor.add_argument("--parent-pid", type=int, default=0, help="stop when this process ends (the ComfyUI panel "
                                                                    "passes ComfyUI's own pid)")
    sub.add_parser("vocab-gaps", help="words typed in the editor that are not in vocab.yaml yet")
    pics = sub.add_parser("images", help="download Civitai example images next to LoRAs that have none")
    pics.add_argument("--only", default="")
    pics.add_argument("--max", type=int, default=12, help="images per LoRA (least NSFW first)")
    pics.add_argument("--fewer-than", type=int, default=1, help="only LoRAs with fewer images than this (1 = none)")
    pics.add_argument("--videos", action="store_true", help="also preview videos")
    pics.add_argument("--dry-run", action="store_true", help="list what would be saved, write nothing")
    pics.add_argument("--limit", type=int, default=0, help="stop after this many LoRAs (testing)")
    sub.add_parser("facets", help="rebuild card_facets, the pick-list view of every card that other tools read")
    slide = sub.add_parser("sliders", help="text-only slider cards: axis, ends, signed range (axes.yaml)")
    slide.add_argument("--only", default="")
    slide.add_argument("--force", action="store_true", help="rewrite slider cards that exist")
    slide.add_argument("--limit", type=int, default=0)
    slide.add_argument("--workers", type=int, default=1)
    slide.add_argument("--model", default="", help="LM Studio model id (default settings.slider_model)")
    slide.add_argument("--gaps", action="store_true", help="list axes the cards use that axes.yaml lacks")
    ladder = sub.add_parser("ladders", help="draft graded prompt-cue ladders per slider axis (axes_ladders_draft.yaml)")
    ladder.add_argument("--draft", action="store_true", help="draft every axis without a ladder (or --axis)")
    ladder.add_argument("--axis", default="", help="draft (or redraft) just this axis")
    ladder.add_argument("--accept", action="store_true", help="copy the draft file to axes_ladders.yaml (live)")
    ladder.add_argument("--tidy", action="store_true", help="re-apply the everyday-word guard to the draft file")
    ladder.add_argument("--model", default="", help="LM Studio model id (default settings.slider_model)")
    reap = sub.add_parser("harvest", help="fetch top community images per LoRA from Civitai (metadata only) for stacks")
    reap.add_argument("--only", default="")
    reap.add_argument("--limit", type=int, default=0)
    reap.add_argument("--force", action="store_true", help="re-fetch LoRAs already harvested")
    pair = sub.add_parser("pairs", help="count LoRA pairs from the harvest; --show <name>, --wishlist")
    pair.add_argument("--show", default="", help="a LoRA's best partners (part of its name)")
    pair.add_argument("--wishlist", action="store_true", help="LoRAs you do not own that pair well with yours, "
                                                                "ranked by pair score x Civitai score")
    pair.add_argument("--base", default="", help="wishlist: only this base model (part of its name, e.g. 'Krea')")
    pair.add_argument("--top", type=int, default=40, help="wishlist length (default 40)")
    pair.add_argument("--plain", action="store_true", help="wishlist by pair scores only (no Civitai calls)")
    pair.add_argument("--no-taste", action="store_true", help="wishlist: ignore your ratings of your own LoRAs "
                      "(the ranking before 2026-10-29)")
    pair.add_argument("--min", type=int, default=2, help="least images a pair must share (default 2)")
    marks = sub.add_parser("stacks", help="your marks on sweep cells (Sweeps page), by stack or --pairs")
    marks.add_argument("--pairs", action="store_true", help="fold the marks into LoRA pairs")
    marks.add_argument("--update", action="store_true", help="prompt fit / type for older marks, then refit the "
                                                             "marks model that the cards and the Suggester read")
    comm = sub.add_parser("community", help="Civitai audience data: score, real weights, NSFW check, test prompts")
    comm.add_argument("--only", default="", help="one shelf / folder, e.g. Krea2")
    comm.add_argument("--limit", type=int, default=0)
    comm.add_argument("--force", action="store_true", help="fetch again even if fetched in the last --refresh-days")
    comm.add_argument("--refresh-days", type=int, default=30)
    comm.add_argument("--score", action="store_true", help="recompute the scores only (no network)")
    comm.add_argument("--show", default=None, help="print what was collected for LoRAs whose name contains this")
    prm = sub.add_parser("prompts", help="the prompt library: build (merge new prompts in), stats, export with "
                                         "filters and formats, optional LLM rewrite (prompt_library.py)")
    prm.add_argument("action", choices=["build", "stats", "export"])
    prm.add_argument("--auto", action="store_true", help="build only when settings prompt_library_auto is on (refresh)")
    prm.add_argument("--base", default="", help="base model, part of the name ('Krea' = Krea 2 and Krea 2 Turbo)")
    prm.add_argument("--max-nsfw", default="X", help="PG | PG-13 | R | X | XXX: worst image level seen (default X)")
    prm.add_argument("--clean-text", action="store_true", help="also drop prompts whose words mention nudity")
    prm.add_argument("--min-reactions", type=int, default=0, help="community likes on its best image")
    prm.add_argument("--lora-rating", type=float, default=0, help="came with a LoRA you rated this many stars or more")
    prm.add_argument("--style", default="", choices=["", "natural", "tags"])
    prm.add_argument("--facet", action="append", default=[], help="k=v of the LoRA it came with, e.g. "
                     "medium=watercolour or period=\"art nouveau\"; a value alone (surreal) = any facet (repeatable)")
    prm.add_argument("--text", action="append", default=[], help="the prompt itself contains this word or phrase "
                     "(repeatable: all must appear)")
    prm.add_argument("--min-words", type=int, default=0)
    prm.add_argument("--max-words", type=int, default=0)
    prm.add_argument("--source", default="", choices=["", "preview", "community"])
    prm.add_argument("--top", type=int, default=0, help="the first N matches in --order")
    prm.add_argument("--order", default="likes", choices=["likes", "newest"],
                     help="likes = most community likes first (default); newest = most recently posted first")
    prm.add_argument("--sample", type=int, default=0, help="a random N of the matches")
    prm.add_argument("--seed", type=int, default=0)
    prm.add_argument("--with-triggers", action="store_true", help="the original wording, triggers kept")
    prm.add_argument("--split-by", default="", help="one file per value of this facet (e.g. medium); --out = a folder")
    prm.add_argument("--format", default="", help="txt | jsonl | csv | sqlite (default: from --out, else txt)")
    prm.add_argument("--out", default="", help="file (or folder with --split-by); default: the library folder")
    prm.add_argument("--rewrite", default="", choices=["", "tidy", "krea2", "expand"],
                     help="pass the prompts through the local LLM first (cached)")
    prm.add_argument("--expand-under", type=int, default=25, help="--rewrite expand: only prompts shorter than this")
    prm.add_argument("--system-prompt", default="", help="a file with your own system prompt for --rewrite")
    prm.add_argument("--yes", action="store_true", help="no question before a long rewrite run")
    ident = sub.add_parser("identify", help="find Civitai ids for files that came without one (lookup by sha256)")
    ident.add_argument("--limit", type=int, default=0)
    ident.add_argument("--retry", action="store_true", help="look up files that were not found last time too")
    planner = sub.add_parser("sweep-plan", help="write a LoRA stack sweep preset for the Krea2 SweepPlan node")
    planner.add_argument("--folders", nargs="+", default=[], help="one or two folders, e.g. Krea2/graphic-art")
    planner.add_argument("--single", default="", help="one LoRA (name or part of it): a characterization sheet of its "
                                                       "own test prompts x three weights + no-LoRA, instead of stacks")
    planner.add_argument("--mode", default="", help="within, across, anchor - comma list (default: within,across "
                                                     "for two folders, within for one, anchor with --anchor)")
    planner.add_argument("--pairs", type=int, default=9, help="how many pairs (shared between the modes; default 9)")
    planner.add_argument("--anchor", default="", help="part of one LoRA's name: pair it with the others")
    planner.add_argument("--roles", default="style", help="card roles allowed, comma list, or 'any' (default style)")
    planner.add_argument("--blend", type=float, default=0.6, help="second LoRA's weight x this (default 0.6)")
    planner.add_argument("--max-uses", type=int, default=2, help="most pairs any one LoRA appears in (default 2)")
    planner.add_argument("--seed", type=int, default=1, help="another seed = other pairs among equals")
    planner.add_argument("--retest", action="store_true", help="include pairs already marked on the Sweeps page")
    planner.add_argument("--nsfw", action="store_true", help="include cards rated nudity / explicit")
    planner.add_argument("--prompts", default="", help="your own prompts separated by |, or from-loras for the pairs' own test prompts "
                                                        "(run.py community first; default: three per kind)")
    planner.add_argument("--seeds", default="101,102", help="render seeds (default 101,102)")
    planner.add_argument("--multipliers", default="", help="optional strength multipliers, e.g. 0.75,1.0")
    planner.add_argument("--max-cells", type=int, default=200, help="refuse bigger sweeps (default 200)")
    planner.add_argument("--name", default="", help="preset name (default 'stacks <folders> <date time>')")
    planner.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    gm = sub.add_parser("genre-mood", help="genre and mood for each card (text only, settings.genre_model)")
    gm.add_argument("--only", default="")
    gm.add_argument("--limit", type=int, default=0)
    gm.add_argument("--force", action="store_true", help="redo cards that have them")
    gm.add_argument("--dry-run", action="store_true", help="print the answers, write nothing")
    gm.add_argument("--sample", action="store_true", help="random LoRAs instead of folder order (with --limit)")
    gm.add_argument("--model", default="", help="override settings.genre_model")
    gm.add_argument("--genre-like", default="", help="only cards whose current genre contains this (a check)")
    trig = sub.add_parser("triggers", help="trigger recipes from the page, example and community prompts "
                                             "(text only, settings.genre_model)")
    trig.add_argument("--names", default="", help="comma list: file or model name contains")
    trig.add_argument("--only", default="", help="folder under the arch folder")
    trig.add_argument("--pilot", action="store_true", help="the 5 pilot LoRAs + 20 'unclear' ones with most prompts")
    trig.add_argument("--limit", type=int, default=0)
    trig.add_argument("--force", action="store_true", help="redo LoRAs whose evidence did not change")
    trig.add_argument("--dry-run", action="store_true", help="print the recipes, write nothing")
    trig.add_argument("--show", action="store_true", help="print stored recipes, ask nothing")
    trig.add_argument("--model", default="", help="override settings.genre_model")
    trig.add_argument("--arch", default="Krea2", help="shelf (arch folder); a comma list, or '*' for every shelf")
    trig.add_argument("--refresh", action="store_true", help="the refresh step: shelves from settings trigger_arches")
    trig.add_argument("--plan-test", action="store_true", help="write a trigger-test SweepPlan preset (combos)")
    trig.add_argument("--untested", action="store_true", help="--plan-test: skip LoRAs with a tested trigger")
    trig.add_argument("--subject", default="", help="--plan-test: subject prompt (default: the overnight woman)")
    trig.add_argument("--megapixels", type=float, default=1.0, help="--plan-test: render size (default 1)")
    trig.add_argument("--name", default="", help="--plan-test: preset name")
    trig.add_argument("--check", default="", help="score a finished trigger-test sweep folder")
    trig.add_argument("--refill-phrases", action="store_true",
                      help="stored recipes without a style phrase take it from the card text (no LLM)")
    trig.add_argument("--sync-picks", action="store_true",
                      help="turn earlier --check suggestions into per-checkpoint 'test' picks and refresh those cards")
    hasher = sub.add_parser("hash", help="compute sha256 for LoRAs no other source gave a hash for")
    hasher.add_argument("--only", default="")
    hasher.add_argument("--limit", type=int, default=0)
    who = sub.add_parser("creators", help="look up who made each LoRA (cache > LoRA Manager mirror > Civitai API)")
    who.add_argument("--only", default="")
    who.add_argument("--no-api", action="store_true", help="cache and mirror only, no network")
    fresh = sub.add_parser("refresh", help="every step after adding LoRAs: index, hash, Civitai, cards, genre-mood, "
                                           "review pages, facets")
    fresh.add_argument("--workers", type=int, default=0, help="cards: LLM requests in flight at once (default: 1 on "
                                                              "Ollama, 3 on LM Studio and other servers)")
    fresh.add_argument("--model", default="", help="cards: another vision model than settings.card_model")
    fresh.add_argument("--skip-internet", action="store_true", help="leave out the Civitai steps")
    fresh.add_argument("--steps", default="", help="only these steps, comma list (e.g. cards,genre-mood,facets)")
    drop = sub.add_parser("unload", help="free an LLM's memory on LM Studio / Ollama")
    drop.add_argument("--model", default="", help="model id (default: settings.card_model and the text model)")
    sub.add_parser("check", help="settings, folders and the LLM server, in plain words")
    sty = sub.add_parser("style", help="style vectors (CSD) from preview images, for 'similar look' and image search "
                                       "(optional: settings.json style_enabled)")
    sty.add_argument("--limit", type=int, default=0, help="stop after this many LoRAs (testing)")
    sty.add_argument("--force", action="store_true", help="recompute LoRAs whose previews have not changed")
    sty.add_argument("--show", default=None, help="print the closest looks to LoRAs whose name contains this")
    sty.add_argument("--all-bases", action="store_true", help="--show: compare across base models too")
    sty.add_argument("--renders", default="", help="a finished sweep folder: does each LoRA show its preview look in "
                                                  "its renders? stores a verdict per LoRA (editor badge, Suggester)")
    sty.add_argument("--kind", default="auto", choices=["auto", "neutral", "creator"],
                     help="--renders: plain prompts (neutral) or each LoRA's creator prompt (creator); auto = creator "
                          "for combos sweeps")
    sty.add_argument("--second-pass", action="store_true", help="write a sweep preset: the 'unclear' LoRAs with "
                                                                "their creator's prompt, each beside a no-LoRA render")
    sty.add_argument("--arch", default="Krea2", help="--second-pass: only this shelf (default Krea2; '' = all)")
    sty.add_argument("--prompts", type=int, default=1, help="--second-pass: creator prompts per LoRA (default 1)")
    sty.add_argument("--nsfw", action="store_true", help="--second-pass: include NSFW cards and R/X prompts")
    sty.add_argument("--name", default="", help="--second-pass: preset name")
    sty.add_argument("--dry-run", action="store_true", help="--second-pass: print the plan, write nothing")
    setup = sub.add_parser("style-setup", help="download (2.4 GB) and convert the style model; or --from a file you have")
    setup.add_argument("--from", dest="from_file", default="", help="a CSD pytorch_model.bin already downloaded")
    side = sub.add_parser("export-sidecars", help="write Forge / A1111 sidecar .json files (trigger words, preferred "
                                                  "weight, notes) from the cards; dry run unless --write")
    side.add_argument("--only", default="")
    side.add_argument("--write", action="store_true", help="write the files (without it: list what would be written)")
    side.add_argument("--overwrite-ours", action="store_true", help="refresh sidecars this tool wrote before "
                                                                    "(a sidecar you wrote yourself is never touched)")
    prune = sub.add_parser("prune-orphans", help="remove cards and other rows for LoRA files no longer indexed "
                                                  "(dry run unless --apply; the catalog is backed up first)")
    prune.add_argument("--apply", action="store_true", help="delete the rows (without it: list them)")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.command in ("serve", "check", "report", "unload", "style-setup") or \
            (args.command == "style" and (args.show is not None or args.dry_run)):
        return _run(args, settings)
    with job(args.command):
        return _run(args, settings)


def _run(args, settings) -> int:
    if args.command == "refresh":
        from .refresh import run as refresh_run
        return refresh_run(settings, workers=args.workers, model=args.model, skip_internet=args.skip_internet,
                           only_steps=args.steps)
    if args.command == "unload":
        from .llm import unload
        for model in [args.model] if args.model else list(dict.fromkeys([settings.card_model, settings.genre_model])):
            print(f"[unload] {unload(settings, model)}")
        return 0
    if args.command == "check":
        from .llm import test
        print(f"settings file: {SETTINGS_FILE}" + ("" if os.path.isfile(SETTINGS_FILE) else "  (not written yet)"))
        print(f"data folder:   {DATA_DIR}")
        print(f"catalog:       {settings.catalog_path}")
        print(f"LoRA folders:  {', '.join(settings.library_roots) or '(none)'}")
        print(f"Civitai key:   {'set' if settings.civitai_key() else 'not set (public records only)'}")
        for problem in settings.problems():
            print(f"PROBLEM: {problem}")
        from .style import available as style_available
        print(f"Style search:  {style_available(settings) or 'ready'}")
        result = test(settings)
        for key in ("server", "text", "vision", "json", "memory"):
            if result.get(key):
                print(f"LLM {key}: {result[key]}")
        return 0 if result.get("ok") and not settings.problems() else 1
    if args.command == "style-setup":
        from .style import setup
        return setup(settings, from_file=args.from_file)
    if args.command == "style":
        from . import style
        if args.show is not None:
            print(style.show(settings, args.show, scope="all" if args.all_bases else "base") if args.show.strip()
                  else "give part of a LoRA name")
            return 0
        if args.renders:
            try:
                result = style.render_check(settings, args.renders, kind=args.kind)
            except ValueError as error:
                print(f"[style] {error}")
                return 1
            print(style.render_report(result))
            return 0
        if args.second_pass:
            try:
                result = style.second_pass(settings, arch=args.arch, limit=args.limit, prompts=args.prompts,
                                           nsfw=args.nsfw, name=args.name)
            except ValueError as error:
                print(f"[style] {error}")
                return 1
            print(f"{result['name']}: {len(result['loras'])} LoRAs, {result['cells']} cells")
            for title, strength, n in result["loras"]:
                print(f"   @{strength:g}  {title}" + (f"  ({n} prompts)" if n > 1 else ""))
            if result["missing"]:
                print(f"   left out, no creator prompt (or only NSFW ones; --nsfw takes them): {', '.join(result['missing'][:20])}"
                      + (f" ... {len(result['missing']) - 20} more" if len(result["missing"]) > 20 else ""))
            if args.dry_run:
                print("[style] dry run: nothing written")
                return 0
            from .sweepplan import write_preset
            backup = write_preset(settings, result)
            print(f"[style] preset written to {settings.krea2_sweep_presets}" + (f" (backup {backup})" if backup else ""))
            return 0
        print(f"[style] done: {style.compute(settings, limit=args.limit, force=args.force)}")
        return 0
    if args.command == "prune-orphans":
        from .prune import run as prune_run
        return prune_run(settings, apply=args.apply)
    if args.command == "export-sidecars":
        from .sidecars import export
        print(f"[sidecars] {export(settings, only=args.only, write=args.write, overwrite_ours=args.overwrite_ours)}")
        return 0
    if args.command == "index":
        counts = index(settings, only=args.only.replace("/", os.sep), force=args.force, limit=args.limit)
        print(f"[LoRA Catalog] done: {counts}")
        catalog = Catalog(settings.catalog_path)
        print(coverage(catalog, args.only))
        catalog.close()
    elif args.command == "report":
        catalog = Catalog(settings.catalog_path)
        print(coverage(catalog, args.only))
        catalog.close()
    elif args.command == "crawl":
        from .crawl import crawl
        print(f"[crawl] done: {crawl(settings, only=args.only, limit=args.limit)}")
    elif args.command == "cards":
        from .cards import write_cards
        if args.model:
            settings.card_model = args.model
        print(f"[cards] done: {write_cards(settings, only=args.only, limit=args.limit, force=args.force, workers=args.workers,
                                              paths_file=args.paths_file, show_evidence=args.show_evidence)}")
    elif args.command == "review":
        from .review import write_review
        print(f"[review] written: {write_review(settings, args.only)}")
    elif args.command == "review-all":
        from .review_all import write_all_reviews
        print(f"[review] index: {write_all_reviews(settings)}")
    elif args.command == "serve":
        from .server import serve
        serve(port=args.port, open_browser=not args.no_browser, parent_pid=args.parent_pid)
    elif args.command == "images":
        from .civitai_images import fetch_missing
        print(f"[images] done: {fetch_missing(settings, only=args.only, count=args.max, videos=args.videos, dry_run=args.dry_run, limit=args.limit, fewer_than=args.fewer_than)}")
    elif args.command == "facets":
        from .facets import build_all
        print(f"[facets] done: {build_all(settings)}")
    elif args.command == "sliders":
        from .sliders import gaps, write_slider_cards
        if args.gaps:
            print(f"[sliders] axes not in axes.yaml: {gaps(settings) or 'none'}")
        else:
            model = args.model or settings.slider_model
            print(f"[sliders] done: {write_slider_cards(settings, model, only=args.only, force=args.force, limit=args.limit, workers=args.workers)}")
    elif args.command == "ladders":
        from .ladders import accept, draft, tidy
        if args.accept:
            print(f"[ladders] {accept()}")
        elif args.tidy:
            print(f"[ladders] tidy: {tidy()}")
        else:
            print(f"[ladders] done: {draft(settings, args.model or settings.slider_model, only_axis=args.axis)}")
    elif args.command == "harvest":
        from .harvest import harvest
        print(f"[harvest] done: {harvest(settings, only=args.only, limit=args.limit, force=args.force)}")
    elif args.command == "pairs":
        from .harvest import count_pairs, show, wishlist
        if args.show:
            print(show(settings, args.show))
        elif args.wishlist:
            print(wishlist(settings, top=args.top, base=args.base, plain=args.plain, use_taste=not args.no_taste))
        else:
            print(f"[pairs] done: {count_pairs(settings, min_n=args.min)}")
    elif args.command == "community":
        from . import community
        if args.show is not None:                    # never falls through to a full fetch, even with --show ""
            print(community.show(settings, args.show) if args.show.strip() else "give part of a LoRA name")
        elif args.score:
            import sqlite3
            connection = sqlite3.connect(settings.catalog_path)
            community.ensure(connection)
            print(f"[community] scored {community.score_all(connection)}, newer versions {community.refresh_newer(connection)}")
            connection.commit()
            connection.close()
        else:
            print(f"[community] done: {community.collect(settings, only=args.only, limit=args.limit, force=args.force, refresh_days=args.refresh_days)}")
            if getattr(settings, "prompt_library_auto", True):      # new community prompts -> the prompt library
                try:
                    from .prompt_library import build as _prompts_build
                    _prompts_build(settings)
                except Exception as error:                         # never fails the community run
                    print(f"[prompts] library not updated: {type(error).__name__}: {error}")
    elif args.command == "prompts":                     # 2026-10-29, prompt_library.py
        from . import prompt_library as PL
        if args.action == "build":
            if args.auto and not getattr(settings, "prompt_library_auto", True):
                print("[prompts] settings prompt_library_auto is off - not rebuilt")
            else:
                PL.build(settings)
        else:
            if not os.path.exists(PL.library_path(settings)):
                PL.build(settings)
            if args.action == "stats":
                print(PL.stats(settings))
            else:
                rows = PL.select(settings, base=args.base, max_nsfw=args.max_nsfw, clean_text=args.clean_text,
                                 min_reactions=args.min_reactions, lora_rating=args.lora_rating, style=args.style,
                                 facets=args.facet, min_words=args.min_words, max_words=args.max_words,
                                 source=args.source, sample=args.sample, seed=args.seed, top=args.top,
                                 text=args.text, order=args.order)
                if not rows:
                    print("[prompts] no prompt matches those filters")
                else:
                    if args.rewrite:
                        PL.rewrite(settings, rows, args.rewrite, args.system_prompt, args.yes, args.expand_under)
                    PL.export(settings, rows, args.out, args.format, args.with_triggers, args.split_by)
    elif args.command == "identify":
        from .community import identify
        print(f"[identify] done: {identify(settings, limit=args.limit, retry_misses=args.retry)}")
    elif args.command == "sweep-plan":
        from .sweepplan import run as run_sweep_plan
        run_sweep_plan(settings, args)
    elif args.command == "stacks":
        from .stacks import Sweeps, report
        if args.update:
            sweeps = Sweeps(settings)
            sweeps.backfill()
            print(f"[stacks] model: {sweeps.update_model()}")
        else:
            report(settings, pairs=args.pairs)
    elif args.command == "genre-mood":
        from .genre_mood import run as genre_mood_run
        print(f"[genre-mood] done: {genre_mood_run(settings, only=args.only, limit=args.limit, force=args.force, dry_run=args.dry_run, model=args.model, sample=args.sample, genre_like=args.genre_like)}")
    elif args.command == "triggers":
        from . import triggers
        if args.refill_phrases:
            print(f"[triggers] phrases filled from card text: {triggers.refill_phrases(settings)}")
        elif args.sync_picks:
            triggers.sync_picks(settings)
        elif args.check:
            print(triggers.check_report(triggers.check_test(settings, args.check)))
        elif args.plan_test:
            result = triggers.plan_test(settings, names=args.names, pilot=args.pilot, untested=args.untested, limit=args.limit, subject=args.subject, megapixels=args.megapixels, name=args.name, arch=args.arch, write=not args.dry_run)
            if args.dry_run:
                for combo in result["combos"]:
                    print("   ", os.path.basename(str(combo.get("lora_file") or "(no LoRA)"))[:40], "|", combo.get("lora_strength", ""), "|", combo["prompt"][:100])
        elif args.show:
            print(triggers.show(settings, names=args.names, pilot=args.pilot, arch=args.arch))
        else:
            arch = args.arch
            if args.refresh:
                arch = ",".join(getattr(settings, "trigger_arches", None) or []) or "*"
            print(f"[triggers] done: {triggers.run(settings, names=args.names, only=args.only, pilot=args.pilot, limit=args.limit, force=args.force, dry_run=args.dry_run, model=args.model, arch=arch, new_only=bool(args.refresh))}")
    elif args.command == "hash":
        from .hashes import hash_missing
        print(f"[hash] done: {hash_missing(settings, only=args.only, limit=args.limit)}")
    elif args.command == "creators":
        from .creators import fill
        print(f"[creators] done: {fill(settings, only=args.only, use_api=not args.no_api)}")
    elif args.command == "vocab-gaps":
        import sqlite3
        from .edits import ensure, vocab_gaps
        from .vocab import load_vocab
        connection = sqlite3.connect(settings.catalog_path)
        ensure(connection)
        gaps = vocab_gaps(connection, load_vocab())
        for facet, words in gaps.items():
            print(f"{facet}: " + ", ".join(f"{w} ({n})" for w, n in sorted(words.items(), key=lambda x: -x[1])))
        print("no new words" if not gaps else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
