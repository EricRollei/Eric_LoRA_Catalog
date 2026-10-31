# Prompt library wiring (2026-10-29): settings, run.py prompts, refresh step, community hook + exact NSFW number.
import os
import shutil
import sys
import py_compile

C = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_LoRA_Catalog"
L = os.path.join(C, "lora_catalog")
TAG = "_pre_promptlib_2026-10-29"
P = {}

P[os.path.join(L, "config.py")] = [
    ('''    slider_model: str = ""                 # text model for run.py sliders / ladders; blank: genre_model''',
     '''    slider_model: str = ""                 # text model for run.py sliders / ladders; blank: genre_model
    # Prompt library (run.py prompts, prompt_library.py, 2026-10-29): folder for _prompt_library.sqlite and exports
    # (blank: a "prompts" folder beside the folder holding the catalog, e.g. L:/Models/prompts), the text model for
    # export --rewrite (blank: genre_model, then card_model), and whether run.py refresh / community rebuild it.
    prompt_library_dir: str = ""
    prompt_model: str = ""
    prompt_library_auto: bool = True''', 1),
]

P[os.path.join(L, "__main__.py")] = [
    ('''    ident = sub.add_parser("identify",''',
     '''    prm = sub.add_parser("prompts", help="the prompt library: build (merge new prompts in), stats, export with "
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
                     "medium=watercolour or period=\\"art nouveau\\" (repeatable)")
    prm.add_argument("--min-words", type=int, default=0)
    prm.add_argument("--max-words", type=int, default=0)
    prm.add_argument("--source", default="", choices=["", "preview", "community"])
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
    ident = sub.add_parser("identify",''', 1),
    ('''    elif args.command == "identify":''',
     '''    elif args.command == "prompts":                     # 2026-10-29, prompt_library.py
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
                                 source=args.source, sample=args.sample, seed=args.seed)
                if not rows:
                    print("[prompts] no prompt matches those filters")
                else:
                    if args.rewrite:
                        PL.rewrite(settings, rows, args.rewrite, args.system_prompt, args.yes, args.expand_under)
                    PL.export(settings, rows, args.out, args.format, args.with_triggers, args.split_by)
    elif args.command == "identify":''', 1),
    ('''            print(f"[community] done: {community.collect(settings, only=args.only, limit=args.limit, force=args.force, refresh_days=args.refresh_days)}")''',
     '''            print(f"[community] done: {community.collect(settings, only=args.only, limit=args.limit, force=args.force, refresh_days=args.refresh_days)}")
            if getattr(settings, "prompt_library_auto", True):      # new community prompts -> the prompt library
                try:
                    from .prompt_library import build as _prompts_build
                    _prompts_build(settings)
                except Exception as error:                         # never fails the community run
                    print(f"[prompts] library not updated: {type(error).__name__}: {error}")''', 1),
]

P[os.path.join(L, "refresh.py")] = [
    ('''    11b marks       your sweep marks: prompt fit for older marks, the model the cards and Suggester read (optional)''',
     '''    11b marks       your sweep marks: prompt fit for older marks, the model the cards and Suggester read (optional)
    11c prompts     the prompt library: merge new preview / community prompts (optional; settings prompt_library_auto)''', 1),
    ('''OPTIONAL = {"style", "pairs", "triggers", "marks"}''', '''OPTIONAL = {"style", "pairs", "triggers", "marks", "prompts"}''', 1),
    ('''              "triggers", "review-all", "facets", "marks", "style"]''',
     '''              "triggers", "review-all", "facets", "marks", "prompts", "style"]''', 1),
    ('''        if name == "triggers":
            args += ["--refresh"]''', '''        if name == "prompts":
            args = ["prompts", "build", "--auto"]   # 2026-10-29: skipped when prompt_library_auto is off
        if name == "triggers":
            args += ["--refresh"]''', 1),
]

P[os.path.join(L, "community.py")] = [
    ('''    have = {r[1] for r in connection.execute("pragma table_info(community_stats)")}
    for column, kind in NEW_COLUMNS.items():
        if column not in have:
            connection.execute(f"alter table community_stats add column {column} {kind}")''',
     '''    have = {r[1] for r in connection.execute("pragma table_info(community_stats)")}
    for column, kind in NEW_COLUMNS.items():
        if column not in have:
            connection.execute(f"alter table community_stats add column {column} {kind}")
    # 2026-10-29: Civitai's exact level number (browsingLevel 1/2/4/8/16 = PG/PG-13/R/X/XXX): the text level "X"
    # covers both X and XXX. Filled as prompts are collected; the prompt library prefers it.
    if "nsfw_num" not in {r[1] for r in connection.execute("pragma table_info(community_prompts)")}:
        connection.execute("alter table community_prompts add column nsfw_num INTEGER")''', 1),
    ('''            connection.execute("""insert or replace into community_prompts (image_id, version_id, source, prompt, cleaned,
                negative, weight, base_model, prompt_style, nsfw_level, reactions, picked) values (?,?,?,?,?,?,?,?,?,?,?,0)""",
                (int(item["id"]), version_id, source, prompt[:4000], cleaned, str(meta.get("negativePrompt") or "")[:1500],
                 _weight_for(item, version_id, names, local), item.get("baseModel") or "", prompt_style(prompt),
                 str(item.get("nsfwLevel") or "None"), _reactions(item.get("stats") or {})))''',
     '''            level_num = item.get("browsingLevel") if isinstance(item.get("browsingLevel"), int) else None
            connection.execute("""insert or replace into community_prompts (image_id, version_id, source, prompt, cleaned,
                negative, weight, base_model, prompt_style, nsfw_level, reactions, picked, nsfw_num)
                values (?,?,?,?,?,?,?,?,?,?,?,0,?)""",
                (int(item["id"]), version_id, source, prompt[:4000], cleaned, str(meta.get("negativePrompt") or "")[:1500],
                 _weight_for(item, version_id, names, local), item.get("baseModel") or "", prompt_style(prompt),
                 str(item.get("nsfwLevel") or "None"), _reactions(item.get("stats") or {}), level_num))''', 1),
    ('''        connection.execute("""insert or replace into community_prompts (image_id, version_id, source, prompt, cleaned,
            negative, weight, base_model, prompt_style, nsfw_level, reactions, picked) values (?,?,?,?,?,?,?,?,?,?,?,0)""",
            (-(n + 1), version_id, "creator", prompt[:4000], clean_prompt(prompt), str(example["negative"] or "")[:1500],
             example["weight"], row["base_model"] or "", prompt_style(prompt), level, None))''',
     '''        connection.execute("""insert or replace into community_prompts (image_id, version_id, source, prompt, cleaned,
            negative, weight, base_model, prompt_style, nsfw_level, reactions, picked, nsfw_num)
            values (?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (-(n + 1), version_id, "creator", prompt[:4000], clean_prompt(prompt), str(example["negative"] or "")[:1500],
             example["weight"], row["base_model"] or "", prompt_style(prompt), level, None, example["nsfw_level"]))''', 1),
]

P[os.path.join(L, "prompt_library.py")] = [
    ('''                   "nsfw": level_index(r["nsfw_level"]), "reactions": r["reactions"], "image_id": r["image_id"],''',
     '''                   "nsfw": level_index(r["nsfw_num"] if "nsfw_num" in r.keys() and r["nsfw_num"] is not None
                                       else r["nsfw_level"]),
                   "reactions": r["reactions"], "image_id": r["image_id"],''', 1),
]


def main():
    staged, ok = {}, True
    for path, edits in P.items():
        text = open(path, encoding="utf-8", newline="").read()
        nl = "\r\n" if "\r\n" in text else "\n"
        for k, (old, new, want) in enumerate(edits):
            o, n = old.replace("\n", nl), new.replace("\n", nl)
            got = text.count(o)
            if got != want:
                print(f"MISMATCH {os.path.basename(path)} edit#{k}: found {got}, expected {want}: {old[:70]!r}")
                ok = False
                continue
            text = text.replace(o, n)
        staged[path] = text
    if not ok:
        print("ABORTED - nothing written")
        return 1
    for path, text in staged.items():
        base, ext = os.path.splitext(os.path.basename(path))
        bk = os.path.join(C, "docs", "backup", f"{base}{TAG}{ext}")
        if not os.path.exists(bk):
            shutil.copy2(path, bk)
        open(path, "w", encoding="utf-8", newline="").write(text)
        py_compile.compile(path, doraise=True)
        print("patched", os.path.basename(path))
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
