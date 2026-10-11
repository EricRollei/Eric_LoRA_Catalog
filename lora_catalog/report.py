"""
Coverage report: what the catalog knows, per architecture shelf, and what is still unidentified.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

from .store import Catalog


def coverage(catalog: Catalog, only: str = "") -> str:
    where = "where rel_dir like ?" if only else ""
    params = (only.replace("\\", "/") + "%",) if only else ()
    rows = catalog.query(f"""
        select arch_folder,
               count(*) as files,
               sum(case when sha256 != '' and sha256 is not null then 1 else 0 end) as with_sha,
               sum(case when civitai_source is not null and civitai_source != '' then 1 else 0 end) as civitai_record,
               sum(case when trained_words is not null and trained_words not in ('', '[]') then 1 else 0 end) as with_triggers,
               sum(case when ss_tags is not null and ss_tags not in ('', '[]') then 1 else 0 end) as with_ss_tags,
               sum(case when example_count > 0 then 1 else 0 end) as with_examples,
               sum(case when description is not null and description != '' then 1 else 0 end) as with_description,
               sum(case when source_kind = 'self_trained' then 1 else 0 end) as self_trained,
               sum(case when (civitai_source is null or civitai_source = '') and civitai_version_id is not null then 1 else 0 end) as crawlable,
               sum(case when (civitai_source is null or civitai_source = '') and civitai_version_id is null
                         and (trained_words is null or trained_words in ('', '[]')) and source_kind != 'self_trained' then 1 else 0 end) as unidentified
        from loras {where}
        group by arch_folder order by files desc""", params)
    header = f"{'shelf':<22}{'files':>7}{'sha':>7}{'civitai':>9}{'trigger':>9}{'ss_tags':>9}{'gallery':>9}{'descr':>7}{'own':>6}{'crawl':>7}{'unknown':>9}"
    lines = [header, "-" * len(header)]
    totals = [0] * 11
    for row in rows:
        values = [row["files"], row["with_sha"], row["civitai_record"], row["with_triggers"], row["with_ss_tags"],
                  row["with_examples"], row["with_description"], row["self_trained"], row["crawlable"], row["unidentified"]]
        totals = [total + value for total, value in zip(totals, values + [0])]
        lines.append(f"{(row['arch_folder'] or '(root)')[:21]:<22}{values[0]:>7}{values[1]:>7}{values[2]:>9}{values[3]:>9}"
                     f"{values[4]:>9}{values[5]:>9}{values[6]:>7}{values[7]:>6}{values[8]:>7}{values[9]:>9}")
    lines.append("-" * len(header))
    lines.append(f"{'total':<22}{totals[0]:>7}{totals[1]:>7}{totals[2]:>9}{totals[3]:>9}{totals[4]:>9}{totals[5]:>9}"
                 f"{totals[6]:>7}{totals[7]:>6}{totals[8]:>7}{totals[9]:>9}")
    lines += ["",
              "sha      sha256 known (LoRA Manager, downloader or embedded) - no hashing needed",
              "civitai  a Civitai record found locally (LoRA Manager's own block, or its offline mirror)",
              "trigger  trained words from Civitai or LoRA Manager",
              "ss_tags  dataset tags in the file header (a trigger hint for kohya / ai-toolkit LoRAs)",
              "gallery  example prompts from the downloader's gallery sidecars",
              "own      matched to an ai-toolkit training job (trigger, arch and captions known)",
              "crawl    no local Civitai record but a version id is known: phase 2 fetches it from the API",
              "unknown  no Civitai id, no trigger, not a training job: the refiner works from the header and previews only"]
    signatures = catalog.query("""select arch_folder, key_signature, count(*) as n from loras where key_signature != ''
                                  group by arch_folder, key_signature order by arch_folder, n desc""")
    lines += ["", "tensor-key signatures per shelf (the most common two; a shelf with several = mixed architectures):"]
    seen = {}
    for row in signatures:
        shown = seen.get(row["arch_folder"], 0)
        if shown < 2:
            lines.append(f"  {(row['arch_folder'] or '(root)'):<14} {row['n']:>5}  {row['key_signature'][:110]}")
            seen[row["arch_folder"]] = shown + 1
    return "\n".join(lines)
