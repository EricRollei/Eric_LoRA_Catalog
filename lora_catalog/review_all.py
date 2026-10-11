"""
Review pages for the whole library: one page per shelf/category folder (Krea2/art-style, Flux/character ...), so no
single page carries thousands of thumbnails, plus reviews/index.html linking them with their card counts.

    python run.py review-all

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import html
import os
from collections import Counter

from .config import DATA_DIR, Settings
from .review import build_review
from .store import Catalog

REVIEW_DIR = os.path.join(DATA_DIR, "reviews")


def _group(rel_dir: str) -> str:
    """The first two folders below the library root ("Krea2/art-style"), or the shelf alone when it has no category."""
    parts = (rel_dir or "").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] or "(root)")


def write_all_reviews(settings: Settings) -> str:
    catalog = Catalog(settings.catalog_path)
    rows = catalog.query("select l.rel_dir from cards c join loras l on l.path = c.path")
    catalog.close()
    groups = Counter(_group(row["rel_dir"]) for row in rows)
    os.makedirs(REVIEW_DIR, exist_ok=True)
    links = []
    for group, count in sorted(groups.items()):
        name = "".join(ch if ch.isalnum() else "_" for ch in group).strip("_") + ".html"
        with open(os.path.join(REVIEW_DIR, name), "w", encoding="utf-8") as handle:
            handle.write(build_review(settings, group if group != "(root)" else ""))
        links.append((group, count, name))
        print(f"[review] {group}: {count} cards", flush=True)
    shelves = Counter(group.split("/")[0] for group in groups.elements())
    items = []
    for shelf in sorted(shelves):
        sub = "".join(f'<li><a href="{name}">{html.escape(group)}</a> <small>{count}</small></li>'
                      for group, count, name in links if group.split("/")[0] == shelf)
        items.append(f"<section><h2>{html.escape(shelf)} <small>{shelves[shelf]}</small></h2><ul>{sub}</ul></section>")
    index = f"""<!doctype html><html><head><meta charset="utf-8"><title>LoRA cards</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#faf9f6; --fg:#222; --muted:#777; --line:#e3e0d8; --accent:#3a5a8c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17181a; --fg:#e6e4df; --muted:#9a978f; --line:#34363a; --accent:#8fb0e0; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.5 system-ui, sans-serif; margin:0; padding:16px; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:16px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin:0 0 4px; border-bottom:1px solid var(--line); }}
small {{ color:var(--muted); }} ul {{ margin:0; padding-left:16px; }} a {{ color:var(--accent); }}
</style></head><body><h1>LoRA cards <small>{sum(groups.values())} LoRAs</small></h1><main>{''.join(items)}</main></body></html>"""
    path = os.path.join(REVIEW_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(index)
    return path
