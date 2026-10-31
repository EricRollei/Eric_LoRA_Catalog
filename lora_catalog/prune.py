"""
Prune orphans (2026-10-16): rows about LoRA files that are no longer in the index.

When a LoRA file is deleted or moved by hand, `index` drops its `loras` row but the cards, slider cards,
style vectors and trigger rows made for it stay behind (the editor then counts more cards than LoRAs).

    python run.py prune-orphans            list what would go (nothing is changed)
    python run.py prune-orphans --apply    back up the catalog, then delete those rows

Kept on purpose, even for a missing file: your own edits (card_edits), the move log (path_history),
marked stacks (stacks) and Civitai harvest records (harvest_seeds) - history, or costly to fetch again.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime
import os
import sqlite3

# tables keyed by a LoRA file path whose rows mean nothing once the file is gone
PRUNE_TABLES = ["cards", "card_facets", "slider_cards", "style_vectors", "trigger_recipes", "trigger_tests",
                "trigger_picks", "examples", "file_hashes"]


def _tables(db) -> set:
    return {r[0] for r in db.execute("select name from sqlite_master where type='table'")}


def find(db) -> dict:
    """{table: [orphan paths]} for the prune tables that exist and have a path column."""
    have = _tables(db)
    out = {}
    for t in PRUNE_TABLES:
        if t not in have:
            continue
        cols = {r[1] for r in db.execute(f"pragma table_info('{t}')")}
        if "path" not in cols:
            continue
        rows = [r[0] for r in db.execute(
            f"select distinct path from '{t}' where path not in (select path from loras)")]
        if rows:
            out[t] = rows
    return out


def run(settings, apply: bool = False) -> int:
    path = settings.catalog_path
    db = sqlite3.connect(path)
    try:
        orphans = find(db)
        if not orphans:
            print("[prune] no orphan rows: every card belongs to an indexed LoRA")
            return 0
        files = sorted({p for rows in orphans.values() for p in rows})
        print(f"[prune] {len(files)} LoRA file(s) no longer indexed:")
        for p in files:
            print(f"   {p}")
        for t, rows in orphans.items():
            print(f"   {t}: {len(rows)} row(s)")
        if not apply:
            print("[prune] dry run: nothing changed (add --apply to delete these rows; the catalog is backed up first)")
            return 0
        stamp = datetime.date.today().isoformat()
        backup = os.path.splitext(path)[0] + f".pre_prune_{stamp}.sqlite"
        n = 1
        while os.path.exists(backup):
            n += 1
            backup = os.path.splitext(path)[0] + f".pre_prune_{stamp}_{n}.sqlite"
        dst = sqlite3.connect(backup)
        db.backup(dst)
        dst.close()
        print(f"[prune] backup: {backup}")
        with db:
            for t in orphans:
                cur = db.execute(f"delete from '{t}' where path not in (select path from loras)")
                print(f"[prune] {t}: {cur.rowcount} row(s) deleted")
        return 0
    finally:
        db.close()
