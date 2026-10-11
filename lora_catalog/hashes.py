"""
Our own sha256 for LoRA files that no other source gave a hash for (2026-10-09).

The indexer takes the hash from LoRA Manager's sidecar, the Civitai downloader's file info or the file's embedded
modelspec hash - nothing is read from the file body, so indexing stays fast. A file with none of those gets no hash,
and then card edits fall back to being keyed by path (they break when the file moves, and cannot be shared with other
users, whose copy of the same LoRA is matched by hash). This fills that gap.

    python run.py hash                     every catalogued LoRA without a sha256
    python run.py hash --only Krea2        one shelf / folder
    python run.py hash --limit 20          a few, for testing

    file_hashes      path, sig (size|mtime), sha256, hashed_at - computed hashes, kept so a later `index --force` or a
                     re-index of an unchanged file reuses them instead of reading the file again. A hash whose sig no
                     longer matches the file is ignored (the file changed).

When a file gets its hash, a card edit that was keyed by its path is re-keyed to the hash, and its card_facets row is
refreshed, so nothing you edited is lost.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from typing import Dict, Optional, Tuple

from .config import Settings

TABLE = """
CREATE TABLE IF NOT EXISTS file_hashes (path TEXT PRIMARY KEY, sig TEXT, sha256 TEXT, hashed_at TEXT);
"""
CHUNK = 8 * 1024 * 1024


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(TABLE)


def load(connection: sqlite3.Connection) -> Dict[str, Tuple[str, str]]:
    """{path: (sig, sha256)} for the indexer."""
    ensure(connection)
    return {row[0]: (row[1], row[2]) for row in connection.execute("select path, sig, sha256 from file_hashes")}


def lookup(known: Dict[str, Tuple[str, str]], path: str, sig: str) -> Optional[str]:
    entry = known.get(path)
    return entry[1] if entry and entry[0] == sig else None


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _rekey_edit(connection: sqlite3.Connection, path: str, sha: str) -> bool:
    """A card edit stored under the path moves to the hash (unless one already exists under the hash)."""
    try:
        old = connection.execute("select 1 from card_edits where key = ?", (path,)).fetchone()
        if not old:
            return False
        taken = connection.execute("select 1 from card_edits where key = ?", (sha,)).fetchone()
        if taken:
            return False
        connection.execute("update card_edits set key = ? where key = ?", (sha, path))
        return True
    except sqlite3.OperationalError:       # no card_edits table yet
        return False


def hash_missing(settings: Settings, only: str = "", limit: int = 0) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=120)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    where, params = "(sha256 is null or sha256 = '')", []
    if only:
        where += " and rel_dir like ?"
        params.append(only.replace("\\", "/") + "%")
    rows = list(connection.execute(f"select path, sig, size from loras where {where} order by size", params))
    if limit:
        rows = rows[:limit]
    total_bytes = sum(int(row["size"] or 0) for row in rows)
    print(f"[hash] {len(rows)} files without a sha256 ({total_bytes / 1e9:.1f} GB)", flush=True)
    counts = {"hashed": 0, "missing": 0, "changed": 0, "edits_rekeyed": 0, "failed": 0}
    started, done_bytes = time.time(), 0
    touched = []
    for number, row in enumerate(rows, 1):
        path = row["path"]
        if not os.path.exists(path):
            counts["missing"] += 1
            continue
        stat = os.stat(path)
        sig = f"{stat.st_size}|{int(stat.st_mtime)}"
        if sig != row["sig"]:
            counts["changed"] += 1          # re-index first; hashing a file the catalog has not seen in this state
            continue
        try:
            sha = sha256_of(path)
        except OSError as error:
            counts["failed"] += 1
            print(f"[hash] failed {path}: {error}", flush=True)
            continue
        connection.execute("insert or replace into file_hashes (path, sig, sha256, hashed_at) values (?,?,?,?)",
                           (path, sig, sha, time.strftime("%Y-%m-%d %H:%M")))
        connection.execute("update loras set sha256 = ?, sha_source = 'computed' where path = ?", (sha, path))
        if _rekey_edit(connection, path, sha):
            counts["edits_rekeyed"] += 1
        counts["hashed"] += 1
        touched.append(path)
        done_bytes += stat.st_size
        if number % 10 == 0 or number == len(rows):
            connection.commit()
            rate = done_bytes / max(0.001, time.time() - started) / 1e6
            print(f"[hash] {number}/{len(rows)}  {rate:.0f} MB/s", flush=True)
    connection.commit()
    try:                                 # keep what the Suggester reads in step (key column, edit lookup)
        from .facets import refresh_one
        for path in touched:
            refresh_one(connection, path)
    except Exception as error:           # facets can always be rebuilt with `run.py facets`
        print(f"[hash] card_facets not refreshed ({error}); run `run.py facets`", flush=True)
    connection.close()
    counts["seconds"] = int(time.time() - started)
    return counts
