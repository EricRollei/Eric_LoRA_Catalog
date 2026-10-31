"""
Which words a LoRA's prompt should get - your pick, per card and optionally per checkpoint (2026-10-15).

A trigger recipe (triggers.py) offers named variants: none / trigger / official / trigger+phrase / full. A trigger
test (`run.py triggers --check <sweep>`) scores them and suggests one, for the checkpoint that sweep ran on. Here
those become "picks":

    trigger_picks   path, checkpoint ('' = every checkpoint), source ('you' | 'test'), variant, text, sweep, picked_at
                    PRIMARY KEY (path, checkpoint, source)

    you    set in the card editor ("Use this") or on the Sweeps page; always wins
    test   written by --check for that sweep's checkpoint (--sync-picks fills them from older checks)

resolve() turns them into three card_facets columns that Eric_Krea2 and the Suggester read:

    prompt_text           the words for this LoRA on any checkpoint
    trigger_status        picked | tested | recipe | trigger | ''   (where prompt_text came from)
    prompt_by_checkpoint  JSON {checkpoint: {"text", "source"}} - a checkpoint-specific choice wins over prompt_text

Order for one checkpoint: your pick for it, the test's pick for it, your pick for every checkpoint, your default word
option (word_options.py, 2026-10-20; status 'option'), the newest test pick, the recipe's trigger+phrase, the first
active trigger. A picked or tested variant with EMPTY text is a real
answer ("this LoRA needs no words") and is kept as such.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

TABLE = """CREATE TABLE IF NOT EXISTS trigger_picks (
    path TEXT, checkpoint TEXT, source TEXT, variant TEXT, text TEXT, sweep TEXT, picked_at TEXT,
    PRIMARY KEY (path, checkpoint, source))"""
SOURCES = ("you", "test")


def ensure(connection: sqlite3.Connection) -> None:
    connection.execute(TABLE)


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def set_pick(connection: sqlite3.Connection, path: str, checkpoint: str, variant: str, text: str,
             source: str = "you", sweep: str = "") -> None:
    ensure(connection)
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    connection.execute("insert or replace into trigger_picks values (?,?,?,?,?,?,?)",
                       (path, (checkpoint or "").strip(), source, variant or "custom", text or "", sweep or "",
                        time.strftime("%Y-%m-%d %H:%M:%S")))


def clear_pick(connection: sqlite3.Connection, path: str, checkpoint: str = "", source: str = "you") -> int:
    ensure(connection)
    cur = connection.execute("delete from trigger_picks where path = ? and checkpoint = ? and source = ?",
                             (path, (checkpoint or "").strip(), source))
    return cur.rowcount


def picks(connection: sqlite3.Connection, path: str) -> List[Dict[str, Any]]:
    ensure(connection)
    rows = connection.execute("select path, checkpoint, source, variant, text, sweep, picked_at from trigger_picks "
                              "where path = ? order by picked_at desc", (path,)).fetchall()
    keys = ["path", "checkpoint", "source", "variant", "text", "sweep", "picked_at"]
    return [dict(zip(keys, tuple(r))) for r in rows]


def variants(connection: sqlite3.Connection, path: str) -> List[Dict[str, Any]]:
    if not _has_table(connection, "trigger_recipes"):
        return []
    row = connection.execute("select variants from trigger_recipes where path = ?", (path,)).fetchone()
    try:
        out = json.loads(row[0]) if row and row[0] else []
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def resolve(connection: sqlite3.Connection, path: str, active_triggers: List[str],
            default_option: Optional[str] = None) -> Dict[str, Any]:
    """{prompt_text, trigger_status, prompt_by_checkpoint (JSON text or None)} for one LoRA. default_option = the
    text of your default word option, if you set one."""
    rows = picks(connection, path)
    by_ck: Dict[str, Dict[str, str]] = {}
    for r in sorted(rows, key=lambda r: (r["source"] == "you", r["picked_at"])):    # tests first, yours overwrite
        if r["checkpoint"]:
            by_ck[r["checkpoint"]] = {"text": r["text"], "source": r["source"]}
    mine_all = next((r for r in rows if r["source"] == "you" and not r["checkpoint"]), None)
    newest_test = next((r for r in rows if r["source"] == "test"), None)        # rows are newest first
    if mine_all is not None:
        text, status = mine_all["text"], "picked"
    elif default_option is not None:
        text, status = default_option, "option"
    elif newest_test is not None:
        text, status = newest_test["text"], "tested"
    else:
        from .triggers import default_text
        text = default_text(variants(connection, path))
        if text:
            status = "recipe"
        elif active_triggers:
            text, status = active_triggers[0], "trigger"
        else:
            text, status = "", ""
    return {"prompt_text": text, "trigger_status": status,
            "prompt_by_checkpoint": json.dumps(by_ck, ensure_ascii=False) if by_ck else None}


def checkpoint_of(sweep_id: str, sweeps_dir: str = "") -> str:
    """The checkpoint a sweep ran on: from its manifest when the folder is found, else from the folder name
    (<YYYYmmdd_HHMMSS>_<checkpoint> for sweeps before 2026-10-15)."""
    if sweeps_dir:
        from .sweepfiles import manifest_path
        mp = manifest_path(os.path.join(sweeps_dir, sweep_id))
        if mp:
            try:
                with open(mp, "r", encoding="utf-8") as handle:
                    ck = json.load(handle).get("checkpoint") or ""
                if ck:
                    return ck
            except (OSError, ValueError):
                pass
    return sweep_id[16:] if len(sweep_id) > 16 and sweep_id[8] == "_" else sweep_id


def sync_from_tests(connection: sqlite3.Connection, sweeps_dir: str = "", sweep: str = "") -> int:
    """Turn the suggested rows of trigger_tests into 'test' picks (per checkpoint); the newest check wins."""
    ensure(connection)
    if not _has_table(connection, "trigger_tests"):
        return 0
    sql = "select path, sweep, variant, text, checked_at from trigger_tests where suggested = 1"
    args: tuple = ()
    if sweep:
        sql += " and sweep = ?"
        args = (sweep,)
    rows = sorted(connection.execute(sql, args).fetchall(), key=lambda r: r[4] or "")
    for path, sw, variant, text, checked in rows:
        ck = checkpoint_of(sw, sweeps_dir)
        connection.execute("insert or replace into trigger_picks values (?,?,?,?,?,?,?)",
                           (path, ck, "test", variant, text or "", sw, checked or time.strftime("%Y-%m-%d %H:%M:%S")))
    return len(rows)


def checkpoints(connection: sqlite3.Connection, sweeps_dir: str = "") -> List[str]:
    """Checkpoints any pick or trigger test mentions - the editor's per-checkpoint choices."""
    ensure(connection)
    found = {r[0] for r in connection.execute("select distinct checkpoint from trigger_picks where checkpoint != ''")}
    if _has_table(connection, "trigger_tests"):
        found |= {checkpoint_of(r[0], sweeps_dir) for r in connection.execute("select distinct sweep from trigger_tests")}
    return sorted(c for c in found if c)


def card_view(connection: sqlite3.Connection, path: str, sweeps_dir: str = "") -> Dict[str, Any]:
    """Everything the editor's 'Prompt words' panel shows for one LoRA."""
    tests: List[Dict[str, Any]] = []
    if _has_table(connection, "trigger_tests"):
        for r in connection.execute("select sweep, variant, text, gain, words_gain, suggested, checked_at "
                                    "from trigger_tests where path = ? order by checked_at desc", (path,)):
            tests.append({"sweep": r[0], "checkpoint": checkpoint_of(r[0], sweeps_dir), "variant": r[1], "text": r[2],
                          "gain": r[3], "words_gain": r[4], "suggested": bool(r[5])})
    try:
        row = connection.execute("select prompt_text, trigger_status, prompt_by_checkpoint from card_facets "
                                 "where path = ?", (path,)).fetchone()
    except sqlite3.OperationalError:          # card_facets not built yet / older columns
        row = None
    current = None
    if row is not None:
        try:
            current = {"text": row[0] or "", "status": row[1] or "",
                       "by_checkpoint": json.loads(row[2]) if row[2] else {}}
        except (TypeError, ValueError, IndexError):
            current = None
    try:                                       # the recipe fields and your corrections (recipe_edits.py, 2026-10-16)
        from . import recipe_edits
        recipe = recipe_edits.view(connection, path)
    except sqlite3.Error:
        recipe = None
    try:                                       # named word options (word_options.py, 2026-10-20)
        from . import word_options
        options = word_options.view(connection, path)
    except sqlite3.Error:
        options = None
    return {"variants": variants(connection, path), "tests": tests, "picks": picks(connection, path),
            "checkpoints": checkpoints(connection, sweeps_dir), "current": current, "recipe": recipe,
            "word_options": options}
