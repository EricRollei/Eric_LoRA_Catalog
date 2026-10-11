"""
Who made each LoRA: the Civitai username per model id, kept in its own table so a re-index never wipes it.

    creators   model_id INTEGER PRIMARY KEY, username TEXT, source TEXT (cache|mirror|api|none), fetched_at TEXT

    python run.py creators [--only Krea2/graphic-art] [--no-api]

Order of lookup per model: our Civitai cache (m{id}.json) > LoRA Manager's offline mirror (models.username) > the
Civitai API (/models/{id}, cached like the crawl). Self-trained LoRAs have no model id; readers show "self-trained".

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import Dict, Optional

from .config import Settings
from .crawl import CACHE_DIR, CivitaiClient

SCHEMA = """
CREATE TABLE IF NOT EXISTS creators (
    model_id INTEGER PRIMARY KEY, username TEXT, source TEXT, fetched_at TEXT
);
"""


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)


def from_cache(model_id: int) -> Optional[str]:
    path = os.path.join(CACHE_DIR, f"m{model_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if data.get("_not_found"):
        return ""
    return (data.get("creator") or {}).get("username") or ""


def lookup(connection: sqlite3.Connection, model_id: Optional[int]) -> str:
    if not model_id:
        return ""
    row = connection.execute("select username from creators where model_id = ?", (model_id,)).fetchone()
    return (row[0] or "") if row else ""


def fill(settings: Settings, only: str = "", use_api: bool = True) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    ensure(connection)
    where = "civitai_model_id is not null"
    params: tuple = ()
    if only:
        where += " and replace(rel_dir, '\\', '/') like ?"
        params = (only.replace("\\", "/") + "%",)
    wanted = [row[0] for row in connection.execute(
        f"select distinct civitai_model_id from loras where {where} and civitai_model_id > 0"
        " and civitai_model_id not in (select model_id from creators where username != '')", params)]
    mirror = None
    if settings.lm_civitai_db and os.path.exists(settings.lm_civitai_db):
        mirror = sqlite3.connect(f"file:{settings.lm_civitai_db}?mode=ro", uri=True, timeout=10)
    client = CivitaiClient(settings) if use_api else None
    counts = {"models": len(wanted), "cache": 0, "mirror": 0, "api": 0, "none": 0}
    print(f"[creators] {len(wanted)} models without a creator; API {'on' if client else 'off'}", flush=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for number, model_id in enumerate(wanted, 1):
        name, source = from_cache(model_id), "cache"
        if not name and mirror is not None:
            found = mirror.execute("select username from models where id = ?", (model_id,)).fetchone()
            name, source = ((found[0] or "") if found else ""), "mirror"
        if not name and client is not None:
            model = client.get("m", model_id) or {}
            name, source = (model.get("creator") or {}).get("username") or "", "api"
        if not name:
            source = "none"
        counts[source] += 1
        connection.execute("insert or replace into creators (model_id, username, source, fetched_at) values (?,?,?,?)",
                           (model_id, name, source, now))
        if number % 50 == 0:
            connection.commit()
            print(f"[creators] {number}/{len(wanted)}  {counts}", flush=True)
    connection.commit()
    connection.close()
    if mirror is not None:
        mirror.close()
    return counts
