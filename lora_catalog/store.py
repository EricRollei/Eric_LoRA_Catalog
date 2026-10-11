"""
The catalog file: one sqlite database beside the library.

    loras      one row per LoRA file: identity, Civitai fields, header facts, training facts, your flags
    examples   gallery prompts that used the LoRA, with the weight they used it at
    cards      (phase 3) what the refiner concludes the LoRA does; created now so readers can rely on it

Rows are keyed by the absolute path. `sig` (size|mtime) lets a re-index skip files that have not changed.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS loras (
    path TEXT PRIMARY KEY,
    file_name TEXT, stem TEXT, size INTEGER, mtime REAL, sig TEXT,
    root TEXT, arch_folder TEXT, rel_dir TEXT,
    sha256 TEXT, sha_source TEXT,
    civitai_model_id INTEGER, civitai_version_id INTEGER, id_source TEXT,
    model_name TEXT, version_name TEXT, base_model TEXT, model_type TEXT,
    tags TEXT, trained_words TEXT, description TEXT, nsfw_level INTEGER,
    allow_commercial TEXT, civitai_source TEXT,
    header_meta TEXT, key_signature TEXT, network_dim INTEGER, network_alpha REAL, ss_tags TEXT, output_name TEXT,
    training_job TEXT, training_trigger TEXT, training_arch TEXT, caption_samples TEXT,
    lm_favorite INTEGER, lm_exclude INTEGER, lm_notes TEXT,
    source_kind TEXT,          -- civitai | self_trained | huggingface | unknown
    example_count INTEGER,
    problems TEXT,
    indexed_at TEXT
);
CREATE INDEX IF NOT EXISTS loras_arch ON loras(arch_folder);
CREATE INDEX IF NOT EXISTS loras_sha ON loras(sha256);
CREATE INDEX IF NOT EXISTS loras_version ON loras(civitai_version_id);

CREATE TABLE IF NOT EXISTS examples (
    path TEXT, media_file TEXT, prompt TEXT, negative TEXT, weight REAL, base_checkpoint TEXT, sampler TEXT,
    steps INTEGER, cfg REAL, nsfw_level INTEGER,
    PRIMARY KEY (path, media_file)
);

CREATE TABLE IF NOT EXISTS cards (
    path TEXT PRIMARY KEY,
    kind TEXT, scope TEXT, effect TEXT, triggers TEXT, weight_typical REAL, weight_low REAL, weight_high REAL,
    overrides TEXT, nsfw_level INTEGER, confidence REAL, evidence TEXT, refined_by TEXT, refined_at TEXT,
    embedding BLOB,
    artists TEXT, medium TEXT, movement_era TEXT, palette TEXT, line_quality TEXT, suited_subjects TEXT,
    genre TEXT, mood TEXT
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""

LORA_COLUMNS = [
    "path", "file_name", "stem", "size", "mtime", "sig", "root", "arch_folder", "rel_dir", "sha256", "sha_source",
    "civitai_model_id", "civitai_version_id", "id_source", "model_name", "version_name", "base_model", "model_type",
    "tags", "trained_words", "description", "nsfw_level", "allow_commercial", "civitai_source", "header_meta",
    "key_signature", "network_dim", "network_alpha", "ss_tags", "output_name", "training_job", "training_trigger",
    "training_arch", "caption_samples", "lm_favorite", "lm_exclude", "lm_notes", "source_kind", "example_count",
    "problems", "indexed_at",
]
EXAMPLE_COLUMNS = ["path", "media_file", "prompt", "negative", "weight", "base_checkpoint", "sampler", "steps", "cfg",
                   "nsfw_level"]


def _cell(value: Any) -> Any:
    """Lists and dicts are stored as JSON text."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


class Catalog:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=120)   # other tools (facets, editor) may hold a write briefly
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Columns added after a catalog file was first created (CREATE TABLE IF NOT EXISTS leaves old tables alone)."""
        added = {"cards": ["artists", "medium", "movement_era", "palette", "line_quality", "suited_subjects",
                           "genre", "mood"]}                    # genre, mood: 2026-10-10
        for table, columns in added.items():
            have = {row[1] for row in self.connection.execute(f"pragma table_info({table})")}
            for column in columns:
                if column not in have:
                    self.connection.execute(f"alter table {table} add column {column} TEXT")
        self.connection.commit()

    def signatures(self) -> Dict[str, str]:
        return {row["path"]: row["sig"] for row in self.connection.execute("select path, sig from loras")}

    def put_lora(self, row: Dict[str, Any]) -> None:
        values = [_cell(row.get(column)) for column in LORA_COLUMNS]
        placeholders = ",".join("?" for _ in LORA_COLUMNS)
        self.connection.execute(f"insert or replace into loras ({','.join(LORA_COLUMNS)}) values ({placeholders})", values)

    def put_examples(self, path: str, examples: Iterable[Dict[str, Any]]) -> None:
        self.connection.execute("delete from examples where path = ?", (path,))
        placeholders = ",".join("?" for _ in EXAMPLE_COLUMNS)
        for example in examples:
            example = dict(example, path=path)
            self.connection.execute(f"insert or replace into examples ({','.join(EXAMPLE_COLUMNS)}) values ({placeholders})",
                                    [_cell(example.get(column)) for column in EXAMPLE_COLUMNS])

    def forget_missing(self, live_paths: set, roots: List[str]) -> int:
        """Rows under an indexed root whose file is gone. Cards are kept (they may be re-linked by sha later)."""
        lowered_roots = [os.path.normcase(root) for root in roots]
        stale = [row["path"] for row in self.connection.execute("select path from loras")
                 if row["path"] not in live_paths and any(os.path.normcase(row["path"]).startswith(root) for root in lowered_roots)]
        for path in stale:
            self.connection.execute("delete from loras where path = ?", (path,))
            self.connection.execute("delete from examples where path = ?", (path,))
        return len(stale)

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute("insert or replace into meta (k, v) values (?, ?)", (key, value))

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def query(self, sql: str, parameters: Optional[tuple] = None) -> List[sqlite3.Row]:
        return list(self.connection.execute(sql, parameters or ()))
