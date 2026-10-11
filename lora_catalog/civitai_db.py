"""
LoRA Manager's offline Civitai database (comfyui-lora-manager/civitai/civitai.sqlite), read-only.

Measured 2026-09-30: 527k models, 685k versions, 717k files; a sha256 match for 57% of a 600-file sample of the
local library. Stale for architectures newer than its snapshot (Krea 2, Qwen, Z-Image, recent Wan): those come
from the live API in phase 2.

model_files has no index on sha256, so a per-file query would scan 717k rows each time. The sha256 -> ids map is
read once into memory instead (a few seconds), and model / version rows are then fetched by primary key.

The data column is Civitai's own JSON. Fields used: version.trainedWords, baseModel, name, nsfwLevel, description;
model.name, type, tags, description, allowCommercialUse.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Optional, Tuple


class CivitaiMirror:
    def __init__(self, path: str):
        self.available = bool(path) and os.path.isfile(path)
        self.by_sha: Dict[str, Tuple[int, int]] = {}
        self.connection: Optional[sqlite3.Connection] = None
        if not self.available:
            return
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for sha, model_id, version_id in self.connection.execute(
                "select sha256, model_id, version_id from model_files where sha256 is not null and sha256 != ''"):
            self.by_sha[sha.lower()] = (model_id, version_id)

    def _data(self, table: str, row_id: Optional[int]) -> Dict[str, Any]:
        if self.connection is None or row_id is None:
            return {}
        row = self.connection.execute(f"select data from {table} where id = ?", (row_id,)).fetchone()
        if not row:
            return {}
        try:
            data = json.loads(row[0])
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def lookup(self, sha256: str = "", version_id: Optional[int] = None) -> Dict[str, Any]:
        """Catalog fields for a file, by sha256 first, else by a version id from the folder layout. {} if unknown."""
        if not self.available:
            return {}
        model_id = None
        if sha256 and sha256.lower() in self.by_sha:
            model_id, version_id = self.by_sha[sha256.lower()]
            how = "sha256"
        elif version_id is not None:
            how = "version_id"
        else:
            return {}
        version = self._data("model_versions", version_id)
        if not version:
            return {}
        model_id = model_id or version.get("modelId")
        model = self._data("models", model_id)
        return {
            "civitai_model_id": model_id, "civitai_version_id": version_id,
            "model_name": model.get("name"), "version_name": version.get("name"), "model_type": model.get("type"),
            "base_model": version.get("baseModel"),
            "trained_words": [str(word).strip() for word in version.get("trainedWords") or [] if str(word).strip()],
            "tags": [str(tag) for tag in model.get("tags") or []],
            "description": "\n\n".join(part for part in (model.get("description") or "", version.get("description") or "") if part),
            "nsfw_level": version.get("nsfwLevel") if version.get("nsfwLevel") is not None else model.get("nsfwLevel"),
            "allow_commercial": json.dumps(model.get("allowCommercialUse")) if model.get("allowCommercialUse") is not None else None,
            "civitai_source": f"lm_offline:{how}",
        }

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
