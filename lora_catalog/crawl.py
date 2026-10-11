"""
Phase 2: fetch Civitai records for LoRAs whose version id is known but that have no local record (LoRA Manager's
mirror predates them: Krea 2, Qwen-Image 2.1, MiniMax H3 ...).

    GET /api/v1/model-versions/{versionId}   trained words, base model, version description, gallery
    GET /api/v1/models/{modelId}             model description, tags, licence

Every answer is cached as raw JSON under cache/civitai/ (v<id>.json, m<id>.json), so a re-run never re-downloads,
and a 404 is cached as a miss. The API key is read from the Civitai downloader's settings.json at run time and is
never copied anywhere.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from .config import DATA_DIR, Settings
from .local_sources import fix_mojibake, strip_html
from .store import Catalog

API = "https://civitai.com/api/v1"      # default only; the client uses settings.civitai_host (civitai.red since 2026-10-09)
CACHE_DIR = os.path.join(DATA_DIR, "cache", "civitai")
PAUSE_SECONDS = 0.6            # polite spacing between live requests


def _api_key(settings: Settings) -> str:
    """Environment, settings.json, then another tool's settings file (config.Settings.civitai_key). Never printed."""
    try:
        return settings.civitai_key()
    except AttributeError:
        return ""


class CivitaiClient:
    def __init__(self, settings: Settings):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "LoRA-Catalog/0.1 (local library index)"
        key = _api_key(settings)
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"
        self.has_key = bool(key)
        self.api = str(getattr(settings, "civitai_host", "") or "https://civitai.com").rstrip("/") + "/api/v1"
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.live_calls = 0
        self.last_status: Optional[int] = None    # HTTP status of the last get_json (None: no answer at all)

    def get_json(self, endpoint: str, params: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """An uncached API call (e.g. /images) with the same retries and politeness as get(). None on failure."""
        url = f"{self.api}/{endpoint.lstrip('/')}"
        self.last_status = None
        for attempt in range(3):
            time.sleep(PAUSE_SECONDS * (attempt + 1))
            try:
                response = self.session.get(url, params=params, timeout=timeout)
            except Exception as error:
                print(f"[civitai] {url}: {error}", flush=True)
                continue
            self.live_calls += 1
            self.last_status = response.status_code
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                time.sleep(20)
                continue
            print(f"[civitai] {url}: HTTP {response.status_code}", flush=True)
            if response.status_code in (400, 401, 403, 404):
                return None
        return None

    def get(self, kind: str, item_id: int) -> Optional[Dict[str, Any]]:
        """The record, from cache or the API. None = not found (cached) or a transient error (not cached)."""
        cache = os.path.join(CACHE_DIR, f"{kind}{item_id}.json")
        if os.path.isfile(cache):
            with open(cache, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return None if data.get("_not_found") else data
        url = f"{self.api}/{'model-versions' if kind == 'v' else 'models'}/{item_id}"
        for attempt in range(3):
            time.sleep(PAUSE_SECONDS * (attempt + 1))
            try:
                response = self.session.get(url, timeout=30)
            except Exception as error:      # network trouble: retry, then give up without caching
                print(f"[crawl] {url}: {error}", flush=True)
                continue
            self.live_calls += 1
            if response.status_code == 200:
                data = response.json()
                with open(cache, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False)
                return data
            if response.status_code == 404:
                with open(cache, "w", encoding="utf-8") as handle:
                    json.dump({"_not_found": True, "at": time.strftime("%Y-%m-%d")}, handle)
                return None
            if response.status_code == 429:
                time.sleep(20)
                continue
            print(f"[crawl] {url}: HTTP {response.status_code}", flush=True)
        return None


def record(version: Dict[str, Any], model: Dict[str, Any], settings: Settings, model_id: Optional[int] = None) -> Dict[str, Any]:
    """Catalog fields from a version record and its model record (lists stay lists)."""
    description = "\n\n".join(part for part in (model.get("description") or "", version.get("description") or "") if part)
    return {
        "civitai_model_id": model_id or version.get("modelId"),
        "civitai_version_id": version.get("id"),
        "model_name": fix_mojibake(str(model.get("name") or (version.get("model") or {}).get("name") or "")),
        "version_name": fix_mojibake(str(version.get("name") or "")),
        "model_type": model.get("type") or (version.get("model") or {}).get("type"),
        "base_model": version.get("baseModel"),
        "trained_words": [str(word).strip() for word in version.get("trainedWords") or [] if str(word).strip()],
        "tags": [str(tag) for tag in model.get("tags") or []],
        "description": strip_html(description, settings.max_description_chars),
        "nsfw_level": version.get("nsfwLevel"),
        "allow_commercial": json.dumps(model.get("allowCommercialUse")) if model.get("allowCommercialUse") is not None else None,
        "civitai_source": "api",
    }


def cached_record(version_id: Optional[int], settings: Settings) -> Dict[str, Any]:
    """A record the crawl fetched earlier, from its cache only (no network). {} when there is none. Lets a re-index
    keep what the crawl found."""
    if version_id is None:
        return {}
    path = os.path.join(CACHE_DIR, f"v{version_id}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            version = json.load(handle)
        if version.get("_not_found"):
            return {}
        model: Dict[str, Any] = {}
        model_path = os.path.join(CACHE_DIR, f"m{version.get('modelId')}.json")
        if os.path.isfile(model_path):
            with open(model_path, "r", encoding="utf-8") as handle:
                model = json.load(handle)
            if model.get("_not_found"):
                model = {}
        return record(version, model, settings)
    except (OSError, ValueError):
        return {}


def crawl(settings: Settings, only: str = "", limit: int = 0) -> Dict[str, int]:
    catalog = Catalog(settings.catalog_path)
    where = "(civitai_source is null or civitai_source = '') and civitai_version_id is not null"
    params: tuple = ()
    if only:
        where += " and rel_dir like ?"
        params = (only.replace("\\", "/") + "%",)
    rows = catalog.query(f"select path, civitai_version_id, civitai_model_id from loras where {where}", params)
    client = CivitaiClient(settings)
    print(f"[crawl] {len(rows)} LoRAs to fetch; API key {'found' if client.has_key else 'NOT found (anonymous)'}", flush=True)
    counts = {"fetched": 0, "not_found": 0}
    for number, row in enumerate(rows, 1):
        version = client.get("v", row["civitai_version_id"])
        if not version:
            counts["not_found"] += 1
            continue
        model_id = version.get("modelId") or row["civitai_model_id"]
        model = (client.get("m", model_id) if model_id else None) or {}
        update = record(version, model, settings, model_id)
        update = {key: (json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value)
                  for key, value in update.items()}
        assignments = ", ".join(f"{column} = ?" for column in update)
        catalog.connection.execute(f"update loras set {assignments} where path = ?", list(update.values()) + [row["path"]])
        counts["fetched"] += 1
        if number % 20 == 0:
            catalog.commit()
            print(f"[crawl] {number}/{len(rows)}  ({client.live_calls} live calls)", flush=True)
        if limit and number >= limit:
            break
    catalog.close()
    counts["live_calls"] = client.live_calls
    return counts
