"""
Example images from Civitai for LoRAs that have none on disk, saved next to the LoRA the way the Civitai downloader
(Eric's Civitai downloader, _download_showcase_images) does it:

    <LoRA folder>\\images\\<versionId>_showcase_<imageId>.<ext>      + <same>.json  (the image record: nsfwLevel ...)

In a folder that holds several LoRAs the name starts with the LoRA's own file stem instead of the version id, so the
editor (cards.preview_images) knows the image belongs to that LoRA:

    <folder>\\images\\<stem>_showcase_<imageId>.<ext>

    python run.py images                     every LoRA with a Civitai version id and no example image
    python run.py images --fewer-than 4      also LoRAs with 1-3 images (fills the editor's 2x2 grid)
    python run.py images --only Krea2/graphic-art --max 12 --dry-run
    fetch_for(path, settings)                one LoRA (the editor's "Get from Civitai" button)

Image list: the version record (cache/civitai/v<id>.json, fetched and cached when missing). Least NSFW first, like the
editor's own order. Images Civitai flags as showing a minor are never downloaded. Videos are skipped unless --videos.
Files are written as .part and renamed when complete; an existing file is never overwritten.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .cards import MODEL_EXTENSIONS, preview_images
from .config import Settings
from .crawl import CivitaiClient

DEFAULT_MAX = 12
PAUSE_SECONDS = 0.3


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "image"


def _shared_folder(folder: str) -> bool:
    """Same test as cards.preview_images: several LoRAs in a folder that is not a downloader version folder."""
    try:
        count = sum(1 for name in os.listdir(folder) if name.lower().endswith(MODEL_EXTENSIONS))
    except OSError:
        return False
    return count > 1 and not re.match(r"^\d+-", os.path.basename(folder))


def version_images(client: CivitaiClient, version_id: int, model_id: Optional[int]) -> List[Dict[str, Any]]:
    version = client.get("v", int(version_id)) or {}
    images = version.get("images") or []
    if not images and model_id:
        model = client.get("m", int(model_id)) or {}
        for entry in model.get("modelVersions") or []:
            if entry.get("id") == int(version_id):
                images = entry.get("images") or []
                break
    return [image for image in images if isinstance(image, dict) and image.get("url")]


def choose(images: List[Dict[str, Any]], count: int, videos: bool) -> List[Dict[str, Any]]:
    keep = [image for image in images
            if not image.get("minor")                                    # never: Civitai's minor flag
            and (videos or (image.get("type") or "image") == "image")]
    keep.sort(key=lambda image: (image.get("nsfwLevel") or 1))
    return keep[:count]


def target_name(image: Dict[str, Any], index: int, version_id: int, stem: str, shared: bool) -> str:
    url = str(image["url"])
    suffix = os.path.splitext(urlparse(url).path)[1].lower() or ".jpg"
    image_id = image.get("id")
    if image_id is None:                                   # the version record's images carry no id: use the file name
        image_id = os.path.splitext(os.path.basename(urlparse(url).path))[0] or index
    prefix = stem if shared else str(version_id)
    return f"{prefix}_showcase_{_slug(image_id)}{suffix}"


def _download(client: CivitaiClient, url: str, target: str) -> bool:
    part = target + ".part"
    for attempt in range(3):
        try:
            with client.session.get(url, stream=True, timeout=60) as response:
                if response.status_code == 429:
                    time.sleep(15)
                    continue
                if response.status_code != 200:
                    print(f"[images] HTTP {response.status_code} {url}", flush=True)
                    return False
                with open(part, "wb") as handle:
                    for chunk in response.iter_content(1 << 16):
                        handle.write(chunk)
            if os.path.getsize(part) == 0:
                raise OSError("empty file")
            os.replace(part, target)
            return True
        except Exception as error:
            print(f"[images] attempt {attempt + 1}: {url}: {error}", flush=True)
            time.sleep(2 * (attempt + 1))
    try:
        os.remove(part)
    except OSError:
        pass
    return False


def fetch_for(path: str, version_id: int, model_id: Optional[int], client: CivitaiClient, count: int = DEFAULT_MAX,
              videos: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    folder = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    shared = _shared_folder(folder)
    picked = choose(version_images(client, version_id, model_id), count, videos)
    out_dir = os.path.join(folder, "images")
    saved, planned = 0, []
    for index, image in enumerate(picked, 1):
        target = os.path.join(out_dir, target_name(image, index, version_id, stem, shared))
        if os.path.exists(target):
            continue
        planned.append(target)
        if dry_run:
            continue
        os.makedirs(out_dir, exist_ok=True)
        if _download(client, str(image["url"]), target):
            with open(target + ".json", "w", encoding="utf-8") as handle:
                json.dump(image, handle, indent=2)
            saved += 1
        time.sleep(PAUSE_SECONDS)
    return {"available": len(picked), "saved": saved, "planned": planned}


def fetch_missing(settings: Settings, only: str = "", count: int = DEFAULT_MAX, videos: bool = False,
                  dry_run: bool = False, limit: int = 0, fewer_than: int = 1) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    sql = "select path, rel_dir, civitai_version_id, civitai_model_id from loras where civitai_version_id is not null"
    params: tuple = ()
    if only:
        sql += " and replace(rel_dir, '\\', '/') like ?"
        params = (only.replace("\\", "/") + "%",)
    rows = list(connection.execute(sql, params))
    connection.close()
    client = CivitaiClient(settings)
    fewer_than = max(1, int(fewer_than))
    counts = {"loras": len(rows), "already_have_images": 0, "fetched_for": 0, "images": 0, "none_on_civitai": 0}
    print(f"[images] {len(rows)} LoRAs with a Civitai version; API key {'found' if client.has_key else 'not found'}"
          f"{'  (DRY RUN: nothing is written)' if dry_run else ''}", flush=True)
    done = 0
    for number, row in enumerate(rows, 1):
        if not os.path.exists(row["path"]):
            continue
        if len(preview_images(row["path"], fewer_than)) >= fewer_than:
            counts["already_have_images"] += 1
            continue
        result = fetch_for(row["path"], row["civitai_version_id"], row["civitai_model_id"], client, count, videos, dry_run)
        if not result["available"]:
            counts["none_on_civitai"] += 1
            continue
        counts["fetched_for"] += 1
        counts["images"] += len(result["planned"]) if dry_run else result["saved"]
        if dry_run and counts["fetched_for"] <= 5:
            print(f"[images] would save {len(result['planned'])} for {row['rel_dir']}: "
                  f"{os.path.basename(result['planned'][0]) if result['planned'] else '-'}", flush=True)
        done += 1
        if number % 25 == 0:
            print(f"[images] {number}/{len(rows)}  {counts}", flush=True)
        if limit and done >= limit:
            break
    counts["live_calls"] = client.live_calls
    return counts
