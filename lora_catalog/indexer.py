"""
The indexer: walk the libraries, assemble each LoRA's evidence, write the catalog.

Evidence order when sources disagree (cheapest and most specific first):
    identity     LoRA Manager sha256 > downloader file info sha256 > embedded modelspec hash > our own computed hash
                 (run.py hash, kept in file_hashes while the file is unchanged); version id from the
                 file name > folder > downloader download url
    Civitai      LoRA Manager's own Civitai block (fetched for THIS file) > offline mirror by sha256 > by version id
    triggers     Civitai trainedWords > ai-toolkit trigger_word > modelspec.trigger_phrase > ss_tag_frequency (kept
                 separately as ss_tags; it is a hint, not a trigger list)

Unchanged files (same size and mtime) are skipped on a re-index unless --force.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import hashes
from . import header as st_header
from .civitai_db import CivitaiMirror
from .config import Settings
from .crawl import cached_record
from .local_sources import (TrainingJobs, civitai_ids_from_path, downloader_file_info, fix_mojibake, gallery_examples,
                            lora_manager, strip_html)
from .store import Catalog

LORA_EXTENSIONS = (".safetensors",)


def walk(settings: Settings, only: str = "") -> Iterator[Tuple[str, str]]:
    """(root, path) for every LoRA file. `only` limits the walk to one relative folder, e.g. 'Krea2/graphic_art'."""
    skip = {name.lower() for name in settings.skip_dirs}
    for root in settings.library_roots:
        if not os.path.isdir(root):
            continue
        start = os.path.join(root, only) if only else root
        if not os.path.isdir(start):
            continue
        for folder, dirs, files in os.walk(start):
            dirs[:] = [name for name in dirs if name.lower() not in skip and not name.startswith((".", "_"))]
            for name in files:
                if name.lower().endswith(LORA_EXTENSIONS):
                    yield root, os.path.join(folder, name)


def _relative(root: str, path: str) -> Tuple[str, str]:
    """(top folder = the architecture shelf, relative folder below the root)."""
    relative = os.path.relpath(os.path.dirname(path), root)
    parts = [] if relative == "." else relative.split(os.sep)
    return (parts[0] if parts else ""), relative.replace(os.sep, "/") if parts else ""


_BY_HASH: Optional[Dict[str, Any]] = None


def _by_hash() -> Dict[str, Any]:
    global _BY_HASH
    if _BY_HASH is None:
        from .community import hashed_versions
        _BY_HASH = hashed_versions()
    return _BY_HASH


def build_row(root: str, path: str, settings: Settings, mirror: CivitaiMirror, jobs: TrainingJobs,
              computed: Optional[Dict[str, Tuple[str, str]]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    stat = os.stat(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    arch_folder, rel_dir = _relative(root, path)
    row: Dict[str, Any] = {
        "path": path, "file_name": os.path.basename(path), "stem": stem, "size": stat.st_size, "mtime": stat.st_mtime,
        "sig": f"{stat.st_size}|{int(stat.st_mtime)}", "root": root, "arch_folder": arch_folder, "rel_dir": rel_dir,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    problems: List[str] = []

    # header
    try:
        metadata, keys = st_header.read_header(path)
    except Exception as error:      # a damaged file is reported, not fatal
        metadata, keys = {}, []
        problems.append(f"header unreadable: {str(error)[:80]}")
    row["header_meta"] = st_header.summarise(metadata)
    row["key_signature"] = st_header.key_signature(keys) if keys else ""
    row["network_dim"] = st_header.number(metadata.get("ss_network_dim"))
    row["network_alpha"] = st_header.number(metadata.get("ss_network_alpha"))
    row["ss_tags"] = st_header.ss_tags(metadata)
    row["output_name"] = str(metadata.get("ss_output_name") or metadata.get("name") or "")

    # identity
    manager = lora_manager(path)
    file_info = downloader_file_info(path)
    sha = manager.get("sha256") or file_info.get("sha256") or st_header.embedded_sha256(metadata)
    own = None if sha else hashes.lookup(computed or {}, path, row["sig"])
    sha = sha or own
    row["sha256"] = sha
    row["sha_source"] = ("lora_manager" if manager.get("sha256") else "downloader" if file_info.get("sha256")
                         else "computed" if own else "embedded" if sha else "")
    model_id, version_id, id_source = civitai_ids_from_path(path)
    if version_id is None and file_info.get("civitai_version_id"):
        version_id, id_source = file_info["civitai_version_id"], "download_url"
    if version_id is None and sha:              # found by `run.py identify` (Civitai lookup by sha256, 2026-10-10)
        found = _by_hash().get(str(sha).lower())
        if found:
            version_id, id_source = int(found), "by_hash"
    for key in ("lm_favorite", "lm_exclude", "lm_notes"):
        row[key] = manager.get(key)

    # Civitai fields
    civitai: Dict[str, Any] = {}
    if manager.get("civitai_source"):
        civitai = {key: value for key, value in manager.items() if key not in ("sha256", "lm_favorite", "lm_exclude", "lm_notes")}
    else:
        civitai = mirror.lookup(sha, version_id) or cached_record(version_id, settings)
        if manager.get("trained_words") and not civitai.get("trained_words"):
            civitai["trained_words"] = manager["trained_words"]
        if manager.get("base_model") and not civitai.get("base_model"):
            civitai["base_model"] = manager["base_model"]
    row.update({key: value for key, value in civitai.items() if value not in (None, "", [])})
    row["civitai_model_id"] = row.get("civitai_model_id") or model_id
    row["civitai_version_id"] = row.get("civitai_version_id") or version_id
    row["id_source"] = "civitai_record" if civitai.get("civitai_version_id") else id_source
    if row.get("description"):
        row["description"] = strip_html(row["description"], settings.max_description_chars)
    for key in ("model_name", "version_name"):
        if isinstance(row.get(key), str):
            row[key] = fix_mojibake(row[key])

    # your own training
    job = jobs.match(stem)
    if job:
        row["training_job"] = job["job"]
        row["training_trigger"] = job["trigger"]
        row["training_arch"] = job["arch"]
        row["caption_samples"] = job["captions"]

    # where it came from
    if row.get("civitai_source") or row.get("civitai_version_id"):
        row["source_kind"] = "civitai"
    elif job or metadata.get("software") or "ai-toolkit" in str(metadata).lower():
        row["source_kind"] = "self_trained" if job else "unknown"
    else:
        row["source_kind"] = "unknown"

    examples = gallery_examples(path, sha, settings.max_examples_per_lora, row["output_name"])
    row["example_count"] = len(examples)
    row["problems"] = problems
    return row, examples


def index(settings: Settings, only: str = "", force: bool = False, limit: int = 0, progress_every: int = 250) -> Dict[str, int]:
    started = time.time()
    catalog = Catalog(settings.catalog_path)
    known = {} if force else catalog.signatures()
    print(f"[LoRA Catalog] loading the offline Civitai mirror ...", flush=True)
    mirror = CivitaiMirror(settings.lm_civitai_db)
    print(f"[LoRA Catalog] mirror: {'%d hashes' % len(mirror.by_sha) if mirror.available else 'not found, skipped'} "
          f"({time.time() - started:.1f} s)", flush=True)
    jobs = TrainingJobs(settings.training_roots, settings.max_caption_samples)
    computed = hashes.load(catalog.connection)
    print(f"[LoRA Catalog] ai-toolkit jobs: {len(jobs.jobs)}", flush=True)

    counts = {"seen": 0, "indexed": 0, "unchanged": 0, "failed": 0}
    live = set()
    for root, path in walk(settings, only):
        counts["seen"] += 1
        live.add(path)
        try:
            stat = os.stat(path)
            if known.get(path) == f"{stat.st_size}|{int(stat.st_mtime)}":
                counts["unchanged"] += 1
                continue
            row, examples = build_row(root, path, settings, mirror, jobs, computed)
            catalog.put_lora(row)
            catalog.put_examples(path, examples)
            counts["indexed"] += 1
        except Exception as error:      # one bad file never stops the walk
            counts["failed"] += 1
            print(f"[LoRA Catalog] failed: {path}: {error}", flush=True)
        if counts["seen"] % progress_every == 0:
            catalog.commit()
            print(f"[LoRA Catalog] {counts['seen']} files ({counts['indexed']} indexed, {counts['unchanged']} unchanged) "
                  f"{time.time() - started:.0f} s", flush=True)
        if limit and counts["indexed"] >= limit:
            break
    if not only and not limit:
        counts["removed"] = catalog.forget_missing(live, settings.library_roots)
    catalog.set_meta("last_index", time.strftime("%Y-%m-%d %H:%M"))
    catalog.close()
    mirror.close()
    counts["seconds"] = int(time.time() - started)
    return counts
