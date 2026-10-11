"""
The safetensors header: the only part of a LoRA file the indexer reads. Never loads a tensor.

What it gives:
    metadata        the __metadata__ dict (kohya ss_*, modelspec.*, ai-toolkit, OneTrainer fields)
    ss_tags         kohya / ai-toolkit dataset tags by frequency; for a single-concept LoRA these ARE the trigger
    network dim     rank and alpha, when written
    key_signature   the tensor names with numbers folded away, most common first. It identifies the architecture the
                    LoRA was trained against without anyone having to know every architecture in advance: two LoRAs
                    for the same base model share a signature. The refiner (phase 3) maps signatures to names.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter
from typing import Any, Dict, List, Tuple

MAX_HEADER_BYTES = 100 * 1024 * 1024
KEEP_META_KEYS = (
    "ss_output_name", "ss_base_model_version", "ss_sd_model_name", "ss_network_module", "ss_network_dim", "ss_network_alpha",
    "ss_resolution", "ss_num_train_images", "ss_max_train_steps", "ss_training_comment", "ss_v2",
    "modelspec.title", "modelspec.architecture", "modelspec.implementation", "modelspec.description", "modelspec.trigger_phrase",
    "modelspec.author", "modelspec.date", "modelspec.hash_sha256", "modelspec.tags", "modelspec.usage_hint",
    "name", "version", "software", "training_info", "format",
)
_DIGITS = re.compile(r"(?<=[._])\d+(?=[._]|$)")


def read_header(path: str) -> Tuple[Dict[str, Any], List[str]]:
    """(metadata dict, tensor key names). ({}, []) when the file is not a readable safetensors file."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            return {}, []
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > MAX_HEADER_BYTES:
            return {}, []
        header = json.loads(handle.read(length))
    metadata = header.pop("__metadata__", {}) if isinstance(header, dict) else {}
    return (metadata if isinstance(metadata, dict) else {}), list(header.keys()) if isinstance(header, dict) else []


def key_signature(keys: List[str], top: int = 6) -> str:
    """The LoRA's tensor-name pattern: numbers folded to '#', the module path cut at 4 parts, LoRA suffixes dropped."""
    shapes: Counter = Counter()
    for key in keys:
        name = re.sub(r"\.(lora_[AaBb]|lora_up|lora_down|alpha|lora\.up|lora\.down|diff|diff_b|dora_scale)(\.weight)?$", "", key)
        name = _DIGITS.sub("#", name)
        shapes[".".join(name.split(".")[:4])] += 1
    return " | ".join(shape for shape, _count in shapes.most_common(top))


def ss_tags(metadata: Dict[str, Any], limit: int = 30) -> List[str]:
    """Dataset tags by frequency from ss_tag_frequency (kohya, ai-toolkit). [] when absent or unreadable."""
    raw = metadata.get("ss_tag_frequency")
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    counts: Counter = Counter()
    for tags in (data or {}).values():
        if isinstance(tags, dict):
            for tag, count in tags.items():
                tag = str(tag).strip()
                if tag:
                    counts[tag] += count if isinstance(count, (int, float)) else 1
    return [tag for tag, _count in counts.most_common(limit)]


def summarise(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The metadata fields worth keeping (the full dict can be megabytes of tag frequencies)."""
    kept = {}
    for key in KEEP_META_KEYS:
        value = metadata.get(key)
        if value not in (None, "", "None"):
            text = str(value)
            kept[key] = text if len(text) <= 600 else text[:600] + "..."
    return kept


def number(value: Any):
    try:
        return float(value) if "." in str(value) else int(value)
    except (TypeError, ValueError):
        return None


def embedded_sha256(metadata: Dict[str, Any]) -> str:
    value = str(metadata.get("modelspec.hash_sha256") or metadata.get("hash_sha256") or "").strip().lower()
    if value.startswith("0x"):
        value = value[2:]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""
