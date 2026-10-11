"""
Evidence that already sits next to each LoRA on disk.

    LoRA Manager    <stem>.metadata.json: sha256, favourite / exclude / notes, and the Civitai block when it was fetched
    Civitai tool    folder layout {modelId}-{Name}\\{versionId}-{VersionName}\\{versionId}_{file}; <file>.json (file info);
                    gallery sidecars <media>.json (in the folder or its images\\ subfolder) with the prompt and the
                    resources used, so the weight the LoRA was actually used at can be read
    ai-toolkit      <output>\\<job>\\config.yaml: trigger_word, arch, dataset folder (captions)

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

MEDIA_SIDECAR = re.compile(r"\.(jpe?g|png|webp|gif|mp4|webm)\.json$", re.I)


def read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# Characters cp1252 maps bytes 0x80-0xFF to: a run of them that starts with a UTF-8 lead byte is mojibake.
_CP1252_HIGH = "\u0080-\u00ff\u20ac\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030\u0160\u2039\u0152\u017d\u2018\u2019\u201c\u201d\u2022\u2013\u2014\u02dc\u2122\u0161\u203a\u0153\u017e\u0178"
_MOJIBAKE_RUN = re.compile(f"[\u00c2-\u00f4][{_CP1252_HIGH}]+")


def fix_mojibake(text: str) -> str:
    """UTF-8 text that was decoded as cp1252 somewhere upstream ('â€“' for an en dash). Each suspicious run is
    repaired on its own, so one genuine non-cp1252 character (an emoji) elsewhere does not block the rest; a run
    whose round trip fails is left as it was."""
    if not text:
        return text

    def repair(match):
        run = match.group(0)
        try:
            return run.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return run

    return _MOJIBAKE_RUN.sub(repair, text)


def strip_html(text: str, limit: int) -> str:
    text = fix_mojibake(text or "")
    text = re.sub(r"<(br|/p|/li|/h\d)\s*/?>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    return text if len(text) <= limit else text[:limit] + " ..."


# ---- LoRA Manager ----------------------------------------------------------------------------------------
def lora_manager(path: str) -> Dict[str, Any]:
    """Fields from <stem>.metadata.json. {} when there is none."""
    data = read_json(os.path.splitext(path)[0] + ".metadata.json")
    if not isinstance(data, dict):
        return {}
    found: Dict[str, Any] = {
        "sha256": str(data.get("sha256") or "").lower(),
        "lm_favorite": 1 if data.get("favorite") else 0,
        "lm_exclude": 1 if data.get("exclude") else 0,
        "lm_notes": str(data.get("notes") or "").strip(),
    }
    civitai = data.get("civitai") if isinstance(data.get("civitai"), dict) else {}
    if civitai.get("id"):
        model = civitai.get("model") if isinstance(civitai.get("model"), dict) else {}
        found.update({
            "civitai_version_id": civitai.get("id"), "civitai_model_id": civitai.get("modelId"),
            "version_name": civitai.get("name"), "model_name": model.get("name"), "model_type": model.get("type"),
            "base_model": civitai.get("baseModel") or data.get("base_model"),
            "trained_words": [str(word).strip() for word in civitai.get("trainedWords") or [] if str(word).strip()],
            "tags": [str(tag) for tag in data.get("tags") or []],
            "description": data.get("modelDescription") or civitai.get("description") or "",
            "nsfw_level": civitai.get("nsfwLevel"),
            "civitai_source": "lm_metadata",
        })
    elif data.get("base_model") and data.get("base_model") != "Unknown":
        found["base_model"] = data.get("base_model")
    words = [str(word).strip() for word in data.get("trainedWords") or [] if str(word).strip()]
    if words and not found.get("trained_words"):
        found["trained_words"] = words
    return found


# ---- the Civitai downloader's layout -----------------------------------------------------------------------
def civitai_ids_from_path(path: str) -> Tuple[Optional[int], Optional[int], str]:
    """(model_id, version_id, source) from {modelId}-{Name}\\{versionId}-{VersionName}\\{versionId}_{file}.
    The file name wins over the folder when they disagree (the same rule as Eric_Krea2/_trigger_words.py)."""
    file_name = os.path.basename(path)
    version_dir = os.path.basename(os.path.dirname(path))
    model_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    from_file = re.match(r"^(\d{4,10})_", file_name)
    from_dir = re.match(r"^(\d{4,10})-", version_dir)
    from_model = re.match(r"^(\d{3,10})-", model_dir)
    version = int(from_file.group(1)) if from_file else (int(from_dir.group(1)) if from_dir else None)
    model = int(from_model.group(1)) if from_model and version is not None else None
    source = "file_name" if from_file else ("folder" if from_dir else "")
    return model, version, source


def downloader_file_info(path: str) -> Dict[str, Any]:
    """<file>.safetensors.json from the Civitai downloader: file-level facts (version id from the download url)."""
    data = read_json(path + ".json")
    if not isinstance(data, dict):
        return {}
    found: Dict[str, Any] = {}
    match = re.search(r"/models/(\d+)", str(data.get("downloadUrl") or ""))
    if match:
        found["civitai_version_id"] = int(match.group(1))
    hashes = data.get("hashes") if isinstance(data.get("hashes"), dict) else {}
    if hashes.get("SHA256"):
        found["sha256"] = str(hashes["SHA256"]).lower()
    return found


_LORA_TAG = re.compile(r"<lora:([^:>]+):([-\d.]+)[^>]*>", re.I)


def gallery_examples(path: str, sha256: str, limit: int, output_name: str = "") -> List[Dict[str, Any]]:
    """Prompts from the gallery sidecars next to the file, with the weight THIS LoRA was used at when it can be told.
    A gallery image names its LoRAs by an AutoV2 hash (the first 10 hex digits of the sha256), or in a <lora:name:w>
    tag whose name is usually the TRAINING output name (ss_output_name), not the file name. The gallery sits in this
    LoRA's own version folder, so a prompt with exactly one <lora:...> tag is taken to be about this LoRA."""
    folder = os.path.dirname(path)
    candidates = [name for name in glob.glob(os.path.join(glob.escape(folder), "*.json")) if MEDIA_SIDECAR.search(name)]
    candidates += [name for name in glob.glob(os.path.join(glob.escape(folder), "images", "*.json")) if MEDIA_SIDECAR.search(name)]
    autov2 = (sha256 or "")[:10].lower()
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    names_for_me = {stem, re.sub(r"^\d{4,10}_", "", stem)}
    if output_name:
        names_for_me.add(output_name.lower())
    examples: List[Dict[str, Any]] = []
    for sidecar in sorted(candidates):
        data = read_json(sidecar)
        if not isinstance(data, dict):
            continue
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        prompt = str(meta.get("prompt") or "").strip()
        if not prompt:
            continue
        weight = None
        for resource in meta.get("resources") or []:
            if isinstance(resource, dict) and str(resource.get("type", "")).lower() in ("lora", "locon", "dora"):
                if autov2 and str(resource.get("hash", "")).lower() == autov2:
                    weight = resource.get("weight")
                    break
        if weight is None:
            hashes = meta.get("hashes") if isinstance(meta.get("hashes"), dict) else {}
            names = {key.split(":", 1)[1].lower() for key, value in hashes.items()
                     if key.lower().startswith("lora:") and autov2 and str(value).lower() == autov2}
            tags = _LORA_TAG.findall(prompt)
            for name, value in tags:
                lowered = name.strip().lower()
                if lowered in names or lowered in names_for_me or (len(lowered) >= 6 and any(
                        lowered.startswith(mine) or mine.startswith(lowered) for mine in names_for_me if len(mine) >= 6)):
                    weight = value
                    break
            if weight is None and len(tags) == 1:
                weight = tags[0][1]
        examples.append({
            "media_file": os.path.relpath(sidecar[:-5], folder),
            "prompt": prompt[:3000],
            "negative": str(meta.get("negativePrompt") or "")[:1000],
            "weight": _as_float(weight),
            "base_checkpoint": str(meta.get("Model") or ""),
            "sampler": str(meta.get("sampler") or ""),
            "steps": meta.get("steps") if isinstance(meta.get("steps"), int) else None,
            "cfg": meta.get("cfgScale") if isinstance(meta.get("cfgScale"), (int, float)) else None,
            "nsfw_level": data.get("nsfwLevel") if isinstance(data.get("nsfwLevel"), int) else None,
        })
        if len(examples) >= limit:
            break
    return examples


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---- ai-toolkit ----------------------------------------------------------------------------------------------
class TrainingJobs:
    """Every ai-toolkit job under the training roots: job name -> config facts and caption samples."""

    def __init__(self, roots: List[str], caption_samples: int):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        for root in roots:
            if not os.path.isdir(root):
                continue
            for job_dir in sorted(glob.glob(os.path.join(glob.escape(root), "*", ""))):
                config_path = os.path.join(job_dir, "config.yaml")
                if os.path.isfile(config_path):
                    facts = self._read(config_path, caption_samples)
                    if facts:
                        self.jobs[os.path.basename(os.path.normpath(job_dir)).lower()] = facts

    @staticmethod
    def _read(config_path: str, caption_samples: int) -> Dict[str, Any]:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception:      # a broken config must not stop the index
            return {}
        process = ((config.get("config") or {}).get("process") or [{}])[0] or {}
        trigger = str(process.get("trigger_word") or "").strip()
        model = process.get("model") or {}
        folders = [dataset.get("folder_path") for dataset in process.get("datasets") or [] if isinstance(dataset, dict)]
        captions: List[str] = []
        for folder in folders:
            if folder and os.path.isdir(folder):
                for caption_file in sorted(glob.glob(os.path.join(glob.escape(folder), "*.txt")))[:caption_samples]:
                    try:
                        with open(caption_file, "r", encoding="utf-8", errors="replace") as handle:
                            captions.append(handle.read().strip()[:500])
                    except OSError:
                        continue
        return {"job": (config.get("config") or {}).get("name") or os.path.basename(os.path.dirname(config_path)),
                "trigger": trigger, "arch": str(model.get("arch") or model.get("name_or_path") or ""),
                "base": str(model.get("name_or_path") or ""), "captions": captions[:caption_samples],
                "config_path": config_path}

    def match(self, stem: str) -> Optional[Dict[str, Any]]:
        """A job whose name is the file's stem, or the stem minus an ai-toolkit step suffix (_000001000)."""
        lowered = stem.lower()
        if lowered in self.jobs:
            return self.jobs[lowered]
        base = re.sub(r"_\d{6,}$", "", lowered)
        return self.jobs.get(base)
