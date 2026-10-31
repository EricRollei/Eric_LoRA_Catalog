"""
Which LoRAs made an image: read from the metadata image files carry, matched to the catalog (2026-10-13).

    refs = read_file(path) / read_bytes(data)      -> [LoraRef]   what the file says (no catalog needed)
    found = match(catalog_path, refs)              -> [dict]      each ref matched to your library, or not

What it reads
    Sweep cells      Eric_Krea2 sweep cell PNGs ('sweep_cell' chunk: the cell's LoRA or stack)
    ComfyUI PNG      'prompt' chunk (the API graph): every LoRA loader input - LoraLoader / LoraLoaderModelOnly,
                     stack and multi-slot loaders (lora_name_1 ...), rgthree Power Lora ({on, lora, strength}),
                     LoRA Manager lists ({name, strength, active}), <lora:..> tags in text inputs. Bypassed or
                     switched-off slots are skipped. Falls back to the 'workflow' chunk (UI graph) without 'prompt'.
    A1111 / Forge    'parameters' text: <lora:name:w> tags, "Lora hashes", "Hashes" JSON, "Civitai resources" JSON
    Civitai          the same text in a JPEG / WebP EXIF UserComment (downloads from civitai.com)
Matching, best first: sha256 (full or AutoV2 = first 10 hex), Civitai version id, file name / stem.
A pasted screenshot has no metadata; an image saved by a browser from a web page usually keeps it.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LORA_TAG = re.compile(r"<(?:lora|lyco):([^:>]+)(?::([-\d.]+))?[^>]*>", re.I)
LORA_EXT = (".safetensors", ".pt", ".ckpt", ".bin")
STRENGTH_KEYS = ("strength_model", "strength", "lora_wt", "weight", "model_strength", "lora_strength")


@dataclass
class LoraRef:
    name: str = ""                       # file name, stem or display name as the image wrote it
    weight: Optional[float] = None
    sha: str = ""                        # full sha256 or AutoV2 (10 hex), lower case
    version_id: Optional[int] = None
    source: str = ""                     # comfy / workflow / a1111 / civitai
    extra: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return self.sha or (f"v{self.version_id}" if self.version_id else _stem(self.name).lower())


def _stem(name: str) -> str:
    base = re.split(r"[\\/]", str(name))[-1]
    return base[: -len(ext)] if (ext := next((e for e in LORA_EXT if base.lower().endswith(e)), "")) else base


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -- reading -------------------------------------------------------------------------------------------------------
def read_bytes(data: bytes) -> List[LoraRef]:
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        return []
    return _read_image(image)


def read_file(path: str) -> List[LoraRef]:
    from PIL import Image
    try:
        with Image.open(path) as image:
            return _read_image(image)
    except Exception:
        return []


def _read_image(image) -> List[LoraRef]:
    info = dict(getattr(image, "info", {}) or {})
    texts: Dict[str, str] = {k: v for k, v in info.items() if isinstance(v, str)}
    comment = _exif_comment(image)
    if comment:
        texts.setdefault("parameters", comment)
    refs: List[LoraRef] = []
    if texts.get("prompt"):
        refs += _from_comfy_prompt(texts["prompt"])
    if not refs and texts.get("workflow"):
        refs += _from_comfy_workflow(texts["workflow"])
    if texts.get("sweep_cell"):
        refs += _from_sweep_cell(texts["sweep_cell"])
    for key in ("parameters", "Comment", "comment", "Description"):
        if texts.get(key):
            refs += _from_a1111(texts[key])
    return _dedupe(refs)


def _exif_comment(image) -> str:
    """EXIF UserComment (0x9286, in the Exif IFD) - where Civitai and A1111 JPEG/WebP files keep their parameters."""
    try:
        exif = image.getexif()
        raw = exif.get_ifd(0x8769).get(0x9286) or exif.get(0x9286)
    except Exception:
        return ""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    head, body = raw[:8], raw[8:]
    if head.startswith(b"UNICODE"):
        for codec in ("utf-16-be", "utf-16-le"):
            try:
                text = body.decode(codec)
                if text.isprintable() or "\n" in text:
                    return text.strip("\x00")
            except UnicodeDecodeError:
                continue
    if head.startswith(b"ASCII") or head == b"\x00" * 8:
        return body.decode("utf-8", "ignore").strip("\x00")
    return raw.decode("utf-8", "ignore").strip("\x00")


def _from_comfy_prompt(text: str) -> List[LoraRef]:
    try:
        graph = json.loads(text)
    except ValueError:
        return []
    refs: List[LoraRef] = []
    if not isinstance(graph, dict):
        return refs
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        if isinstance(inputs, dict):
            refs += _walk_inputs(inputs, str(node.get("class_type", "")))
    return refs


def _walk_inputs(inputs: Dict[str, Any], class_type: str) -> List[LoraRef]:
    refs: List[LoraRef] = []
    for key, value in inputs.items():
        low = key.lower()
        if isinstance(value, str):
            if "lora" in low and value.lower().endswith(LORA_EXT):
                if _switched_off(inputs, key):
                    continue
                refs.append(LoraRef(name=value, weight=_weight_for(inputs, key), source="comfy",
                                    extra={"node": class_type}))
            elif "<lora:" in value.lower() or "<lyco:" in value.lower():
                refs += [LoraRef(name=m.group(1), weight=_num(m.group(2)), source="comfy") for m in LORA_TAG.finditer(value)]
        elif isinstance(value, dict):
            refs += _from_entry(value, class_type)
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            for entry in value:
                refs += _from_entry(entry, class_type)
    return refs


def _from_entry(entry: Dict[str, Any], class_type: str) -> List[LoraRef]:
    """rgthree Power Lora {on, lora, strength}, LoRA Manager {name, strength, active}, {'__value__': [...]} lists."""
    if "__value__" in entry and isinstance(entry["__value__"], list):
        return [r for e in entry["__value__"] if isinstance(e, dict) for r in _from_entry(e, class_type)]
    if entry.get("on") is False or entry.get("active") is False or entry.get("enabled") is False:
        return []
    name = entry.get("lora") or entry.get("lora_name") or entry.get("name") or entry.get("file")
    if not isinstance(name, str) or not name or name.lower() in ("none", ""):
        return []
    if not (name.lower().endswith(LORA_EXT) or "lora" in class_type.lower()):
        return []
    weight = next((_num(entry[k]) for k in STRENGTH_KEYS if k in entry and _num(entry[k]) is not None), None)
    return [LoraRef(name=name, weight=weight, source="comfy", extra={"node": class_type})]


def _switched_off(inputs: Dict[str, Any], key: str) -> bool:
    suffix = re.search(r"(_?\d+)$", key)
    tail = suffix.group(1) if suffix else ""
    for flag in (f"on{tail}", f"enabled{tail}", f"active{tail}", f"switch{tail}", f"lora{tail}_on"):
        if flag in inputs and inputs[flag] in (False, "Off", "off", 0):
            return True
    return False


def _weight_for(inputs: Dict[str, Any], key: str) -> Optional[float]:
    suffix = re.search(r"(_?\d+)$", key)
    tail = suffix.group(1) if suffix else ""
    candidates = [f"{k}{tail}" for k in STRENGTH_KEYS] + [f"{k}_{tail.lstrip('_')}" for k in STRENGTH_KEYS if tail]
    candidates += [k for k in inputs if k.lower().startswith("strength") and tail and k.endswith(tail.lstrip("_"))]
    if not tail:
        candidates += list(STRENGTH_KEYS)
    for name in candidates:
        if name in inputs and _num(inputs[name]) is not None:
            return _num(inputs[name])
    return None


def _from_comfy_workflow(text: str) -> List[LoraRef]:
    """UI-format graph: LoRA-ish nodes (type mentions lora) that are not bypassed/muted; a .safetensors widget value
    and the first number after it."""
    try:
        graph = json.loads(text)
    except ValueError:
        return []
    refs: List[LoraRef] = []
    for node in (graph.get("nodes") or []) if isinstance(graph, dict) else []:
        if not isinstance(node, dict) or node.get("mode") in (2, 4) or "lora" not in str(node.get("type", "")).lower():
            continue
        values = node.get("widgets_values")
        if isinstance(values, dict):
            refs += _walk_inputs(values, str(node.get("type")))
            continue
        values = values or []
        for i, value in enumerate(values):
            if isinstance(value, dict):
                refs += _from_entry(value, str(node.get("type")))
            elif isinstance(value, str) and value.lower().endswith(LORA_EXT):
                weight = next((_num(v) for v in values[i + 1:i + 3] if _num(v) is not None and not isinstance(v, bool)), None)
                refs.append(LoraRef(name=value, weight=weight, source="workflow", extra={"node": node.get("type")}))
    return refs


def _from_sweep_cell(text: str) -> List[LoraRef]:
    """An Eric_Krea2 sweep cell PNG: its overrides (lora_file + lora_strength, or a 'name @w | name @w' stack)."""
    try:
        cell = json.loads(text)
    except ValueError:
        return []
    if not isinstance(cell, dict):
        return []
    if cell.get("lora_file"):
        return [LoraRef(name=str(cell["lora_file"]).partition(" #")[0].strip(),    # ' #word option' (2026-10-20)
                        weight=_num(cell.get("lora_strength")), source="sweep")]
    refs = []
    for part in str(cell.get("lora_stack") or "").split("|"):
        name, _, w = part.strip().rpartition("@") if "@" in part else (part.strip(), "", "")
        name = name.partition(" #")[0]
        if name.strip():
            refs.append(LoraRef(name=name.strip(), weight=_num(w.split("/")[0]) if w else 1.0, source="sweep"))
    return refs


def _from_a1111(text: str) -> List[LoraRef]:
    refs: List[LoraRef] = []
    for m in LORA_TAG.finditer(text):
        refs.append(LoraRef(name=m.group(1), weight=_num(m.group(2)), source="a1111"))
    hashes = re.search(r'Lora hashes:\s*"([^"]+)"', text)
    if hashes:
        for part in hashes.group(1).split(","):
            if ":" in part:
                name, h = part.rsplit(":", 1)
                refs.append(LoraRef(name=name.strip(), sha=h.strip().lower(), source="a1111"))
    block = re.search(r"Hashes:\s*(\{.*?\})", text)
    if block:
        try:
            for k, h in json.loads(block.group(1)).items():
                if str(k).lower().startswith(("lora:", "lyco:")) and isinstance(h, str):
                    refs.append(LoraRef(name=str(k).split(":", 1)[1], sha=h.lower(), source="a1111"))
        except ValueError:
            pass
    civ = re.search(r"Civitai resources:\s*(\[.*?\])(?:,\s*\w[\w ]*:|\s*$)", text, re.S)
    if civ:
        try:
            for r in json.loads(civ.group(1)):
                if str(r.get("type", "")).lower() in ("lora", "locon", "lycoris", "dora"):
                    refs.append(LoraRef(name=str(r.get("modelName") or ""), weight=_num(r.get("weight")),
                                        version_id=int(r["modelVersionId"]) if r.get("modelVersionId") else None,
                                        source="civitai", extra={"version": r.get("modelVersionName") or ""}))
        except (ValueError, TypeError, AttributeError):
            pass
    return refs


def _dedupe(refs: List[LoraRef]) -> List[LoraRef]:
    """One entry per LoRA: refs naming the same file / hash / version merge (a hash and a weight from different lines
    of the same text end up together)."""
    out: List[LoraRef] = []
    for r in refs:
        twin = next((o for o in out if (r.sha and o.sha and (r.sha.startswith(o.sha) or o.sha.startswith(r.sha)))
                     or (r.version_id and r.version_id == o.version_id)
                     or (r.name and o.name and _stem(r.name).lower() == _stem(o.name).lower())), None)
        if twin is None:
            out.append(r)
            continue
        twin.weight = twin.weight if twin.weight is not None else r.weight
        twin.sha = twin.sha or r.sha
        twin.version_id = twin.version_id or r.version_id
        if len(r.name) > len(twin.name) and not twin.name.lower().endswith(LORA_EXT):
            twin.name = r.name
    return out


# -- matching to the catalog ---------------------------------------------------------------------------------------
def match(catalog_path: str, refs: List[LoraRef]) -> List[Dict[str, Any]]:
    """Each ref -> {name, weight, source, owned, path, arch, card_name, how, version_id, url}. Not owned: path None,
    url to its Civitai version when the image gave one."""
    if not refs:
        return []
    out = []
    connection = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True, timeout=10)
    try:
        for r in refs:
            row, how = None, ""
            if r.sha:
                row = connection.execute(
                    "select path, arch_folder, coalesce(model_name, stem), civitai_version_id from loras where "
                    + ("lower(sha256) = ?" if len(r.sha) > 12 else "lower(substr(sha256, 1, 10)) = ?"),
                    (r.sha if len(r.sha) > 12 else r.sha[:10],)).fetchone()
                how = "hash" if row else ""
            if row is None and r.version_id:
                row = connection.execute("select path, arch_folder, coalesce(model_name, stem), civitai_version_id "
                                         "from loras where civitai_version_id = ?", (r.version_id,)).fetchone()
                how = "Civitai version" if row else ""
            if row is None and r.name:
                stem = _stem(r.name).lower()
                row = connection.execute("select path, arch_folder, coalesce(model_name, stem), civitai_version_id "
                                         "from loras where lower(stem) = ? or lower(file_name) = ?",
                                         (stem, re.split(r"[\\/]", r.name)[-1].lower())).fetchone()
                how = "file name" if row else ""
            card_name = None
            if row is not None:
                hit = connection.execute("select name from card_facets where path = ?", (row[0],)).fetchone()
                card_name = hit[0] if hit else None
            version = r.version_id or (row[3] if row else None)
            out.append({"name": card_name or (row[2] if row else None) or _stem(r.name) or f"version {r.version_id}",
                        "image_name": r.name, "weight": r.weight, "source": r.source, "owned": row is not None,
                        "path": row[0] if row else None, "arch": (row[1] or "") if row else "", "how": how,
                        "version_id": version,
                        "url": f"https://civitai.com/model-versions/{version}" if version and row is None else ""})
    finally:
        connection.close()
    return out


def describe(found: List[Dict[str, Any]]) -> List[str]:
    """Report lines (Suggester report, run.py)."""
    lines = []
    for f in found:
        w = f" @{f['weight']:g}" if f.get("weight") is not None else ""
        if f["owned"]:
            lines.append(f"  {f['name']}{w}  - yours ({f['arch'] or '?'}, matched by {f['how']})")
        else:
            lines.append(f"  {f['name']}{w}  - not in your library" + (f": {f['url']}" if f.get("url") else ""))
    return lines
