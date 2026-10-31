"""
LoRA Suggester (Advanced).

Everything the LoRA Suggester does, plus a reference image: style_image steers the look LoRAs (style, transform,
treatment) toward the image's style, measured with the LoRA Catalog's style vectors (CSD, from each LoRA's preview
images; docs/design/STYLE_SEARCH.md parts 3-4). Lighting, detailers, subjects and sliders stay with the prompt words.

    style_image          one or more images (a batch). Nothing connected = the plain Suggester.
    style_image_weight   how much the look counts as evidence next to the prompt words (engine LOOK_RAW): 0 = report
                         only; 0.4 default = a clear look match about equals a strong word match (an image alone
                         scores its best match 0.80); a look below the median counts against; 1 = the look dominates
    image_mode           one look (average): several examples of ONE style (4 images of one LoRA: right LoRA 1st 84%
                         of the time, vs 65% for one image); each image (best match): references that DIFFER (a
                         LoRA counts by its closest image; averaging two different looks lands between them)

Needs style search set up in the LoRA Catalog (settings.json "style_enabled", the CSD weights, run.py style). Without
it the image is ignored and the report says why. The model runs on ComfyUI's device and unloads after
style_idle_unload_min idle minutes.

Style words (2026-10-13): a picked style / transform / treatment LoRA that the LoRA Catalog's render check found to
show its look only with its own style words in the prompt (run.py style --renders: needs_words) gets up to three
words from its card (medium, rendering, line), added with the triggers. The report says which.

LoRAs named in the image (2026-10-13): the IMAGE input carries pixels only, so the node follows the style_image wire
back through the graph to the Load Image node(s) feeding it and reads those files' metadata (ComfyUI workflow, A1111 /
Civitai parameters, sweep cells - lora_catalog/image_meta.py). LoRAs you own count as strong evidence for their role
(engine USED_RAW; not utility helpers); the report lists every LoRA found, owned or not, with the image's weight.

The plain LoRA Suggester node is untouched; this is a separate node with the same outputs. Its own widgets come
first, so new widgets the plain node may append later (always at the end) cannot shift them in saved workflows.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use). See LICENSE.
"""

import os
import traceback

from ..suggest import SuggestRequest, suggest
from .lora_suggest_node import LoRASuggester, _Sections, checkpoint_in_graph, lora_text, merge_triggers

IMAGE_MODES = ["one look (average)", "each image (best match)"]
_HOLDER = None


def _embed(style_image):
    """IMAGE batch -> (CSD vectors, '') or (None, why not)."""
    global _HOLDER
    from ..lora_catalog import style
    from ..lora_catalog.config import load_settings as catalog_settings

    settings = catalog_settings()
    reason = style.available(settings)
    if reason:
        return None, reason
    try:
        import comfy.model_management as model_management
        device = str(model_management.get_torch_device())
    except Exception:                     # outside ComfyUI (tests): settings style_device
        device = None
    if _HOLDER is None or _HOLDER.device != device:
        _HOLDER = style.ModelHolder(settings, device=device)
    from PIL import Image
    batch = style_image if style_image.dim() == 4 else style_image.unsqueeze(0)
    pixels = (batch[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy()
    return _HOLDER.embed_images([Image.fromarray(p) for p in pixels]), ""


def _image_files(prompt, unique_id, limit=32):
    """The image files behind this node's style_image input: walk the graph upstream from it and collect any string input
    that names an existing image (Load Image's 'image' widget resolves through ComfyUI's input folder)."""
    if not isinstance(prompt, dict) or unique_id is None:
        return []
    me = prompt.get(str(unique_id)) or {}
    start = (me.get("inputs") or {}).get("style_image")
    if not isinstance(start, list) or not start:
        return []
    try:
        import folder_paths
    except ImportError:
        folder_paths = None
    files, seen, todo = [], set(), [str(start[0])]
    while todo and len(seen) < limit:
        node_id = todo.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        for value in ((prompt.get(node_id) or {}).get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                todo.append(str(value[0]))
            elif isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                path = value if os.path.isabs(value) else None
                if path is None and folder_paths is not None:
                    try:
                        path = folder_paths.get_annotated_filepath(value)
                    except Exception:
                        path = None
                if path and os.path.isfile(path) and path not in files:
                    files.append(path)
    return files


def _image_loras(prompt, unique_id):
    """({LoRA path: reason note} for owned LoRAs named in the style images' metadata, report lines)."""
    from ..lora_catalog import image_meta
    from ..lora_catalog.config import load_settings as catalog_settings

    files = _image_files(prompt, unique_id)
    if not files:
        return {}, ["LoRAs in the image: no image file found behind style_image (only Load Image files can be read)"]
    catalog = catalog_settings().catalog_path
    notes, lines = {}, []
    for path in files:
        found = image_meta.match(catalog, image_meta.read_file(path))
        name = os.path.basename(path)
        if not found:
            lines.append(f"LoRAs in {name}: none named in its metadata")
            continue
        lines.append(f"LoRAs in {name}:")
        lines += image_meta.describe(found)
        for f in found:
            if f["owned"] and f["path"] not in notes:
                w = f" @{f['weight']:g}" if f.get("weight") is not None else ""
                notes[f["path"]] = f"used in the style image ({name}{w})"
    return notes, lines


class LoRASuggesterAdvanced(LoRASuggester):
    CATEGORY = "Eric Prompt Enhancers"
    FUNCTION = "run_advanced"

    @classmethod
    def INPUT_TYPES(cls):
        base = LoRASuggester.INPUT_TYPES()
        required = {
            "style_image_weight": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "How much the style image's look counts, as evidence next to the prompt words, "
                                              "for style / transform / treatment LoRAs. 0 = report only. 0.4: a clear look "
                                              "match counts about as much as a strong word match, so an image alone picks a "
                                              "style and a styled prompt still has its say; a look far from the image counts "
                                              "against. 1 = the look outweighs most prompt words."}),
            "image_mode": (IMAGE_MODES, {"default": IMAGE_MODES[0],
                           "tooltip": "Several style images: one look (average) for examples of ONE style (more images = "
                                      "more accurate); each image (best match) for references that DIFFER (each LoRA is "
                                      "compared with its closest image). One image: the same either way."}),
        }
        required.update(base["required"])
        optional = dict(base["optional"])
        optional["style_image"] = ("IMAGE", {"tooltip": "Reference image(s) for the look. Needs style search in the LoRA "
                                                        "Catalog (settings.json style_enabled, run.py style-setup, run.py style). "
                                                        "LoRAs named in a loaded image's own metadata (ComfyUI workflow, "
                                                        "A1111 / Civitai parameters) count as strong evidence."})
        # hidden: the graph, to find the Load Image file(s) behind style_image (no widget, no effect on saved workflows)
        return {"required": required, "optional": optional, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    def run_advanced(self, style_image_weight=0.4, image_mode=IMAGE_MODES[0], platform="krea2", mode="suggest",
                     nsfw="off", max_loras=4, style_max=2, treatment_max=2, detailer_max=1, environment_max=1,
                     lighting_max=1, subject_max=1, layout_max=1, utility_max=1, apply_threshold=0.35,
                     weight_budget=2.5, stage_profile="role", variety=0.2, seed=0, sections=None, text="", exclude="",
                     pin="", transform_max=1, slider_max=2, slider_range=3.0, slider_min_strength=2, style_image=None,
                     prompt=None, unique_id=None):
        vectors, note = None, ""
        image_loras, meta_lines = {}, []
        if style_image is not None:
            try:
                image_loras, meta_lines = _image_loras(prompt, unique_id)
            except Exception as error:                  # never break the graph over metadata
                traceback.print_exc()
                meta_lines = [f"LoRAs in the image: not read ({type(error).__name__}: {error})"]
        if style_image is not None:
            try:
                vectors, note = _embed(style_image)
            except Exception as error:                  # never break the graph over the image
                traceback.print_exc()
                vectors, note = None, f"{type(error).__name__}: {error}"
        bundle = _Sections(sections)
        request = SuggestRequest(
            platform=platform, sections=dict(bundle.sections), text=text or "", max_loras=int(max_loras),
            role_max={"style": style_max, "transform": transform_max, "treatment": treatment_max, "detailer": detailer_max,
                      "environment": environment_max, "lighting": lighting_max, "subject": subject_max,
                      "layout": layout_max, "utility": utility_max, "slider": slider_max},
            threshold=float(apply_threshold), weight_budget=float(weight_budget), stage_profile=stage_profile,
            variety=float(variety), seed=int(seed), nsfw=nsfw, exclude=exclude or "", pin=pin or "",
            prompt_nsfw=bundle.nsfw if sections is not None else None,
            slider_range=float(slider_range), slider_min_strength=int(slider_min_strength),
            style_vectors=vectors, style_weight=float(style_image_weight),
            image_mode="each" if str(image_mode).startswith("each") else "average", add_style_words=True,
            image_loras=image_loras or None, checkpoint=checkpoint_in_graph(prompt))
        try:
            result = suggest(request)
        except Exception as error:      # never break the graph over a suggestion
            traceback.print_exc()
            return ([], "", text or "", f"LoRA Suggester error: {error}", "")
        entries = [pick.entry() for pick in result.picks]
        header = "APPLY: sent to the loader" if mode == "apply" else "SUGGEST: report only (set mode to apply to load these)"
        image_line = f"\nstyle image ignored: {note}" if note else ""
        if meta_lines:
            image_line += "\n" + "\n".join(meta_lines)
        report = f"{header}{image_line}\n{result.report}"
        triggers = ", ".join(result.triggers)
        print(f"[LoRA Suggester Advanced] {mode}: " + (", ".join(f"{e['role']}={e['name']}" for e in entries) or "no picks"))
        return (entries if mode == "apply" else [], triggers, merge_triggers(result.prompt, result.triggers), report,
                lora_text(entries) if mode == "apply" else "")
