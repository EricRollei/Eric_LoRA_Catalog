"""
LoRA Suggester.

Reads a Wildcard Composer's sections (or any prompt text) and picks LoRAs from your LoRA Catalog for it: by ROLE
(style, treatment, detailer, environment, lighting, subject, layout, utility), with per-stage weights for the
Krea2 Multi-Stage (S1 composition, S2 refine, S3 detail). No LLM call; the cards were written beforehand by the
LoRA Catalog tool and corrected in its card editor.

    Wildcard Composer --sections--> LoRA Suggester --suggestions--> Eric Krea2 Multi-LoRA Stack (suggestions input)
                                                   --report------> a text display

mode suggest (default): the report shows what it would pick and why; nothing is sent to the loader.
mode apply: the picks go out on `suggestions`.

Without Eric_Krea2: connect `lora_text` (<lora:name:weight> tags) to any loader that reads that format.

Settings that are not on the node (settings.json, "suggest" section): platform shelves, blend ratio and
block_real_person_nsfw (default on). The catalog, vocab.yaml and axes.yaml are the LoRA Catalog's own.
See suggest/settings.py.

Moved here from video_prompter (2026-10-11). Same node id and the same inputs and outputs in the same order, so saved
workflows load unchanged.

Author: Eric Hiss (GitHub: EricRollei)
License: Dual License (Non-Commercial and Commercial Use). See LICENSE.
"""

import os
import traceback

from ..suggest import NSFW_LEVELS, SuggestRequest, load_settings, suggest
from ..suggest.settings import PLATFORMS

DEFAULT_PLATFORM = "krea2"
GENERIC_PLATFORM = "generic"
# Every platform id the node offered while it lived in video_prompter (its prompt profiles, 2026-10-11). A saved
# workflow keeps its platform as text, and ComfyUI refuses a value missing from the list, so none is ever removed.
# Ids the Suggester has no LoRA shelf for simply find no LoRAs (add them under settings.json "suggest" -> "platforms").
LEGACY_PLATFORM_IDS = [
    "anima", "aura_flow", "chroma", "ernie_image", "flux", "flux2_dev", "flux2_json", "flux2_klein", "flux2_pro",
    "gpt_image", "grok_imagine_image", "hunyuan_image", "hunyuan_image_3", "ideogram3", "ideogram4", "illustrious",
    "kling_image3", "kolors", "krea2", "leonardo_phoenix", "luma_uni1", "midjourney_v8", "minimax_image_01",
    "nano_banana", "noobai", "pixart_sigma", "pony", "pony_v7", "qwen_image", "qwen_image_21", "recraft_v4",
    "runway_gen4_image", "sd_1_5", "sd_xl", "seedream5", "wan_image", "wan22", "z_image", "z_image_turbo",
    "alice_t2v", "gemini_omni_flash", "grok_imagine_video", "hailuo_23", "hunyuan_video_15", "kling3", "ltx2", "ltx25",
    "luma_ray32", "minimax_h3", "minimax_h3_api", "pixverse_v6", "runway_aleph2", "runway_gen45", "seedance2",
    "seedance25", "veo31", "vidu_q3", "wan_api_audio", "wan22_video", "wan3_api",
]


def platform_choices():
    """generic, the platforms with LoRA shelves (settings included), then the rest of the old list."""
    try:
        known = list(load_settings().platforms)
    except Exception:          # a broken settings.json must not stop ComfyUI from listing the node
        known = list(PLATFORMS)
    return list(dict.fromkeys([GENERIC_PLATFORM] + known + LEGACY_PLATFORM_IDS))


class _Sections:
    """The parts of video_prompter's PROMPT_SECTIONS bundle the Suggester reads (sections, nsfw), from the bundle
    object, a plain dict of section -> text, or nothing. Read by attribute, so no import of video_prompter."""

    def __init__(self, value):
        if isinstance(value, dict):
            self.sections = {str(k): str(v) for k, v in value.items() if v}
            self.nsfw = None
        else:
            raw = getattr(value, "sections", None)
            self.sections = {str(k): str(v) for k, v in raw.items() if v} if isinstance(raw, dict) else {}
            self.nsfw = getattr(value, "nsfw", None)

# style 2 (2026-10-08): a second style is only added when it is from the same family (a blend); see engine.py.
ROLE_DEFAULTS = [("style", 2), ("treatment", 2), ("detailer", 1), ("environment", 1), ("lighting", 1),
                 ("subject", 1), ("layout", 1), ("utility", 1)]
ROLE_HELP = {
    "style": "base look: artist, movement, illustration or photo style",
    "transform": "changes content and composition but stacks on a style: surreal, dreamlike, glitch",
    "treatment": "layered on a style: linework, halftone, paper, grain, colour grade, material looks",
    "detailer": "detail, sharpness, skin, hands, anatomy",
    "environment": "location or background",
    "lighting": "light and colour mood",
    "subject": "character, person, creature, object, clothing, pose",
    "layout": "composition, page layout, framing",
    "utility": "speed LoRAs, nudity enablers - only when clearly asked for",
    "slider": "dials (age, body, shot size, saturation ...), set from what the prompt asks for on that axis",
}


def merge_triggers(prompt: str, triggers):
    """The prompt with each trigger appended once (a trigger already in the prompt is not repeated)."""
    prompt = (prompt or "").strip()
    lowered = prompt.lower()
    extra = [t for t in triggers if t and t.lower() not in lowered]
    if not extra:
        return prompt
    return f"{prompt}, {', '.join(extra)}" if prompt else ", ".join(extra)


class LoRASuggester:
    CATEGORY = "Eric Prompt Enhancers"
    FUNCTION = "run"
    # New outputs and inputs are only ever APPENDED, so saved workflows keep their links and widget values.
    RETURN_TYPES = ("LORA_SUGGESTIONS", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("suggestions", "triggers", "prompt_with_triggers", "report", "lora_text")
    OUTPUT_TOOLTIPS = (
        "The picked LoRAs with stage weights. Connect to the Eric Krea2 Multi-LoRA Stack's `suggestions` input. "
        "Empty in suggest mode.",
        "Trigger words of the picks, comma separated.",
        "The prompt with the triggers added (for workflows without a rewriter; with a rewriter, let the "
        "Multi-LoRA's add_triggers merge them after the rewrite instead).",
        "What was picked and why, the runner-up per role, and what was excluded.",
        "The picks as <lora:name:weight> tags, one weight per LoRA (the strongest of its three stage weights), for "
        "loaders outside Eric_Krea2 that read this format (e.g. LoRA Manager's text loader). Empty in suggest mode.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        choices = platform_choices()
        required = {
            "platform": (choices, {"default": DEFAULT_PLATFORM if DEFAULT_PLATFORM in choices else choices[0],
                                   "tooltip": "Target model: only LoRAs for it are considered (its LoRA shelf folder / Civitai base model)."}),
            "mode": (["suggest", "apply"], {"default": "suggest",
                     "tooltip": "suggest: report only, nothing loads. apply: send the picks to the loader."}),
            "nsfw": (NSFW_LEVELS, {"default": "off",
                     "tooltip": "Highest NSFW level a LoRA may have. NSFW LoRAs are used only when the prompt itself asks for "
                                "that content. off still allows suggestive styles, and styles that can paint nudes (turned down a little)."}),
            "max_loras": ("INT", {"default": 4, "min": 0, "max": 10, "tooltip": "Most LoRAs in the stack, all roles together."}),
        }
        for role, default in ROLE_DEFAULTS:
            required[f"{role}_max"] = ("INT", {"default": default, "min": 0, "max": 4,
                                               "tooltip": f"Most {role} LoRAs ({ROLE_HELP[role]}). 0 = never."})
        required.update({
            "apply_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                "tooltip": "How good a match must be before a LoRA is used. Lower = adventurous stacks, higher = only clear matches."}),
            "weight_budget": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1,
                              "tooltip": "Most total Stage-1 weight. The strongest pick keeps its weight, the others are scaled down to fit. 0 = no limit."}),
            "stage_profile": (["role", "flat"], {"default": "role",
                              "tooltip": "role: each role gets its own S1/S2/S3 shape (detailers late, environments early). flat: the same weight in every stage."}),
            "variety": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "0 = always the best match. Higher = with a new seed, close runners-up can win instead."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "control_after_generate": True,
                     "tooltip": "Seed for variety. Connect the Composer's seed to re-roll LoRAs with the prompt."}),
        })
        optional = {
            "sections": ("PROMPT_SECTIONS", {"tooltip": "From a Wildcard Composer: each section feeds its roles (style -> style and treatment, setting -> environment, ...)."}),
            "text": ("STRING", {"default": "", "forceInput": True, "tooltip": "Plain prompt text, when there are no sections (or in addition to them)."}),
            "exclude": ("STRING", {"default": "", "multiline": True,
                        "tooltip": "Never use these LoRAs. One entry per line or comma separated; case does not matter.\n"
                                   "Whole words only: 'age' matches 'Age Slider' but not 'image' or 'vintage'; "
                                   "'hair' does not match 'chair'. Several words must appear together: 'waist slider'.\n"
                                   "  plain words - the LoRA's name or file name (vallejo, hair, waist slider)\n"
                                   "  folder:<words> - its folder below the LoRA root (folder:sliders, folder:art-style)\n"
                                   "  role:<role> - a Suggester role (role:treatment, role:detailer, role:slider)\n"
                                   "  axis:<axis> - sliders on that axes.yaml axis (axis:age, axis:body weight)\n"
                                   "The report shows how many LoRAs each entry excluded."}),
            "pin": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Always use these LoRAs, whatever their score (they skip the threshold, role limits and "
                               "NSFW gate, and count toward max_loras). One entry per line or comma separated; whole words.\n"
                               "  a name - ONE LoRA: the best match (your stars, favourite, card confidence). If the words "
                               "match several, the report lists the others - add more of the name to pick a different one.\n"
                               "  name=weight - that LoRA at this weight (all stages; a slider keeps its stage shape): 'waist slider=-8', 'vallejo=0.7'\n"
                               "  axis:<axis>=weight - the best slider on that axis at this weight: 'axis:age=+3'\n"
                               "Give sliders a weight; without one they run at their card's typical weight. "
                               "folder: and role: work in exclude only."}),
            # Appended 2026-10-02 - new inputs only ever go at the end.
            "transform_max": ("INT", {"default": 1, "min": 0, "max": 4,
                              "tooltip": f"Most transform LoRAs ({ROLE_HELP['transform']}). 0 = never."}),
            # Appended 2026-10-08 - sliders.
            "slider_max": ("INT", {"default": 2, "min": 0, "max": 6,
                           "tooltip": f"Most slider LoRAs ({ROLE_HELP['slider']}). They do not count toward max_loras "
                                      "or the weight budget. 0 = never."}),
            "slider_range": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 15.0, "step": 0.1,
                             "tooltip": "For a slider whose card has no believable range: it dials within -range..+range "
                                        "(a strongest-cue prompt gets the full range). Ranges set in the card editor, or found "
                                        "by the slider card pass, are used instead."}),
            "slider_min_strength": ("INT", {"default": 2, "min": 1, "max": 3,
                                    "tooltip": "How clearly the prompt must ask before a slider is used, by the cue's position on "
                                               "its axis (axes.yaml ladder, -1..1). 1 = any cue; 2 = 0.35 or more (everyday words "
                                               "stay off); 3 = 0.65 or more, only strong asks ('wasp waist', 'ancient')."}),
        }
        return {"required": required, "optional": optional}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Re-run when the catalog changes (new cards, your edits), not only when the inputs do."""
        try:
            catalog = load_settings().catalog
            parts = [os.path.getmtime(catalog)]
            if os.path.exists(catalog + "-wal"):
                parts.append(os.path.getmtime(catalog + "-wal"))
            return "|".join(str(p) for p in parts)
        except OSError:
            return ""

    def run(self, platform, mode, nsfw, max_loras, style_max, treatment_max, detailer_max, environment_max,
            lighting_max, subject_max, layout_max, utility_max, apply_threshold, weight_budget, stage_profile,
            variety, seed, sections=None, text="", exclude="", pin="", transform_max=1, slider_max=2,
            slider_range=3.0, slider_min_strength=2):
        bundle = _Sections(sections)
        request = SuggestRequest(
            platform=platform, sections=dict(bundle.sections), text=text or "", max_loras=int(max_loras),
            role_max={"style": style_max, "transform": transform_max, "treatment": treatment_max, "detailer": detailer_max,
                      "environment": environment_max, "lighting": lighting_max, "subject": subject_max,
                      "layout": layout_max, "utility": utility_max, "slider": slider_max},
            threshold=float(apply_threshold), weight_budget=float(weight_budget), stage_profile=stage_profile,
            variety=float(variety), seed=int(seed), nsfw=nsfw, exclude=exclude or "", pin=pin or "",
            prompt_nsfw=bundle.nsfw if sections is not None else None,
            slider_range=float(slider_range), slider_min_strength=int(slider_min_strength))
        try:
            result = suggest(request)
        except Exception as error:      # never break the graph over a suggestion
            traceback.print_exc()
            return ([], "", text or "", f"LoRA Suggester error: {error}", "")
        entries = [pick.entry() for pick in result.picks]
        header = "APPLY: sent to the loader" if mode == "apply" else "SUGGEST: report only (set mode to apply to load these)"
        report = f"{header}\n{result.report}"
        triggers = ", ".join(result.triggers)
        print(f"[LoRA Suggester] {mode}: " + (", ".join(f"{e['role']}={e['name']}" for e in entries) or "no picks"))
        return (entries if mode == "apply" else [], triggers, merge_triggers(result.prompt, result.triggers), report,
                lora_text(entries) if mode == "apply" else "")


def lora_text(entries) -> str:
    """<lora:name:weight> per pick for single-stage loaders: the file name without extension, and the stage weight with
    the largest size (keeps a slider's sign; a detailer's S1 weight is 0, its S3 one is the real one)."""
    tags = []
    for entry in entries:
        name = os.path.splitext(os.path.basename(str(entry.get("lora_name") or entry.get("path") or "")))[0]
        weights = [float(entry.get(k) or 0.0) for k in ("weight_s1", "weight_s2", "weight_s3")]
        weight = max(weights, key=abs)
        if name and weight:
            tags.append(f"<lora:{name}:{round(weight, 2):g}>")
    return " ".join(tags)
