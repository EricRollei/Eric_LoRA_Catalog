"""
Settings for the LoRA Suggester. The catalog, vocab.yaml and axes.yaml are the LoRA Catalog's own (same settings.json,
same data folder), so the Suggester always reads what the catalog wrote. The "suggest" section of settings.json holds
the Suggester's own options (all optional):

    "suggest": {
        "catalog": "...", "vocab": "...", "axes": "..."        only to point the Suggester somewhere else
        "blend_ratio": 0.6,                                    a second style of the same family runs at this x its weight
        "family_friends": [["comic_illustration", "print_graphic"], ["comic_illustration", "anime"]],
                                                               style families that may blend with each other
        "block_real_person_nsfw": true,                        a real-person likeness LoRA is never stacked with a
                                                               nudity/explicit LoRA (set false to allow)
        "platforms": {"krea2": {"shelves": ["Krea2"], "base_models": ["Krea 2"]}}   add or override platforms
    }


(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from ..lora_catalog import config as catalog_config        # inside the ComfyUI pack (Eric_LoRA_Catalog.suggest)
except ImportError:
    from lora_catalog import config as catalog_config          # standalone (run.py / tests put the repo on sys.path)

DEFAULT_FAMILY_FRIENDS = [["comic_illustration", "print_graphic"], ["comic_illustration", "anime"]]

# platform id (profiles/image/*.yaml) -> (shelf folders under the LoRA root, Civitai base models).
# A LoRA belongs to a platform when its shelf folder is listed; a LoRA on an unknown shelf belongs when its Civitai
# base model is listed.
PLATFORMS: Dict[str, Tuple[List[str], List[str]]] = {
    "krea2": (["Krea2"], ["Krea 2"]),
    "qwen_image": (["Qwen"], ["Qwen"]),
    "qwen_image_21": (["Qwen21"], ["Qwen 2.1"]),
    "flux": (["Flux"], ["Flux.1 D", "Flux.1 Krea"]),
    "flux2_dev": (["Flux2"], ["Flux.2 D"]),
    "flux2_pro": (["Flux2"], ["Flux.2 D"]),
    "flux2_json": (["Flux2"], ["Flux.2 D"]),
    "flux2_klein": (["Flux2klein"], ["Flux.2 Klein 9B", "Flux.2 Klein 9B-base"]),
    "sd_xl": (["SDXL"], ["SDXL 1.0", "SDXL 0.9", "SDXL Hyper"]),
    "pony": (["Pony"], ["Pony"]),
    "pony_v7": (["PonyV7"], ["Pony V7"]),
    "illustrious": (["illustrious"], ["Illustrious"]),
    "noobai": (["NoobAI"], ["NoobAI"]),
    "sd_1_5": (["SD15"], ["SD 1.5", "SD 1.5 LCM"]),
    "z_image": (["Z_Image"], ["ZImageTurbo", "ZImage"]),
    "z_image_turbo": (["Z_Image"], ["ZImageTurbo"]),
    "chroma": (["Chroma"], ["Chroma"]),
    "wan_image": (["Wan"], ["Wan Video 2.2 T2V-A14B", "Wan Video 14B t2v"]),
    "wan22": (["Wan"], ["Wan Video 2.2 T2V-A14B", "Wan Video 2.2 I2V-A14B"]),
    "ideogram4": (["Ideogram4"], ["Ideogram 4.0"]),
    "ernie_image": (["Ernie"], []),
    "anima": (["Anima"], ["Anima"]),
    "hunyuan_image": (["HunyuanImage"], ["Hunyuan Image"]),
}


@dataclass
class SuggestSettings:
    catalog: str = ""
    vocab: str = ""
    block_real_person_nsfw: bool = True
    axes: str = ""
    blend_ratio: float = 0.6
    family_friends: List[List[str]] = field(default_factory=lambda: [list(p) for p in DEFAULT_FAMILY_FRIENDS])
    platforms: Dict[str, Tuple[List[str], List[str]]] = field(default_factory=lambda: dict(PLATFORMS))

    def platform_filter(self, platform: str) -> Tuple[List[str], List[str]]:
        return self.platforms.get(platform, ([], []))

    def known_shelves(self) -> set:
        return {shelf.casefold() for shelves, _bases in self.platforms.values() for shelf in shelves}


def load_settings() -> SuggestSettings:
    catalog = catalog_config.load_settings()
    settings = SuggestSettings(catalog=catalog.catalog_path, vocab=catalog_config.user_file("vocab.yaml"),
                               axes=catalog_config.user_file("axes.yaml"))
    section = catalog.suggest if isinstance(catalog.suggest, dict) else {}
    if section.get("catalog"):
        settings.catalog = str(section["catalog"])
    if section.get("vocab"):
        settings.vocab = str(section["vocab"])
    if section.get("axes"):
        settings.axes = str(section["axes"])
    if "blend_ratio" in section:
        settings.blend_ratio = float(section["blend_ratio"])
    if isinstance(section.get("family_friends"), list):
        settings.family_friends = [[str(f) for f in pair] for pair in section["family_friends"] if isinstance(pair, list)]
    if "block_real_person_nsfw" in section:
        settings.block_real_person_nsfw = bool(section["block_real_person_nsfw"])
    for platform, spec in (section.get("platforms") or {}).items():
        if isinstance(spec, dict):
            settings.platforms[str(platform)] = ([str(s) for s in spec.get("shelves") or []],
                                                 [str(b) for b in spec.get("base_models") or []])
    return settings
