"""
Checks for the LoRA Suggester (Advanced) node, 2026-10-12 (needs style search set up; run with ComfyUI's Python).

    python dev/verify_suggest_advanced.py

1. node layout: its own widgets first, then every input of the plain Suggester in the same order, style_image last
2. image only (empty prompt): a LoRA's own preview as style_image picks that LoRA (or a near look) as the style
3. weight 0: the image changes nothing in the picks (report only)
4. image + prompt words: both show in the reasons
5. two different looks, each image (best match): both looks reach the top of the report's closest list
6. no image: identical to the plain Suggester
"""

from __future__ import annotations

import importlib
import os
import sys

CUSTOM_NODES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CUSTOM_NODES)
pack = importlib.import_module("Eric_LoRA_Catalog")
adv_mod = importlib.import_module("Eric_LoRA_Catalog.nodes.lora_suggest_advanced_node")
plain_mod = importlib.import_module("Eric_LoRA_Catalog.nodes.lora_suggest_node")
style = importlib.import_module("Eric_LoRA_Catalog.lora_catalog.style")
cfg = importlib.import_module("Eric_LoRA_Catalog.lora_catalog.config")
cards_mod = importlib.import_module("Eric_LoRA_Catalog.lora_catalog.cards")

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{('  - ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def image_tensor(paths):
    import numpy as np
    import torch
    from PIL import Image
    arrays = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((512, 512))
        arrays.append(np.asarray(im, dtype="float32") / 255.0)
    return torch.from_numpy(np.stack(arrays))


def picks(result):
    report = result[3]
    return [line for line in report.splitlines() if line.startswith("+ ")], report


def main():
    plain_inputs = plain_mod.LoRASuggester.INPUT_TYPES()
    adv_inputs = adv_mod.LoRASuggesterAdvanced.INPUT_TYPES()
    req = list(adv_inputs["required"])
    check("Advanced widgets first", req[:2] == ["style_image_weight", "image_mode"], str(req[:3]))
    check("plain Suggester inputs follow in order", req[2:] == list(plain_inputs["required"]) and
          list(adv_inputs["optional"])[:-1] == list(plain_inputs["optional"]))
    check("style_image last", list(adv_inputs["optional"])[-1] == "style_image")
    check("registered", "EricLoRASuggesterAdvanced" in pack.NODE_CLASS_MAPPINGS)

    settings = cfg.load_settings()
    if style.available(settings):
        print("style search not available:", style.available(settings))
        return 1
    index = style.StyleIndex(settings)
    index.ensure()
    krea = {i["name"]: i for i in index.items if i["arch"] == "Krea2" and i["kind"] == "style_medium"}
    node = adv_mod.LoRASuggesterAdvanced()
    plain = plain_mod.LoRASuggester()

    def preview(name, k=0):
        return cards_mod.preview_images(krea[name]["path"], 12)[k]

    targets = [n for n in ("Mike Mignola Style", "@Gustav Klimt", "Araminta Frazetta Style [Krea2]", "Anton Semenov - ArtStyle (KREA2)")
               if n in krea]
    import sqlite3
    con = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True)
    nsfw_of = {n: (con.execute("select nsfw from card_facets where path = ?", (krea[n]["path"],)).fetchone() or [0])[0]
               for n in targets}
    con.close()
    ranked = picked = expected = 0
    for name in targets:
        # variety 0: with the default 0.2, picks within ~0.1 trade places by seed (by design) - here we test the image
        result = node.run_advanced(style_image=image_tensor([preview(name, 1)]), platform="krea2", text="", seed=1,
                                   variety=0.0)
        lines, report = picks(result)
        style_lines = [l for l in lines if l.startswith("+ style")]
        closest = next((l for l in report.splitlines() if l.startswith("style image:")), "")
        first = closest.split("closest: ", 1)[-1]
        ranked += first.startswith(name[:20])
        own = any(name[:20] in l for l in style_lines)
        # a nudity-rated (2) style card that clearly matches the image is no longer turned down (Eric 2026-10-12),
        # so every target below explicit (3) should be picked from its own image
        if (nsfw_of[name] or 0) < 3:
            expected += 1
            picked += own
        print(f"     {name} (card nsfw {nsfw_of[name]}): style picks {[l.split(']')[0] for l in style_lines]}")
        print(f"       {closest[:220]}")
    check(f"image only: own LoRA is the closest look {ranked}/{len(targets)}", ranked == len(targets))
    check(f"image only: own LoRA picked as style (cards below explicit) {picked}/{expected}", picked == expected)

    name = targets[0]
    img = image_tensor([preview(name, 1)])
    base = plain.run(platform="krea2", mode="suggest", nsfw="off", max_loras=4, style_max=2, treatment_max=2, detailer_max=1,
                     environment_max=1, lighting_max=1, subject_max=1, layout_max=1, utility_max=1, apply_threshold=0.35,
                     weight_budget=2.5, stage_profile="role", variety=0.2, seed=3,
                     text="a knight on horseback in a misty forest, dramatic")
    zero = node.run_advanced(style_image_weight=0.0, style_image=img, platform="krea2", seed=3,
                             text="a knight on horseback in a misty forest, dramatic")
    check("weight 0: same picks as the plain node", picks(base)[0] == [l for l in picks(zero)[0]],
          f"{len(picks(base)[0])} picks")
    mixed = node.run_advanced(style_image=img, platform="krea2", seed=3,
                              text="a knight on horseback in a misty forest, dramatic, comic book ink")
    lines, report = picks(mixed)
    reasons = [l for l in report.splitlines() if "look " in l and "style image" in l and not l.startswith("style image:")]
    check("image + prompt: look reasons in the report", bool(reasons), (reasons[0].strip()[:150] if reasons else report[:300]))
    none = node.run_advanced(platform="krea2", seed=3, text="a knight on horseback in a misty forest, dramatic")
    check("no image: same picks as the plain node", picks(none)[0] == picks(base)[0])

    if len(targets) >= 2:
        a, b = targets[0], targets[1]
        two = image_tensor([preview(a, 2), preview(b, 2)])
        for mode in adv_mod.IMAGE_MODES:
            result = node.run_advanced(style_image=two, image_mode=mode, platform="krea2", text="", seed=1)
            closest = next((l for l in result[3].splitlines() if l.startswith("style image:")), "")
            found = [n for n in (a, b) if n[:18] in closest]
            print(f"     {mode}: {closest[:230]}")
            if mode.startswith("each"):
                check("two looks, best match: both in the closest 5", len(found) == 2, f"found {found}")
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
