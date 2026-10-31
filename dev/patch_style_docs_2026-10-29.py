# README: "Search by look" section; USER_GUIDE: fuller image-search, similar-look and look-alike notes (2026-10-29).
import os
import shutil
import sys

C = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_LoRA_Catalog"
TAG = "_pre_styledocs_2026-10-29"

README_SEC = """## Search by look: style vectors

Words describe what a LoRA is *about*; they are poor at what it *looks like*. So the catalog can also describe each
LoRA by its look - optional, off until you set it up (below).

**How it works.** A style model, **CSD** (Contrastive Style Descriptors, Somepalli et al., ECCV 2024 - a CLIP ViT-L/14
fine-tuned so that images in the same style land close together *whatever they show*), turns every image into 768
numbers: its style vector. Each LoRA gets one from its Civitai preview images (up to 12, averaged). Two vectors are
compared by cosine: 1 = the same look, about 0.1 = unrelated. A LoRA's previews usually agree with each other at about
0.5 and with other LoRAs' previews at about 0.1, which is why no test rendering is needed. Strong on look LoRAs
(style, medium, photo look, lighting): two Mignola LoRAs match at 0.87, Frazetta sits with Vallejo and Brom. Weak on
photoreal LoRAs, which all look like "a realistic photo" to it - so look features only use look-type cards.

**Search by image.** Click **By image** in the editor, or drop or paste an image onto the search box:
1. the image becomes a style vector (the first search loads the model, about 15 s);
2. the list fills with your LoRAs closest in look, up to 200 - inside the current filters, so pick the base model
   first (Krea2 - all folders = Krea2 LoRAs only);
3. if the image carries generation metadata (ComfyUI, A1111 / Forge, Civitai), a **made with** line names the LoRAs
   it used - yours open in the editor, others link to Civitai. A pasted screenshot has no metadata; save or download
   the file instead.
It matches **style, not subject**: a watercolour fox finds watercolour LoRAs, not fox LoRAs.

**Also uses it:** the **Similar look** strip on every card, **Look-alikes** (near-copies of one look, for culling),
the LoRA Suggester's `style_image` input, and render checks of sweep cells. Setup and details:
[user guide](docs/USER_GUIDE.md#style-vectors-csd-search-by-look).

    python run.py style-setup            download the CSD weights (2.4 GB, kept as 1.2 GB)
    settings.json: "style_enabled": true
    python run.py style                  vectors for every LoRA (GPU ~80 images/s, once; then only new previews)

"""

GUIDE_OLD = """- **Editor - image search:** pick, drop or paste an image onto the search box; the list becomes the LoRAs whose look
  is closest to it (the first search loads the model, about 15 s)."""
GUIDE_NEW = """- **Editor - image search:** click **By image**, or drop or paste an image onto the search box. The image is embedded
  with the same model and the list becomes your LoRAs ordered by closeness of look, at most 200, inside the current
  filters (base-model shelf, status, rating) - choose the shelf first. The first search loads the model (about 15 s);
  it unloads again when idle. Tips:
  - It matches style, not subject. For "LoRAs like this one", use its card's Similar look strip instead.
  - **made with:** when the image carries generation metadata (ComfyUI workflow, A1111 / Forge parameters, Civitai),
    a line above the list names the LoRAs it was made with and their weights - yours open in the editor, the others
    link to their Civitai page. Screenshots and most social-media copies have no metadata.
  - A new text search (or the search box's clear button) ends the image search.
- **Editor - Similar look strip** (under each card's images): the closest looks to this LoRA. *previews agree*
  (high / mixed / low) says how consistent its own previews are - low means its signature is a blur of different
  looks and the matches are less reliable. Toggles: **all bases** (off = same base model only), **all kinds** (off =
  style, photo-look and lighting LoRAs only), **NSFW**, and **list all** (the whole ranked list on the left). A
  render-check verdict shows here too (shows its look / needs its style words / weak).
- **Editor - Look-alikes:** pairs of style LoRAs on the current shelf whose previews look nearly the same, side by
  side - for culling near-copies of one look."""


def main():
    rp = os.path.join(C, "README.md")
    gp = os.path.join(C, "docs", "USER_GUIDE.md")
    r = open(rp, encoding="utf-8", newline="").read()
    g = open(gp, encoding="utf-8", newline="").read()
    rnl = "\r\n" if "\r\n" in r else "\n"
    gnl = "\r\n" if "\r\n" in g else "\n"
    anchor = "## Your sweep ratings: scores, stacks and partners"
    if r.count(anchor) != 1 or "## Search by look" in r:
        print("README anchor problem or section exists"); return 1
    r = r.replace(anchor, README_SEC.replace("\n", rnl) + anchor)
    o = GUIDE_OLD.replace("\n", gnl)
    if g.count(o) != 1:
        print("GUIDE anchor count", g.count(o)); return 1
    g = g.replace(o, GUIDE_NEW.replace("\n", gnl))
    for path in (rp, gp):
        base, ext = os.path.splitext(os.path.basename(path))
        bk = os.path.join(C, "docs", "backup", f"{base}{TAG}{ext}")
        if not os.path.exists(bk):
            shutil.copy2(path, bk)
    open(rp, "w", encoding="utf-8", newline="").write(r)
    open(gp, "w", encoding="utf-8", newline="").write(g)
    print("README + USER_GUIDE updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
