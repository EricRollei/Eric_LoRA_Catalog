"""Checks for 'LoRAs named in the style image' (2026-10-13): the metadata reader on real files, the graph walk from
the Advanced node's style_image to the Load Image file, and the engine evidence (USED_RAW) with its report.
    python dev/verify_image_loras.py"""
import glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
PACK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(PACK))
import importlib
pkg = os.path.basename(PACK)
image_meta = importlib.import_module(f"{pkg}.lora_catalog.image_meta")
config = importlib.import_module(f"{pkg}.lora_catalog.config")
engine = importlib.import_module(f"{pkg}.suggest.engine")
node_mod = importlib.import_module(f"{pkg}.nodes.lora_suggest_advanced_node")

fails = 0
def check(ok, what, detail=""):
    global fails
    fails += 0 if ok else 1
    print(("ok   " if ok else "FAIL ") + what + (f"  - {detail}" if detail else ""))

settings = config.load_settings()
OUT = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\output"
ultra = sorted(glob.glob(OUT + r"\Krea2\Krea2-Ultra-v2_*.png"), key=os.path.getmtime)[-1]
cell = sorted(glob.glob(OUT + r"\sweeps\20261013_001804*\cell_01*.png"))[0]
civ = glob.glob(r"L:\Models\loras\Krea2\art-style\1148649-Style_of_Santiago_Caruso\**\images\145046753.jpeg", recursive=True)[0]

found = image_meta.match(settings.catalog_path, image_meta.read_file(ultra))
check(any(f["owned"] and "Parkes" in f["name"] for f in found), "ComfyUI PNG (own Ultra render): its LoRA stack",
      ", ".join(f"{f['name']} @{f['weight']}" for f in found))
found = image_meta.match(settings.catalog_path, image_meta.read_file(cell))
check(len(found) == 1 and found[0]["owned"] and found[0]["weight"] == 0.6, "sweep cell PNG: its swept LoRA",
      ", ".join(f"{f['name']} @{f['weight']}" for f in found))
found = image_meta.match(settings.catalog_path, image_meta.read_file(civ))
check(any(f["owned"] and "Caruso" in f["name"] and f["how"] == "hash" for f in found)
      and any(not f["owned"] for f in found), "Civitai JPEG (EXIF): owned by hash + not-owned listed",
      ", ".join(f"{f['name']}{'' if f['owned'] else ' (not yours)'}" for f in found))
with open(civ, "rb") as h:
    check(len(image_meta.read_bytes(h.read())) == len(image_meta.read_file(civ)), "read_bytes == read_file (editor upload)")

# graph walk: a fake API graph  LoadImage(abs path) -> ImageBatch -> Advanced.style_image
prompt = {"5": {"class_type": "LoadImage", "inputs": {"image": ultra, "upload": "image"}},
          "6": {"class_type": "LoadImage", "inputs": {"image": civ}},
          "7": {"class_type": "ImageBatch", "inputs": {"image1": ["5", 0], "image2": ["6", 0]}},
          "9": {"class_type": "EricLoRASuggesterAdvanced", "inputs": {"style_image": ["7", 0], "text": "x"}}}
files = node_mod._image_files(prompt, "9")
check(set(files) == {ultra, civ}, "graph walk finds both Load Image files through a batch node", str([os.path.basename(f) for f in files]))
notes, lines = node_mod._image_loras(prompt, "9")
check(len(notes) >= 2 and any("Caruso" in l for l in lines), "owned LoRAs from both images become evidence",
      f"{len(notes)} owned; report {len(lines)} lines")

# engine: the image's style LoRA wins its slot on a prompt that does not ask for it; the plain request is unchanged
text = "a lighthouse on a rocky coast at dusk"
plain = engine.suggest(engine.SuggestRequest(platform="krea2", text=text, variety=0))
used = engine.suggest(engine.SuggestRequest(platform="krea2", text=text, variety=0, image_loras=notes))
names_plain = [p.card.name for p in plain.picks]
names_used = [p.card.name for p in used.picks]
check(any("Caruso" in n or "Parkes" in n for n in names_used), "image LoRAs picked as evidence", str(names_used))
check(not any("Caruso" in n or "Parkes" in n for n in names_plain), "without image_loras they are not picked", str(names_plain))
reasons = [r for p in used.picks for r in p.reasons if r.startswith("used in the style image")]
check(bool(reasons), "the reason names the image", reasons[0] if reasons else "")
utility = [p.card.name for p in used.picks if p.role == "utility" and p.card.path in notes]
check(not utility, "utility helpers (turbo / filter LoRAs) are not boosted", str(utility))
print("\nALL OK" if not fails else f"\n{fails} FAILED")
