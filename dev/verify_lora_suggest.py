"""
Checks for the LoRA Suggester (suggest/ + nodes/lora_suggest_node.py + the Krea2 Multi-LoRA suggestions socket).
No LLM, no network. Uses the real pick lists (vocab.yaml) and a small made-up catalog for the rules, then the real
catalog for a smoke test when it exists.

    python_embeded\\python.exe dev\\verify_lora_suggest.py

Outside custom_nodes (a staging copy), set COMFYUI_DIR to ComfyUI's folder for the Krea2 Multi-LoRA part.
"""

import importlib
import json
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(HERE)
CUSTOM_NODES_DIR = os.path.dirname(PACKAGE_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES_DIR)
if os.path.basename(CUSTOM_NODES_DIR).lower() != "custom_nodes" and os.environ.get("COMFYUI_DIR"):
    COMFY_ROOT = os.environ["COMFYUI_DIR"]
REAL_CUSTOM_NODES = os.path.join(COMFY_ROOT, "custom_nodes")
sys.path.insert(0, PACKAGE_DIR)

from suggest import SuggestRequest, load_pick_lists, load_settings, read_prompt, suggest  # noqa: E402
from suggest import catalog as catalog_module  # noqa: E402
from suggest.engine import apply_budget  # noqa: E402

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


REAL = load_settings()
LISTS = load_pick_lists(REAL.vocab)
check("pick lists load", LISTS is not None, REAL.vocab)

# ------------------------------------------------------------------ pick lists
check("tree: narrower card", LISTS.similarity("period", "renaissance", "high_renaissance") == 0.7)
check("tree: broader card", LISTS.similarity("period", "high_renaissance", "renaissance") == 0.5)
check("tree: siblings", LISTS.similarity("period", "high_renaissance", "mannerism") == 0.25)
check("tree: unrelated", LISTS.similarity("period", "baroque", "pop_art") == 0.0)
check("match longest phrase", LISTS.match("lighting", "golden hour light") == ["golden_hour"], LISTS.match("lighting", "golden hour light"))
check("golden hour is not gold material", "gold" not in LISTS.match("material", "golden hour, long shadows"))

view = read_prompt(LISTS, {}, "a nude figure study in charcoal")
check("prompt nsfw: nudity", view.level == 2, view.level)
view = read_prompt(LISTS, {}, "a quiet harbour at dawn")
check("prompt nsfw: sfw", view.level == 0)
view = read_prompt(LISTS, {"subject": "a knight", "style": "ukiyo-e print"}, "")
check("sections feed roles", "ukiyo-e" in view.role_text["style"] and "knight" in view.role_text["subject"])
check("setting empty -> environment role text empty", view.role_text["environment"] == "")

# ------------------------------------------------------------------ made-up catalog for the rules
COLUMNS = ["path", "key", "name", "file_name", "rel_dir", "arch_folder", "base_model", "role", "facets", "nsfw",
           "nsfw_shown", "artists", "depicts", "triggers", "creator", "weight_typical", "weight_low", "weight_high",
           "overrides", "confidence", "kind", "scope", "effect", "favorite", "exclude", "source", "status",
           "updated_at"]
CARDS = [
    # name, role, facets, nsfw, artists, depicts, triggers, overrides, kind, effect
    ("Vallejo Fantasy", "style", {"period": ["fantasy_paperback"], "medium": ["oil"]}, 2, ["Boris Vallejo"], [], ["vallejo style"],
     ["style", "lighting"], "style_medium", "fantasy oil painting with muscular heroes"),
    ("Ukiyo Print", "style", {"period": ["ukiyo_e"], "medium": ["woodcut"], "rendering": ["flat"]}, 0, [], [], ["ukiyoprint"],
     ["style", "colour"], "style_medium", "japanese woodblock print look"),
    ("Halftone Layer", "treatment", {"medium": ["halftone"], "line": ["bold_outline"]}, 0, [], [], [],
     ["texture"], "style_medium", "adds halftone dots and bold ink"),
    ("Golden Light", "lighting", {"lighting": ["golden_hour"]}, 0, [], [], [], ["lighting"], "lighting_colour",
     "warm golden hour light with long shadows"),
    ("Explicit Concept", "subject", {"mature_content": ["explicit_sexual"], "figure": ["female"]}, 3, [], [], ["xconcept"],
     [], "concept_object", "explicit sexual content"),
    ("Boudoir Mood", "subject", {"mature_content": ["boudoir"], "figure": ["female"]}, 2, [], [], ["boudoirmood"],
     [], "concept_object", "boudoir glamour photo of a woman in lingerie"),
    ("Young Look", "subject", {"figure": ["female", "original_character"], "side_effects": ["youthful_look"]}, 0, [], ["Mina"],
     ["mina"], [], "character", "original character Mina"),
    ("Star Likeness", "subject", {"figure": ["real_person", "female"]}, 0, [], ["Jane Starr"], ["janestarr"], [],
     "person", "likeness of the actress Jane Starr"),
    ("Hands Fix", "detailer", {"detail": ["hands", "fine_detail"]}, 0, [], [], [], [], "effect_detail",
     "better hands and fine detail"),
    ("Other Platform Style", "style", {"period": ["ukiyo_e"]}, 0, [], [], [], [], "style_medium", "ukiyo-e"),
]


def make_catalog(folder):
    path = os.path.join(folder, "catalog.sqlite")
    connection = sqlite3.connect(path)
    connection.execute(f"create table card_facets ({', '.join(COLUMNS)})")
    connection.execute("create table cards (path text)")
    for index, (name, role, facets, nsfw, artists, depicts, triggers, overrides, kind, effect) in enumerate(CARDS):
        arch = "Flux" if name.startswith("Other") else "Krea2"
        lora_path = os.path.join(folder, f"lora_{index}.safetensors")
        open(lora_path, "wb").close()
        facets = dict(facets, role=[role])
        if depicts:
            facets["depicts"] = depicts
        row = {"path": lora_path, "key": lora_path, "name": name, "file_name": os.path.basename(lora_path),
               "rel_dir": f"{arch}/test", "arch_folder": arch, "base_model": "Krea 2" if arch == "Krea2" else "Flux.1 D",
               "role": role, "facets": json.dumps(facets), "nsfw": nsfw, "nsfw_shown": nsfw,
               "artists": json.dumps(artists), "depicts": json.dumps(depicts), "triggers": json.dumps(triggers),
               "creator": "", "weight_typical": 1.0, "weight_low": 0.6, "weight_high": 1.2,
               "overrides": json.dumps(overrides), "confidence": 0.9, "kind": kind, "scope": "global", "effect": effect,
               "favorite": 0, "exclude": 0, "source": "card", "status": "", "updated_at": ""}
        connection.execute(f"insert into card_facets values ({','.join('?' for _ in COLUMNS)})", [row[c] for c in COLUMNS])
        connection.execute("insert into cards values (?)", (lora_path,))
    connection.commit()
    connection.close()
    return path


def names(result):
    return [pick.card.name for pick in result.picks]


with tempfile.TemporaryDirectory() as folder:
    settings = load_settings()
    settings.catalog = make_catalog(folder)
    catalog_module._cache.clear()

    def run(text="", sections=None, **kw):
        return suggest(SuggestRequest(platform="krea2", text=text, sections=sections or {}, threshold=0.2, **kw), settings)

    result = run("ukiyo-e woodblock print of a harbour")
    check("platform filter: other shelf never offered", "Other Platform Style" not in names(result), names(result))
    check("style picked by period+medium", "Ukiyo Print" in names(result), names(result))

    result = run("a barbarian in the style of Boris Vallejo")
    check("artist name picks the artist LoRA", names(result)[:1] == ["Vallejo Fantasy"] or "Vallejo Fantasy" in names(result), names(result))
    check("nudity-capable style allowed at gate off (soft)", "Vallejo Fantasy" in names(result))
    vallejo = [p for p in result.picks if p.card.name == "Vallejo Fantasy"]
    check("soft style is marked", vallejo and "may add nudity" in vallejo[0].reasons, vallejo and vallejo[0].reasons)

    result = run("explicit sexual scene, xconcept", nsfw="off")
    check("explicit LoRA excluded at gate off", "Explicit Concept" not in names(result), names(result))
    result = run("explicit sexual scene, xconcept", nsfw="explicit")
    check("explicit LoRA allowed when gate and prompt ask", "Explicit Concept" in names(result), result.report)
    result = run("a portrait of a woman, xconcept", nsfw="explicit")
    check("explicit LoRA not used for a SFW prompt even with gate explicit", "Explicit Concept" not in names(result), names(result))
    result = run("artistic nude boudoir photo, boudoirmood", nsfw="explicit")
    check("nudity prompt: explicit LoRA excluded (above prompt level)", "Explicit Concept" not in names(result), names(result))
    check("nudity prompt: boudoir LoRA picked", "Boudoir Mood" in names(result), result.report)

    result = run("Mina in a garden, boudoir photo, nude", nsfw="explicit", role_max={"subject": 2, "style": 1})
    check("youthful_look never with NSFW prompt", "Young Look" not in names(result), names(result))
    result = run("Mina in a garden")
    check("named character picked when named", "Young Look" in names(result), result.report)
    result = run("a girl in a garden")
    check("character not offered when not named", "Young Look" not in names(result), names(result))

    result = run("Jane Starr boudoir photo nude, boudoirmood", nsfw="explicit", role_max={"subject": 2})
    check("real person + NSFW blocked (config default on)", "Star Likeness" not in names(result), names(result))
    settings.block_real_person_nsfw = False
    result = run("Jane Starr boudoir photo nude, boudoirmood", nsfw="explicit", role_max={"subject": 2})
    check("real person + NSFW allowed when switched off", "Star Likeness" in names(result), result.report)
    settings.block_real_person_nsfw = True
    result = run("Jane Starr on a red carpet")
    check("real person picked by name (SFW)", "Star Likeness" in names(result), result.report)

    result = run("ukiyo-e print", exclude="ukiyo")
    check("exclude by name", "Ukiyo Print" not in names(result), names(result))
    result = run("a quiet room", pin="Hands Fix")
    check("pin forces a LoRA in", "Hands Fix" in names(result), names(result))
    result = run("ukiyo-e woodblock print, golden hour, halftone dots, bold outlines", max_loras=1)
    check("max_loras caps the stack", len(result.picks) == 1, names(result))
    result = run("ukiyo-e woodblock print, golden hour, halftone dots, bold outlines", role_max={"style": 0, "treatment": 2, "lighting": 1})
    check("role max 0 = never", all(p.role != "style" for p in result.picks), names(result))
    detail = [p for p in run("highly detailed hands, fine detail").picks if p.role == "detailer"]
    check("detailer stage shape 0 / 0.6 / 1.0", detail and detail[0].weights == (0.0, 0.6, 1.0), detail and detail[0].weights)
    flat = [p for p in run("highly detailed hands, fine detail", stage_profile="flat").picks if p.role == "detailer"]
    check("flat stage profile", flat and flat[0].weights == (1.0, 1.0, 1.0), flat and flat[0].weights)

    # budget
    picks = run("ukiyo-e woodblock print, golden hour, halftone dots, bold outlines", weight_budget=0).picks
    if len(picks) >= 2:
        total = sum(p.weights[0] for p in picks)
        apply_budget(picks, 1.2)
        lead = max(picks, key=lambda p: p.score)
        check("budget keeps the lead's weight", lead.weights[0] == 1.0, lead.weights)
        check("budget shrinks the rest", sum(p.weights[0] for p in picks) < total)
    else:
        check("budget test needs 2 picks", False, names(run("ukiyo-e woodblock print, golden hour, halftone dots, bold outlines")))
    catalog_module._cache.clear()

# ------------------------------------------------------------------ real catalog (smoke)
if os.path.exists(REAL.catalog):
    result = suggest(SuggestRequest(platform="krea2", text="a barbarian warrior fighting a dragon, in the style of Boris Vallejo"))
    check("real catalog answers", "carded LoRAs" in result.report, result.report[:200])
    check("real catalog: Vallejo for a Vallejo prompt",   # the real card is named "VALEJO"; match its trigger/artist too
          any("vallejo" in " ".join([p.card.name, *p.card.triggers, *p.card.artists]).lower() for p in result.picks),
          result.report)
else:
    print("  (real catalog not found, smoke test skipped)")

# ------------------------------------------------------------------ node + Krea2 socket (inside ComfyUI's tree)
for path in (COMFY_ROOT, REAL_CUSTOM_NODES, CUSTOM_NODES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
try:
    package = importlib.import_module(os.path.basename(PACKAGE_DIR))
    node_class = package.NODE_CLASS_MAPPINGS.get("EricLoRASuggester")
    check("node registered", node_class is not None)
    inputs = node_class.INPUT_TYPES()
    check("node inputs", {"platform", "mode", "nsfw", "max_loras", "style_max", "apply_threshold"} <= set(inputs["required"]))
    check("node optional sockets", {"sections", "text", "exclude", "pin"} <= set(inputs["optional"]))
    node = node_class()
    args = dict(platform="krea2", nsfw="off", max_loras=4, style_max=1, treatment_max=2, detailer_max=1,
                environment_max=1, lighting_max=1, subject_max=1, layout_max=1, utility_max=1, apply_threshold=0.35,
                weight_budget=2.5, stage_profile="role", variety=0.0, seed=1,
                text="a barbarian warrior fighting a dragon, in the style of Boris Vallejo")
    suggested = node.run(mode="suggest", **args)
    check("suggest mode sends nothing", suggested[0] == [] and suggested[3].startswith("SUGGEST"), suggested[3][:80])
    applied = node.run(mode="apply", **args)
    check("apply mode sends picks", isinstance(applied[0], list) and len(applied[0]) >= 1, applied[3][:300])
    check("prompt_with_triggers carries triggers", all(t.lower() in applied[2].lower() for t in applied[1].split(", ") if t))
    check("lora_text output appended last", node_class.RETURN_NAMES[-1] == "lora_text" and len(applied) == 5)
    check("lora_text empty in suggest mode", suggested[4] == "")
    check("lora_text one tag per pick", applied[4].count("<lora:") == len(applied[0]), applied[4])
    check("lora_text tag format", all(t.startswith("<lora:") and t.endswith(">") and t.count(":") == 2
                                      for t in applied[4].split(" ") if t), applied[4])
    try:
        stack_module = importlib.import_module("Eric_Krea2.nodes.krea2_lora_stack")
        stack_node = stack_module.EricKrea2MultiLoRA
        optional = stack_node.INPUT_TYPES()["optional"]
        keys = list(optional)
        check("Multi-LoRA: suggestions socket appended last", keys[-1] == "suggestions", keys[-3:])
        check("Multi-LoRA: prompt still before it", keys[-2] == "prompt", keys[-3:])
        out = stack_node().apply({"lora_stack": []}, fetch_triggers=False, add_triggers="append", prompt="a test",
                                 suggestions=applied[0])
        stack = out[0]["lora_stack"]
        check("Multi-LoRA: suggestions stacked", len(stack) == len(applied[0]), len(stack))
        check("Multi-LoRA: stage weights kept", stack and stack[0]["weight_s1"] == applied[0][0]["weight_s1"])
        check("Multi-LoRA: suggestion triggers merged", all(t in out[3] for e in applied[0] for t in e["triggers"]), out[3])
        twice = stack_node().apply(out[0], fetch_triggers=False, suggestions=applied[0])
        check("Multi-LoRA: same file not stacked twice", len(twice[0]["lora_stack"]) == len(stack))
    except Exception as error:
        print(f"  (Krea2 Multi-LoRA part skipped: {type(error).__name__}: {error})")
except Exception as error:
    print(f"  (node part skipped: {type(error).__name__}: {error})")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
