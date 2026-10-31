# Tests for the prompt library (2026-10-29) on a small fixture catalog. python dev/test_prompt_library.py
import json
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lora_catalog import prompt_library as PL  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name} {detail}")


d = tempfile.mkdtemp()
cat_path = os.path.join(d, "loras", "cat.sqlite")
os.makedirs(os.path.dirname(cat_path))
con = sqlite3.connect(cat_path)
con.executescript("""
CREATE TABLE loras (path TEXT, civitai_version_id INTEGER, stem TEXT, base_model TEXT);
CREATE TABLE card_facets (path TEXT, name TEXT, triggers TEXT, facets TEXT, base_model TEXT, arch_folder TEXT,
                          rating INTEGER);
CREATE TABLE examples (path TEXT, media_file TEXT, prompt TEXT, negative TEXT, weight REAL, base_checkpoint TEXT,
                       sampler TEXT, steps INTEGER, cfg REAL, nsfw_level INTEGER);
CREATE TABLE community_prompts (image_id INTEGER, version_id INTEGER, source TEXT, prompt TEXT, cleaned TEXT,
                                negative TEXT, weight REAL, base_model TEXT, prompt_style TEXT, nsfw_level TEXT,
                                reactions INTEGER, picked INTEGER, nsfw_num INTEGER);
""")
con.execute("insert into loras values ('L:/A.safetensors', 11, 'A', 'Krea 2')")
con.execute("insert into loras values ('L:/B.safetensors', 22, 'B', 'Flux.1 D')")
con.execute("insert into card_facets values ('L:/A.safetensors', 'Ink A', ?, ?, 'Krea 2', 'Krea2', 5)",
            (json.dumps(["@inkA", "woman"]), json.dumps({"medium": ["ink"], "period": ["art_nouveau"]})))
con.execute("insert into card_facets values ('L:/B.safetensors', 'Neon B', ?, ?, 'Flux.1 D', 'Flux', 2)",
            (json.dumps(["neonB style"]), json.dumps({"medium": ["neon"]})))
con.execute("insert into examples values ('L:/A.safetensors', 'img1.jpg', '@inkA, a woman reading under a lamp <lora:A:0.8>', '', 0.8, 'ck', 'Euler', 8, 1, 1)")
rows = [
    (1, 11, "top", "@inkA, a woman reading under a lamp", "@inkA, a woman reading under a lamp", "Mature", 50, None),
    (2, 11, "top", "  @INKA ,  A woman reading under a LAMP  ", "@INKA , A woman reading under a LAMP", "X", 10, 16),
    (3, 22, "newest", "neonB style, rainy street at night, (glow:1.2)", "neonB style, rainy street at night, glow", "None", 3, None),
    (4, 22, "top", "1girl, solo, masterpiece, woman, red dress", "1girl, solo, masterpiece, woman, red dress", "Soft", 200, None),
]
for i, v, s, p, c, lvl, rx, num in rows:
    con.execute("insert into community_prompts values (?,?,?,?,?,'',1.0,'',?,?,?,0,?)",
                (i, v, s, p, c, "natural", lvl, rx, num))
# lots of ordinary prompts that say "woman" -> "woman" is a generic word, never stripped
for k in range(60):
    con.execute("insert into community_prompts values (?,22,'newest',?,?,'',1.0,'','natural','None',0,0,NULL)",
                (100 + k, f"a woman number {k} in a garden", f"a woman number {k} in a garden"))
con.commit()
con.close()
S = types.SimpleNamespace(catalog_path=cat_path, prompt_library_dir=os.path.join(d, "prompts"), card_model="m")

print("[1] build")
r1 = PL.build(S, log=lambda m: None)
lib = sqlite3.connect(PL.library_path(S))
texts = [t for (t,) in lib.execute("select text from prompts")]
check("library in prompt_library_dir", os.path.exists(os.path.join(d, "prompts", "_prompt_library.sqlite")))
check("trigger @inkA stripped", all("@inka" not in t.lower() for t in texts))
check("'woman' kept (ordinary word)", any("woman reading" in t.lower() for t in texts))
check("phrase trigger 'neonB style' stripped", any(t.startswith("rainy street") for t in texts), str(texts[:3]))
pid = PL.prompt_id("a woman reading under a lamp")
occ = lib.execute("select count(*) from occurrences where prompt_id=?", (pid,)).fetchone()[0]
check("case / spacing / lora tag variants = one prompt, 3 occurrences", occ == 3, f"occ={occ}")
nsfw = lib.execute("select nsfw from prompts where id=?", (pid,)).fetchone()[0]
check("worst NSFW wins; exact number preferred (16 -> XXX)", nsfw == 4, f"nsfw={nsfw}")
check("stripped recorded", lib.execute("select count(*) from stripped where prompt_id=?", (pid,)).fetchone()[0] >= 1)
check("facets copied", lib.execute("select count(*) from facets where prompt_id=? and value='art_nouveau'", (pid,)).fetchone()[0] == 1)
lib.close()

print("[2] incremental")
c2 = sqlite3.connect(cat_path)
c2.execute("insert into community_prompts values (5,11,'newest','@inkA, a woman reading under a lamp','@inkA, a woman reading under a lamp','','',1.0,'','natural','None',7,0,1)".replace(",'','',1.0", ",'',1.0"))
c2.commit()
c2.close()
before = r1["prompts"]
r2 = PL.build(S, log=lambda m: None)
lib = sqlite3.connect(PL.library_path(S))
check("no new prompt, one more occurrence", r2["prompts"] == before and r2["new_prompts"] == 0 and
      lib.execute("select occurrences from prompts where id=?", (pid,)).fetchone()[0] == 4)
lib.close()

print("[3] select / export")
sel = PL.select(S, max_nsfw="R")
check("max R excludes the XXX prompt", all(r["id"] != pid for r in sel))
sel = PL.select(S, max_nsfw="XXX", base="krea")
check("base filter (Krea via its LoRA)", any(r["id"] == pid for r in sel))
sel = PL.select(S, max_nsfw="XXX", lora_rating=4)
check("lora-rating 4: only prompts that came with A (5 stars)", sel and all("A" in r["came_with"] for r in sel))
sel = PL.select(S, max_nsfw="XXX", facets=['period="art nouveau"'])
check("facet filter", [r["id"] for r in sel] == [pid])
sel = PL.select(S, max_nsfw="XXX", clean_text=True, min_reactions=100)
check("min reactions", [r["text"] for r in sel] == ["1girl, solo, masterpiece, woman, red dress"], str([r["text"] for r in sel]))
sel = PL.select(S, max_nsfw="XXX")
out = os.path.join(d, "out.txt")
PL.export(S, sel, out, log=lambda m: None)
lines = open(out, encoding="utf-8").read().splitlines()
check("txt: one prompt per line", len(lines) == len(sel))
PL.export(S, sel, os.path.join(d, "split"), "txt", split_by="medium", log=lambda m: None)
check("split by facet", sorted(os.listdir(os.path.join(d, "split"))) == ["ink.txt", "neon.txt"],
      str(sorted(os.listdir(os.path.join(d, "split")))))
PL.export(S, sel, os.path.join(d, "o.jsonl"), log=lambda m: None)
first = json.loads(open(os.path.join(d, "o.jsonl"), encoding="utf-8").readline())
check("jsonl carries original + triggers", "original" in first and "triggers_stripped" in first)
wt = os.path.join(d, "wt.txt")
PL.export(S, [r for r in sel if r["id"] == pid], wt, with_triggers=True, log=lambda m: None)
check("--with-triggers gives the original", "@inka" in open(wt, encoding="utf-8").read().lower())

print("[4] levels")
check("text levels", [PL.level_index(x) for x in ("None", "Soft", "Mature", "X")] == [0, 1, 2, 3])
check("numbers", [PL.level_index(x) for x in (1, 2, 4, 8, 16, 32)] == [0, 1, 2, 3, 4, PL.BLOCKED])
check("expander system prompt read from Eric_Krea2", PL.krea2_expander_system()[1] == "Eric_Krea2 Magic Prompt")

print("ALL OK" if fails == 0 else f"FAILURES: {fails}")
