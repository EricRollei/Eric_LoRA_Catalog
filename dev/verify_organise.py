"""
Round-trip test for organise.py on a throwaway library (dev\\scratch_lib) with its own throwaway catalog.
Nothing in the real library or catalog is touched.

    python dev\\verify_organise.py
"""
import json
import os
import shutil
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lora_catalog import edits, facets                 # noqa: E402
from lora_catalog.config import Settings               # noqa: E402
from lora_catalog.organise import Organiser            # noqa: E402
from lora_catalog.store import Catalog                 # noqa: E402

SCRATCH = os.path.join(HERE, "scratch_lib")
LIB = os.path.join(SCRATCH, "loras")
DB = os.path.join(SCRATCH, "catalog.sqlite")
FAILS = []


def check(label, ok, extra=""):
    print(("  ok   " if ok else "  FAIL ") + label + (f"   {extra}" if extra and not ok else ""))
    if not ok:
        FAILS.append(label)


def touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def tree():
    out = []
    for folder, _dirs, files in os.walk(LIB):
        rel = os.path.relpath(folder, LIB)
        out.append(rel + os.sep)
        out += [os.path.join(rel, f) for f in files]
    return sorted(out)


def lm(path, stem):
    return json.dumps({"file_name": stem, "file_path": path.replace(os.sep, "/"),
                       "preview_url": os.path.join(os.path.dirname(path), stem + ".preview.png").replace(os.sep, "/"),
                       "sha256": "ab" * 32})


# -- build the scratch library -----------------------------------------------------------------------------------------
shutil.rmtree(SCRATCH, ignore_errors=True)
v1 = os.path.join(LIB, "Krea2", "graphic-art", "1001-Model_A", "2001-v1", "2001_a.safetensors")
v2 = os.path.join(LIB, "Krea2", "graphic-art", "1001-Model_A", "2002-v2", "2002_a.safetensors")
solo = os.path.join(LIB, "Krea2", "graphic-art", "1002-Model_B", "2003-v1", "2003_b.safetensors")
flat = os.path.join(LIB, "Krea2", "tools", "foo.safetensors")
flat2 = os.path.join(LIB, "Krea2", "tools", "foo.v2.safetensors")
clash = os.path.join(LIB, "Krea2", "tools", "bar.safetensors")
for path in (v1, v2, solo):
    stem = os.path.splitext(os.path.basename(path))[0]
    touch(path)
    touch(path + ".json", "{}")
    touch(os.path.splitext(path)[0] + ".metadata.json", lm(path, stem))
    touch(os.path.join(os.path.dirname(path), "images", stem.split("_")[0] + "_image.jpeg"))
for path in (flat, flat2, clash):
    stem = os.path.splitext(os.path.basename(path))[0]
    touch(path)
    touch(os.path.splitext(path)[0] + ".metadata.json", lm(path, stem))
    touch(os.path.splitext(path)[0] + ".preview.png")
touch(os.path.join(LIB, "Krea2", "tools", "images", "foo_1.jpg"))
touch(os.path.join(LIB, "Krea2", "tools", "images", "foo.v2_1.jpg"))
touch(os.path.join(LIB, "Krea2", "illustration", "bar.safetensors"), "already here")       # clash target
os.makedirs(os.path.join(LIB, "SDXL", "misc"))

settings = Settings(library_roots=[LIB], catalog_path=DB, moves_dir=os.path.join(SCRATCH, "moves"),
                    reference_dirs=[os.path.join(SCRATCH, "workflows")], comfy_url="http://127.0.0.1:9")
touch(os.path.join(SCRATCH, "workflows", "wf.json"), json.dumps({"lora_name": "Krea2\\tools\\foo.safetensors"}))
catalog = Catalog(DB)
for index, path in enumerate((v1, v2, solo, flat, flat2, clash)):
    rel = os.path.relpath(os.path.dirname(path), LIB).replace(os.sep, "/")
    catalog.put_lora({"path": path, "file_name": os.path.basename(path), "stem": os.path.splitext(os.path.basename(path))[0],
                      "root": LIB, "arch_folder": rel.split("/")[0], "rel_dir": rel, "sha256": f"{index:064x}",
                      "model_name": f"LoRA {index}"})
    catalog.connection.execute("insert into cards (path, kind, scope, effect, confidence) values (?, 'style_medium', "
                               "'global', 'test card', 0.9)", (path,))
catalog.close()
connection = sqlite3.connect(DB)
edits.ensure(connection)
facets.ensure(connection)
connection.execute("insert into card_edits (key, path, status, rating) values (?, ?, 'edited', 4)", (f"{0:064x}", v1))
connection.commit()
for path in (v1, v2, solo, flat, flat2, clash):
    facets.refresh_one(connection, path)
connection.close()
before = tree()

org = Organiser(settings)
print("folders:", [(f["rel"], f["loras"]) for f in org.folders()])
check("tree hides downloader folders", all("1001" not in f["rel"] for f in org.folders()))
check("graphic-art counts 3", next(f for f in org.folders() if f["rel"] == "Krea2/graphic-art")["loras"] == 3)

# -- units --------------------------------------------------------------------------------------------------------------
unit = org.unit_for(flat)
names = sorted(os.path.relpath(f, os.path.dirname(flat)) for f in unit["files"])
check("flat unit = foo + its companions, not foo.v2's", names == sorted(
    ["foo.safetensors", "foo.metadata.json", "foo.preview.png", os.path.join("images", "foo_1.jpg")]), names)
check("version unit", org.unit_for(v1)["kind"] == "version_folder")

# -- move ---------------------------------------------------------------------------------------------------------------
target = os.path.join(LIB, "Krea2", "illustration")
plan = org.plan_move([v1, v2, solo, flat, clash], target)
print("plan counts:", plan["counts"], "skipped:", [s["reason"] for s in plan["skipped"]], "cleanup:", plan["cleanup"])
check("clash skipped", any("already has bar.safetensors" in s["reason"] for s in plan["skipped"]))
check("model A folder will be emptied", len(plan["cleanup"]) == 2)

locked = open(os.path.join(os.path.dirname(solo), "images", "2003_image.jpeg"), "r")     # ComfyUI holding a file
result = org.apply(plan["id"])
locked.close()
print("apply:", {k: v for k, v in result.items() if k != "mapping"})
check("locked LoRA reported", any("Windows would not let go" in f["reason"] for f in result["failed"]))
check("solo still in place", os.path.isfile(solo))
check("no empty folder left for the failed LoRA", not os.path.exists(os.path.join(target, "1002-Model_B")))
new_v1 = os.path.join(target, "1001-Model_A", "2001-v1", "2001_a.safetensors")
new_flat = os.path.join(target, "foo.safetensors")
check("v1 moved with its folder", os.path.isfile(new_v1) and os.path.isfile(os.path.join(os.path.dirname(new_v1), "images", "2001_image.jpeg")))
check("flat moved with companions", all(os.path.isfile(os.path.join(target, n)) for n in
                                        ("foo.safetensors", "foo.metadata.json", "foo.preview.png", os.path.join("images", "foo_1.jpg"))))
check("foo.v2 left alone", os.path.isfile(flat2) and os.path.isfile(os.path.join(LIB, "Krea2", "tools", "images", "foo.v2_1.jpg")))
check("empty model folder removed", not os.path.exists(os.path.dirname(os.path.dirname(v1))))
meta = json.load(open(os.path.splitext(new_v1)[0] + ".metadata.json", encoding="utf-8"))
check("LM metadata file_path", meta["file_path"] == new_v1.replace(os.sep, "/"))
check("LM metadata preview_url", meta["preview_url"].startswith(os.path.dirname(new_v1).replace(os.sep, "/")))

db = sqlite3.connect(DB)
check("loras row moved", db.execute("select rel_dir from loras where path = ?", (new_v1,)).fetchone() ==
      ("Krea2/illustration/1001-Model_A/2001-v1",))
check("no row left at old path", db.execute("select count(*) from loras where path = ?", (v1,)).fetchone()[0] == 0)
check("card moved", db.execute("select count(*) from cards where path = ?", (new_v1,)).fetchone()[0] == 1)
check("card_facets moved", db.execute("select rel_dir from card_facets where path = ?", (new_v1,)).fetchone() ==
      ("Krea2/illustration/1001-Model_A/2001-v1",))
check("edit follows (rating kept)", db.execute("select path, rating from card_edits where key = ?", (f"{0:064x}",)).fetchone()
      == (new_v1, 4))
check("path_history", db.execute("select count(*) from path_history where batch = ?", (result["id"],)).fetchone()[0] == 3)
db.close()
refs = org.references(result["id"])
check("reference report finds the workflow", len(refs["files"]) == 1, refs)

# -- folders ------------------------------------------------------------------------------------------------------------
plan_new = org.plan_new_folder(os.path.join(LIB, "Krea2"), "comic")
check("new folder plan", not plan_new.get("error"))
check("bad name refused", bool(org.plan_new_folder(os.path.join(LIB, "Krea2"), "a:b").get("error")))
check("arch folder rename refused", bool(org.plan_rename_folder(os.path.join(LIB, "Krea2"), os.path.join(LIB, "K2")).get("error")))
new_result = org.apply(plan_new["id"])
check("new folder made", os.path.isdir(os.path.join(LIB, "Krea2", "comic")))
renamed = os.path.join(LIB, "Krea2", "comic", "illustration_moved")
plan_ren = org.plan_rename_folder(target, renamed)
check("into-itself refused", bool(org.plan_rename_folder(target, os.path.join(target, "x")).get("error")))
ren = org.apply(plan_ren["id"])
print("rename:", {k: v for k, v in ren.items() if k != "mapping"})
moved_v1 = os.path.join(renamed, "1001-Model_A", "2001-v1", "2001_a.safetensors")
check("folder moved", os.path.isfile(moved_v1))
db = sqlite3.connect(DB)
check("catalog follows folder move", db.execute("select rel_dir from loras where path = ?", (moved_v1,)).fetchone() ==
      ("Krea2/comic/illustration_moved/1001-Model_A/2001-v1",))
db.close()
check("expired plan refused", not org.apply(plan_ren["id"])["ok"])

# -- delete = trash ------------------------------------------------------------------------------------------------------
flat_was, v2_was = flat, v2                              # after the moves above they live in the renamed folder
flat = os.path.join(renamed, "foo.safetensors")
v2 = os.path.join(renamed, "1001-Model_A", "2002-v2", "2002_a.safetensors")
db = sqlite3.connect(DB)
db.execute("insert or replace into card_edits (key, path, status, rating) values (?, ?, 'edited', 5)", (f"{3:064x}", flat))
db.execute("insert into loras (path, file_name, stem, root, rel_dir, sha256) values (?, 'copy.safetensors', 'copy', ?, "
           "'SDXL/misc', ?)", (os.path.join(LIB, "SDXL", "misc", "copy.safetensors"), LIB, f"{3:064x}"))   # a second copy
db.commit(); db.close()
touch(os.path.join(LIB, "SDXL", "misc", "copy.safetensors"))
plan_del = org.plan_delete([flat, v2])
print("delete plan:", plan_del["counts"], plan_del["target"])
check("trash is beside the library, not inside it", org.root_of(plan_del["target"]) is None and "_lora_trash" in plan_del["target"])
del_result = org.apply(plan_del["id"])
print("delete:", {k: v for k, v in del_result.items() if k != "mapping"})
trash_flat = del_result["mapping"][flat]
check("flat LoRA + companions in the trash", os.path.isfile(trash_flat) and not os.path.exists(flat)
      and os.path.isfile(os.path.splitext(trash_flat)[0] + ".preview.png"))
check("version folder in the trash", os.path.isfile(del_result["mapping"][v2]) and not os.path.exists(os.path.dirname(v2)))
check("foo.v2 still there", os.path.isfile(flat2))
check("renamed folder lost only what was deleted", os.path.isfile(moved_v1))
db = sqlite3.connect(DB)
check("catalog rows gone", db.execute("select count(*) from loras where path in (?, ?)", (flat, v2)).fetchone()[0] == 0
      and db.execute("select count(*) from cards where path = ?", (flat,)).fetchone()[0] == 0)
check("edit kept, pointed at the other copy", db.execute("select path, rating from card_edits where key = ?", (f"{3:064x}",)).fetchone()
      == (os.path.join(LIB, "SDXL", "misc", "copy.safetensors"), 5))
db.close()
check("history lists the delete", any(b["kind"] == "delete" and b["moved"] == 2 for b in org.batches()))

# -- undo, newest first ---------------------------------------------------------------------------------------------------
check("older undo refused while newer stands", not org.undo(result["id"])["ok"])
outcome = org.undo(del_result["id"])
print("undo delete", outcome)
check("undo delete", outcome["ok"])
db = sqlite3.connect(DB)
check("rows back after undo", db.execute("select count(*) from loras where path in (?, ?)", (flat, v2)).fetchone()[0] == 2
      and db.execute("select count(*) from card_facets where path = ?", (flat,)).fetchone()[0] == 1)
check("edit points back", db.execute("select path from card_edits where key = ?", (f"{3:064x}",)).fetchone() == (flat,))
db.execute("delete from loras where path = ?", (os.path.join(LIB, "SDXL", "misc", "copy.safetensors"),))
db.commit(); db.close()
os.remove(os.path.join(LIB, "SDXL", "misc", "copy.safetensors"))
check("trash emptied of this batch", not os.path.exists(plan_del["target"]))
check("files back after undo", os.path.isfile(flat) and os.path.isfile(v2))
flat, v2 = flat_was, v2_was
for batch in (ren["id"], new_result["id"], result["id"]):
    outcome = org.undo(batch)
    print("undo", batch, outcome)
    check(f"undo {batch}", outcome["ok"])
after = tree()
check("disk back exactly as it was", before == after, sorted(set(before) ^ set(after)))
db = sqlite3.connect(DB)
check("catalog back", db.execute("select rel_dir from loras where path = ?", (v1,)).fetchone() ==
      ("Krea2/graphic-art/1001-Model_A/2001-v1",))
check("edit back", db.execute("select path from card_edits where key = ?", (f"{0:064x}",)).fetchone() == (v1,))
check("history marked undone", db.execute("select count(*) from path_history where undone = 0").fetchone()[0] == 0)
db.close()
meta = json.load(open(os.path.splitext(v1)[0] + ".metadata.json", encoding="utf-8"))
check("LM metadata back", meta["file_path"] == v1.replace(os.sep, "/"))

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
if not FAILS:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.rmtree(os.path.join(SCRATCH + "_trash_check"), ignore_errors=True)
shutil.rmtree(os.path.join(os.path.dirname(LIB), "_lora_trash"), ignore_errors=True)
