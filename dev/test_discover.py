# Tests for Discover stage 1 (2026-10-29) on a small fixture catalog. python dev/test_discover.py
import json
import os
import sqlite3
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lora_catalog import discover, harvest  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name} {detail}")


def fixture() -> types.SimpleNamespace:
    d = tempfile.mkdtemp()
    path = os.path.join(d, "cat.sqlite")
    con = sqlite3.connect(path)
    con.executescript(harvest.TABLES)
    con.executescript("""
    CREATE TABLE loras (path TEXT, sha256 TEXT, civitai_version_id INTEGER, civitai_model_id INTEGER, model_name TEXT,
                        stem TEXT, base_model TEXT, arch_folder TEXT, rel_dir TEXT);
    CREATE TABLE lora_observed (key TEXT, path TEXT, name TEXT, quality REAL, n REAL, n_marks INTEGER, slope REAL,
                                by_checkpoint TEXT, flaws TEXT, praise TEXT, updated_at TEXT);
    CREATE TABLE card_edits (key TEXT, rating INTEGER);
    """)
    # yours: GOOD (rated high), BAD (rated low), MEH (unrated), FLUX (another shelf)
    mine = [("GOOD", "aa", 1, 101, "Krea2/x"), ("BAD", "bb", 2, 102, "Krea2/y"), ("MEH", "cc", 3, 103, "Krea2"),
            ("FLUX", "dd", 4, 104, "Flux/z")]
    for name, sha, vid, mid, rel in mine:
        con.execute("insert into loras values (?,?,?,?,?,?,?,?,?)",
                    (f"L:/{rel}/{name}.safetensors", sha, vid, mid, name, name, "", rel.split("/")[0], rel))
    con.execute("insert into lora_observed (key, quality) values ('aa', 1.0)")    # taste 1.5
    con.execute("insert into lora_observed (key, quality) values ('bb', -2.0)")   # taste 0
    con.execute("insert into card_edits values ('aa', 5)")
    # candidates you don't own: X with GOOD, Y with BAD, Z with MEH - equal pair scores; W only on Flux; V = other
    # version of a model you own (model id 101 via its record is not available offline -> not tested here)
    pairs = [("v:1", "v:900", "Krea 2", 1, 0), ("v:2", "v:901", "Krea 2", 1, 0), ("v:3", "v:902", "Krea 2 Turbo", 1, 0),
             ("v:4", "v:903", "Flux.1 D", 1, 0)]
    for a, b, base, oa, ob in pairs:
        con.execute("insert into lora_pairs values (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (a, b, base, 5, 100, 4.0, 2.0, "[1.0]", "[0.8, 1.0]", json.dumps([11, 12]), oa, ob))
    con.commit()
    con.close()
    return types.SimpleNamespace(catalog_path=path, civitai_host="https://civitai.com")


s = fixture()
con = sqlite3.connect(s.catalog_path)
con.row_factory = sqlite3.Row

print("[1] taste")
check("unrated = 1", discover.taste(None) == 1.0)
check("quality 1.0 = 1.5", discover.taste(1.0) == 1.5)
check("quality -2 = 0 (cut)", discover.taste(-2.0) == 0.0)
check("clamped high", discover.taste(5.0) == 2.0)
check("clamped low", discover.taste(-1.4) == 0.3 or abs(discover.taste(-1.4) - 0.3) < 1e-9)

print("[2] shelf bases")
check("Krea2 -> Krea 2 + Turbo", discover.shelf_bases(con, s, "Krea2") == ["Krea 2", "Krea 2 Turbo"],
      str(discover.shelf_bases(con, s, "Krea2")))
check("Flux -> Flux.1 D", discover.shelf_bases(con, s, "Flux") == ["Flux.1 D"])
check("unknown shelf -> []", discover.shelf_bases(con, s, "Moss") == [])
check("no shelf -> None (all)", discover.shelf_bases(con, s, "") is None)
s2 = types.SimpleNamespace(**vars(s), discover_bases={"Krea2": ["Krea 2"]})
check("settings override", discover.shelf_bases(con, s2, "Krea2") == ["Krea 2"])

print("[3] ranking")
r = discover.rank_candidates(con, s, ["Krea 2", "Krea 2 Turbo"])
check("X (with your 5-star) first", r["v:900"]["total"] > r["v:902"]["total"] > r["v:901"]["total"],
      f"X={r['v:900']['total']} Z={r['v:902']['total']} Y={r['v:901']['total']}")
check("Y (with a LoRA you rated poorly) counts 0", r["v:901"]["total"] == 0.0)
check("Flux candidate not on the Krea2 shelf", "v:903" not in r)
r0 = discover.rank_candidates(con, s, ["Krea 2", "Krea 2 Turbo"], use_taste=False)
check("--no-taste: equal pair scores rank equal", r0["v:900"]["total"] == r0["v:901"]["total"] == r0["v:902"]["total"] == 2.0)
check("partner carries your rating", r["v:900"]["partners"][0]["rating"] == 5)
check("weights collected", r["v:900"]["weights"] == [0.8, 1.0])

print("[4] dismiss")
con.close()
discover.dismiss(s, "v:900", "X")
d = discover.candidates(s, "Krea2", civitai=False)
check("dismissed left out", all(it["key"] != "v:900" for it in d["items"]), str([it["key"] for it in d["items"]]))
check("dismissed counted", d["dismissed"] == 1)
discover.dismiss(s, "v:900", undo=True)
d = discover.candidates(s, "Krea2", civitai=False)
check("undo restores", any(it["key"] == "v:900" for it in d["items"]))
check("first item is X", d["items"][0]["key"] == "v:900", d["items"][0]["key"])

class _NoNet:
    def __init__(self, *a, **k): pass
    def get(self, *a, **k): return {}


harvest.CivitaiClient = _NoNet            # the CLI looks up missing names online; not in a test
print("[5] wishlist CLI parity")
w_taste = harvest.wishlist(s, top=5, plain=True)
w_plain = harvest.wishlist(s, top=5, plain=True, use_taste=False)
check("taste header", "taste-weighted" in w_taste and "taste-weighted" not in w_plain)
discover.dismiss(s, "v:902", "Z")
check("wishlist leaves dismissed out", "v:902" not in harvest.wishlist(s, top=5, plain=True))

print("[6] empty shelf message")
d = discover.candidates(s, "Moss", civitai=False)
check("note for an unharvested shelf", d["items"] == [] and "harvest --only Moss" in (d.get("note") or ""))

print("ALL OK" if fails == 0 else f"FAILURES: {fails}")
