# 2026-10-27: stack synergy - markmodel fits synergy[a, b] (how much better/worse a pair scored together than the
# two LoRAs' own ratings predict, prompt and checkpoint taken out), the card shows "Pairs well with" (yours) and
# "Others stack it with" (Civitai), and the Suggester orders its picks by your synergy with what it already chose.
import os
import shutil
import sys
import py_compile

C = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_LoRA_Catalog"
L = os.path.join(C, "lora_catalog")
TAG = "_pre_synergy_2026-10-27"
P = {}

P[os.path.join(L, "markmodel.py")] = [
    ('''ITERATIONS = 40
''', '''ITERATIONS = 40

# Stack synergy (2026-10-27). After the main fit, every rated stack of 2+ swept LoRAs is compared with what its
# members predict on their own (the mean of their quality + checkpoint pair + prompt-fit terms, on that prompt and
# checkpoint). The difference is credited to every pair in the stack (split evenly) and shrunk toward zero:
#     synergy[a, b] = sum(w * (stars - expected)) / (sum(w) + SYNERGY_LAMBDA)
# + = better together than apart, - = worse together. Most reliable when the LoRAs were also rated alone
# (sweep stack_parts "each LoRA alone"); a pair only ever seen together partly hides in each LoRA's own quality.
SYNERGY_LAMBDA = 2.0
SYN_TABLE = """
CREATE TABLE IF NOT EXISTS lora_synergy (
    a TEXT, b TEXT, a_path TEXT, a_name TEXT, b_path TEXT, b_name TEXT, synergy REAL, n REAL, n_marks INTEGER,
    stars REAL, expected REAL, by_checkpoint TEXT, updated_at TEXT, PRIMARY KEY (a, b))
"""
''', 1),
    ('''def explain(model: Dict[str, Any], key: str) -> Dict[str, Any]:''', '''def stack_marks(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    """One record per rated mark of a stack with 2+ swept LoRAs (2026-10-27)."""
    cols = {r[1] for r in connection.execute("pragma table_info(stacks)")}
    if "fit" not in cols:
        return []
    out = []
    for row in connection.execute("select * from stacks where n_loras > 1"):
        y, w = _score(row)
        if y is None:
            continue
        loras = [l for l in json.loads(row["loras"] or "[]") if not l.get("panel")]
        if len(loras) < 2:
            continue
        try:
            fits = json.loads(row["fit"] or "{}")
        except ValueError:
            fits = {}
        members = []
        for entry in loras:
            key = lora_key(entry)
            f = fits.get(key)
            members.append({"key": key, "fit": f.get("fit") if isinstance(f, dict) else None,
                            "path": entry.get("path") or "", "name": entry.get("name") or ""})
        pk = _prompt_key(row)
        out.append({"y": y, "w": w, "pk": pk, "ck": pk[0], "members": members})
    return out


def fit_synergy(model: Dict[str, Any], marks: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(a, b) sorted keys: {synergy, n, n_marks, stars, expected, by_checkpoint, a_path, a_name, b_path, b_name}}."""
    acc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for m in marks:
        base = model["mu"] + model["prompt"].get(m["pk"], 0.0) + model["checkpoint"].get(m["ck"], 0.0)
        own = []
        for e in m["members"]:
            x = (e["fit"] - model["fbar"]) if e["fit"] is not None else 0.0
            own.append(model["quality"].get(e["key"], 0.0) + model["pair"].get((e["key"], m["ck"]), 0.0)
                       + (model["B"] + model["slope"].get(e["key"], 0.0)) * x)
        expected = base + sum(own) / len(own)
        resid = m["y"] - expected
        info = {e["key"]: e for e in m["members"]}
        keys = sorted(info)
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        if not pairs:
            continue
        wp = m["w"] / len(pairs)
        for a, b in pairs:
            s = acc.setdefault((a, b), {"num": 0.0, "den": 0.0, "n_marks": 0, "ys": [], "ex": [], "ck": {},
                                        "a_path": info[a]["path"], "a_name": info[a]["name"],
                                        "b_path": info[b]["path"], "b_name": info[b]["name"]})
            s["num"] += wp * resid
            s["den"] += wp
            s["n_marks"] += 1
            s["ys"].append(m["y"])
            s["ex"].append(expected)
            s["ck"].setdefault(m["ck"], []).append(resid)
    out = {}
    for pair, s in acc.items():
        out[pair] = {"synergy": s["num"] / (s["den"] + SYNERGY_LAMBDA), "n": s["den"], "n_marks": s["n_marks"],
                     "stars": sum(s["ys"]) / len(s["ys"]), "expected": sum(s["ex"]) / len(s["ex"]),
                     "by_checkpoint": {ck: round(sum(v) / len(v), 2) for ck, v in s["ck"].items()},
                     "a_path": s["a_path"], "a_name": s["a_name"], "b_path": s["b_path"], "b_name": s["b_name"]}
    return out


def explain(model: Dict[str, Any], key: str) -> Dict[str, Any]:''', 1),
    ('''        connection.commit()
    finally:
        connection.close()
    log(f"[marks] model:''', '''        # stack synergy (2026-10-27)
        connection.execute(SYN_TABLE)
        syn = fit_synergy(model, stack_marks(connection))
        connection.execute("DELETE FROM lora_synergy")
        for (a, b), s in syn.items():
            connection.execute("INSERT OR REPLACE INTO lora_synergy VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (a, b, s["a_path"], s["a_name"], s["b_path"], s["b_name"], round(s["synergy"], 4),
                                round(s["n"], 3), s["n_marks"], round(s["stars"], 3), round(s["expected"], 3),
                                json.dumps(s["by_checkpoint"]), now))
        connection.commit()
    finally:
        connection.close()
    log(f"[marks] stack synergy: {len(syn)} pair(s)")
    log(f"[marks] model:''', 1),
]

P[os.path.join(L, "stacks.py")] = [
    ('''        record["stars_mean"] = round(sum(_st) / len(_st), 2) if _st else None
''', '''        record["stars_mean"] = round(sum(_st) / len(_st), 2) if _st else None
        record["synergy"], record["co_use"] = self._partners(key, path)       # 2026-10-27
''', 1),
    ('''    @staticmethod
    def _tag_summary(obs: List[Dict[str, Any]], library: Dict[str, float]):''', '''    def _partners(self, key: str, path: str):
        """This LoRA's partners (2026-10-27): your stack synergy (markmodel lora_synergy) and what Civitai users stack
        it with (run.py harvest -> lora_pairs). Either list may be empty."""
        syn, co = [], []
        with self.connect() as connection:
            tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
            if "lora_synergy" in tables:
                for r in connection.execute("select * from lora_synergy where a = ? or b = ?", (key, key)):
                    mine_a = r["a"] == key
                    syn.append({"name": r["b_name"] if mine_a else r["a_name"],
                                "path": r["b_path"] if mine_a else r["a_path"],
                                "synergy": round(r["synergy"], 2), "n": r["n_marks"],
                                "stars": round(r["stars"], 2), "expected": round(r["expected"], 2)})
                syn.sort(key=lambda s: -s["synergy"])
            if "lora_pairs" in tables and "loras" in tables:
                cols = {r[1] for r in connection.execute("pragma table_info(loras)")}
                if "civitai_version_id" in cols:
                    v = connection.execute("select civitai_version_id from loras where path = ?", (path,)).fetchone()
                    if v and v[0]:
                        vk = f"v:{int(v[0])}"
                        for r in connection.execute("select * from lora_pairs where a = ? or b = ? order by score desc "
                                                    "limit 8", (vk, vk)).fetchall():
                            other = r["b"] if r["a"] == vk else r["a"]
                            owned = r["owned_b"] if r["a"] == vk else r["owned_a"]
                            name, opath = other, ""
                            if other.startswith("v:"):
                                lr = connection.execute("select path from loras where civitai_version_id = ?",
                                                        (int(other[2:]),)).fetchone()
                                if lr and lr[0]:
                                    opath = lr[0]
                                    name = os.path.splitext(os.path.basename(lr[0]))[0]
                            co.append({"name": name, "path": opath, "owned": bool(owned), "users": r["n"],
                                       "score": round(r["score"] or 0, 2)})
        return syn, co

    @staticmethod
    def _tag_summary(obs: List[Dict[str, Any]], library: Dict[str, float]):''', 1),
]

P[os.path.join(L, "editor.html")] = [
    ('''function recordInner() {
  if (!RECORD) return `<small class="muted">loading...</small>`;
  if (RECORD.error) return `<small class="muted">${esc(RECORD.error)}</small>`;
  if (!RECORD.n) return `<small class="muted">No sweep ratings yet. Rate its cells on the <a href="/sweeps">Sweeps page</a>.</small>`;''',
     '''function partnersHtml() {        // 2026-10-27: your stack synergy + Civitai co-use
  const syn = RECORD.synergy || [], co = RECORD.co_use || [];
  if (!syn.length && !co.length) return "";
  const row = s => `<tr><td>${esc(s.name)}</td><td class="n">${s.synergy > 0 ? "+" : ""}${s.synergy.toFixed(2)}&#9733;</td>
     <td class="n muted">${s.stars.toFixed(1)} vs ${s.expected.toFixed(1)}</td><td class="n muted">${s.n}</td></tr>`;
  let h = "";
  if (syn.length) {
    const good = syn.filter(s => s.synergy >= 0.05).slice(0, 6);
    const bad = syn.filter(s => s.synergy <= -0.05).slice(-4).reverse();
    h += `<div class="rectbl"><b title="How much better your stacks with this partner scored than the two LoRAs' own ratings predict (prompt and checkpoint taken out). Columns: synergy, stack stars vs expected, marks. Pulled toward 0 until several stacks back it; most reliable when both LoRAs were also rated alone.">Pairs well with (your stacks)</b><table>${
      good.map(row).join("") || `<tr><td class="muted">no partner above expectation yet</td></tr>`}</table></div>`;
    if (bad.length) h += `<div class="rectbl"><b title="Your stacks with these partners scored below what the two LoRAs predict.">Worse together</b><table>${bad.map(row).join("")}</table></div>`;
  }
  if (co.length) h += `<div class="rectbl"><b title="Other Civitai users' stacks (run.py harvest): how many different users put the two together, beyond chance. Popularity, not your taste.">Others stack it with (Civitai)</b><table>${
    co.slice(0, 6).map(c => `<tr><td>${esc(c.name)}</td><td class="n muted">${c.users} users</td><td class="n">${c.owned ? "" : "not owned"}</td></tr>`).join("")}</table></div>`;
  return `<div class="recgrid">${h}</div>`;
}
function recordInner() {
  if (!RECORD) return `<small class="muted">loading...</small>`;
  if (RECORD.error) return `<small class="muted">${esc(RECORD.error)}</small>`;
  if (!RECORD.n) return `<small class="muted">No sweep ratings yet. Rate its cells on the <a href="/sweeps">Sweeps page</a>.</small>` + partnersHtml();''', 1),
    ('''    tbl("With", RECORD.by_partner)}</div>`;
''', '''    tbl("With", RECORD.by_partner)}</div>`;
  h += partnersHtml();
''', 1),
]

_SE = os.path.join(C, "suggest", "engine.py")
P[_SE] = [
    ('''def suggest(request: SuggestRequest, settings: Optional[SuggestSettings] = None) -> SuggestResult:''',
     '''# Your stack ratings (LoRA Catalog -> markmodel.py -> lora_synergy), 2026-10-27. synergy = how much better (+) or
# worse (-) a pair scored together than the two LoRAs' own ratings predict. Once a pair has SYNERGY_MIN_MARKS rated
# stacks, a candidate's place in the queue moves by SYNERGY_STEP x the summed synergy with the LoRAs already chosen
# (pins included), within +-SYNERGY_CAP - it reorders the picks, it never changes their scores.
SYNERGY_STEP = 0.06
SYNERGY_CAP = 0.15
SYNERGY_MIN_MARKS = 2


def load_synergy(catalog: str) -> Dict[Tuple[str, str], Tuple[float, int]]:
    import os as _os
    import sqlite3 as _sqlite3
    out: Dict[Tuple[str, str], Tuple[float, int]] = {}
    try:
        con = _sqlite3.connect(f"file:{catalog}?mode=ro", uri=True, timeout=10)
        try:
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            if "lora_synergy" not in tables or "loras" not in tables:
                return out
            rows = con.execute("select la.path, lb.path, s.synergy, s.n_marks from lora_synergy s "
                               "join loras la on lower(la.sha256) = s.a join loras lb on lower(lb.sha256) = s.b").fetchall()
        finally:
            con.close()
    except Exception:
        return out
    for pa, pb, syn, n in rows:
        if pa and pb:
            out[tuple(sorted((_os.path.normcase(pa), _os.path.normcase(pb))))] = (float(syn), int(n or 0))
    return out


def synergy_bonus(pick: "Pick", chosen: List["Pick"],
                  synergy: Dict[Tuple[str, str], Tuple[float, int]]) -> Tuple[float, List[str]]:
    import os as _os
    total, notes = 0.0, []
    for other in chosen:
        key = tuple(sorted((_os.path.normcase(pick.card.path), _os.path.normcase(other.card.path))))
        s = synergy.get(key)
        if not s or s[1] < SYNERGY_MIN_MARKS:
            continue
        total += s[0]
        notes.append(f"your stacks: {'+' if s[0] >= 0 else ''}{s[0]:.1f} stars with {other.card.name} ({s[1]} marks)")
    return max(-SYNERGY_CAP, min(SYNERGY_CAP, SYNERGY_STEP * total)), notes


def suggest(request: SuggestRequest, settings: Optional[SuggestSettings] = None) -> SuggestResult:''', 1),
    ('''    for _order, pick in candidates:
        if len([p for p in chosen if p.role != "slider"]) >= request.max_loras:
            break''', '''    synergy = load_synergy(settings.catalog)          # your stack ratings, 2026-10-27
    remaining = [c for c in candidates if c[1].score >= request.threshold] if synergy else list(candidates)
    while remaining:
        j = 0
        if synergy and chosen:
            j = max(range(len(remaining)),
                    key=lambda i: remaining[i][0] + synergy_bonus(remaining[i][1], chosen, synergy)[0])
        _order, pick = remaining.pop(j)
        if synergy and chosen:
            _bonus, _notes = synergy_bonus(pick, chosen, synergy)
            pick.reasons.extend(_notes)
        if len([p for p in chosen if p.role != "slider"]) >= request.max_loras:
            break''', 1),
]


def main():
    staged, ok = {}, True
    for path, edits in P.items():
        text = open(path, encoding="utf-8", newline="").read()
        nl = "\r\n" if "\r\n" in text else "\n"
        for k, (old, new, want) in enumerate(edits):
            o, n = old.replace("\n", nl), new.replace("\n", nl)
            got = text.count(o)
            if got != want:
                print(f"MISMATCH {os.path.basename(path)} edit#{k}: found {got}, expected {want}: {old[:70]!r}")
                ok = False
                continue
            text = text.replace(o, n)
        staged[path] = text
    if not ok:
        print("ABORTED - nothing written")
        return 1
    bkdir = os.path.join(C, "docs", "backup")
    os.makedirs(bkdir, exist_ok=True)
    for path, text in staged.items():
        base, ext = os.path.splitext(os.path.basename(path))
        bk = os.path.join(bkdir, f"{base}{TAG}{ext}")
        if not os.path.exists(bk):
            shutil.copy2(path, bk)
        open(path, "w", encoding="utf-8", newline="").write(text)
        if path.endswith(".py"):
            py_compile.compile(path, doraise=True)
        print("patched", os.path.relpath(path, C))
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
