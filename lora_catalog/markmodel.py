"""
What your sweep marks say about each LoRA, with the prompt and the checkpoint taken out (2026-10-17).

A star rating mixes four things: how good the LoRA is, how well the prompt suits it, how well it suits the
checkpoint, and how hard the prompt is on that checkpoint anyway. A sweep crosses LoRAs x prompts x checkpoints at
one seed, so the four can be told apart. This is an additive model fitted by backfitting with ridge shrinkage (each
estimate is pulled toward zero until the marks earn it - three ratings never make a verdict):

    stars = mu + prompt[checkpoint, prompt] + checkpoint[ck]               (baseline cells: no LoRA terms)
               + quality[lora] + (slope + slope[lora]) * (fit - mean fit)  (fit: markfit.py, 0..1)
               + pair[lora, ck]                                             (this LoRA on this checkpoint)

    quality[lora]   stars above (+) or below (-) the average LoRA at an average prompt fit, any checkpoint
    slope[lora]     how much more it depends on the prompt being in its range than LoRAs do on average
    pair[lora, ck]  how much better or worse it does on that checkpoint than its quality says

Weights: a starred mark counts 1, an old good / unsure / bad mark (no stars; 4 / 3 / 2) counts 0.5 - those were
'did it do anything' judgments. A stack cell counts 1/n for each of its n LoRAs. Baseline cells (no LoRA) rated
with stars measure how hard the prompt is on that checkpoint.

Results go to the catalog table lora_observed (one row per LoRA, keyed by sha256), which the card editor and the
LoRA Suggester read. update() rewrites it; the editor runs it a few seconds after you stop marking.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

LEGACY_WEIGHT = 0.5
VERDICT_SCORE = {"good": 4.0, "unsure": 3.0, "bad": 2.0}
LAMBDA = {"prompt": 2.0, "checkpoint": 5.0, "quality": 3.0, "pair": 4.0, "slope": 0.6}
ITERATIONS = 40

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
IN_RANGE, OUT_RANGE = 0.8, 0.2       # "in its range" vs "outside it" when the fit effect is put into words

TABLE = """
CREATE TABLE IF NOT EXISTS lora_observed (
    key TEXT PRIMARY KEY, path TEXT, name TEXT, quality REAL, n REAL, n_marks INTEGER, slope REAL,
    by_checkpoint TEXT, flaws TEXT, praise TEXT, updated_at TEXT)
"""


def _score(row) -> Tuple[Optional[float], float]:
    if row["stars"]:
        return float(row["stars"]), 1.0
    v = VERDICT_SCORE.get(row["verdict"] or "")
    return (v, LEGACY_WEIGHT) if v is not None else (None, 0.0)


def _prompt_key(row) -> Tuple[str, str]:
    text = str(row["prompt_base"] or row["prompt"] or "").strip().lower()
    return (row["checkpoint"] or "?", " ".join(text.split())[:200])


def lora_key(entry: Dict[str, Any]) -> str:
    return (entry.get("sha256") or "").lower() or os.path.basename(str(entry.get("path") or entry.get("name") or ""))


def observations(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    """One record per (mark, swept LoRA), plus one per rated baseline mark (lora None)."""
    cols = {r[1] for r in connection.execute("pragma table_info(stacks)")}
    if "fit" not in cols:
        return []
    out = []
    for row in connection.execute("select * from stacks"):
        y, w = _score(row)
        if y is None:
            continue
        pk = _prompt_key(row)
        loras = [l for l in json.loads(row["loras"] or "[]") if not l.get("panel")]
        if not loras:
            if row["n_loras"] == 0:
                out.append({"y": y, "w": w, "pk": pk, "ck": pk[0], "lora": None})
            continue
        fits = {}
        try:
            fits = json.loads(row["fit"] or "{}")
        except ValueError:
            fits = {}
        for entry in loras:
            key = lora_key(entry)
            f = (fits.get(key) or {}).get("fit") if isinstance(fits.get(key), dict) else None
            out.append({"y": y, "w": w / len(loras), "pk": pk, "ck": pk[0], "lora": key, "fit": f,
                        "path": entry.get("path"), "name": entry.get("name")})
    return out


def fit_model(obs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not obs:
        return {"mu": 3.0, "quality": {}, "slope": {}, "pair": {}, "checkpoint": {}, "prompt": {}, "B": 0.0, "fbar": 0.5,
                "n": {}}
    fits = [o["fit"] for o in obs if o["lora"] and o.get("fit") is not None]
    fbar = sum(fits) / len(fits) if fits else 0.5
    for o in obs:
        o["x"] = (o["fit"] - fbar) if (o["lora"] and o.get("fit") is not None) else 0.0
    W = sum(o["w"] for o in obs)
    mu = sum(o["w"] * o["y"] for o in obs) / W
    P: Dict[Any, float] = collections.defaultdict(float)
    C: Dict[str, float] = collections.defaultdict(float)
    Q: Dict[str, float] = collections.defaultdict(float)
    S: Dict[str, float] = collections.defaultdict(float)
    I: Dict[Tuple[str, str], float] = collections.defaultdict(float)
    B = 0.0

    def pred(o, skip: str = "") -> float:
        v = mu + (P[o["pk"]] if skip != "P" else 0) + (C[o["ck"]] if skip != "C" else 0)
        if o["lora"]:
            l = o["lora"]
            v += (Q[l] if skip != "Q" else 0) + (I[(l, o["ck"])] if skip != "I" else 0)
            v += (B if skip != "B" else 0) * o["x"] + (S[l] if skip != "S" else 0) * o["x"]
        return v

    def update(skip: str, key_of, lam: float, only_lora: bool = False):
        num: Dict[Any, float] = collections.defaultdict(float)
        den: Dict[Any, float] = collections.defaultdict(float)
        for o in obs:
            if only_lora and not o["lora"]:
                continue
            k = key_of(o)
            num[k] += o["w"] * (o["y"] - pred(o, skip))
            den[k] += o["w"]
        return {k: num[k] / (den[k] + lam) for k in num}

    for _ in range(ITERATIONS):
        P = collections.defaultdict(float, update("P", lambda o: o["pk"], LAMBDA["prompt"]))
        C = collections.defaultdict(float, update("C", lambda o: o["ck"], LAMBDA["checkpoint"]))
        Q = collections.defaultdict(float, update("Q", lambda o: o["lora"], LAMBDA["quality"], True))
        I = collections.defaultdict(float, update("I", lambda o: (o["lora"], o["ck"]), LAMBDA["pair"], True))
        num = sum(o["w"] * o["x"] * (o["y"] - pred(o, "B")) for o in obs if o["lora"])
        den = sum(o["w"] * o["x"] * o["x"] for o in obs if o["lora"])
        B = num / (den + LAMBDA["slope"]) if den else 0.0
        sn: Dict[str, float] = collections.defaultdict(float)
        sd: Dict[str, float] = collections.defaultdict(float)
        for o in obs:
            if o["lora"] and o["x"]:
                sn[o["lora"]] += o["w"] * o["x"] * (o["y"] - pred(o, "S"))
                sd[o["lora"]] += o["w"] * o["x"] * o["x"]
        S = collections.defaultdict(float, {k: sn[k] / (sd[k] + LAMBDA["slope"]) for k in sn})
    n: Dict[str, float] = collections.defaultdict(float)
    for o in obs:
        if o["lora"]:
            n[o["lora"]] += o["w"]
    return {"mu": mu, "quality": dict(Q), "slope": dict(S), "pair": dict(I), "checkpoint": dict(C), "prompt": dict(P),
            "B": B, "fbar": fbar, "n": dict(n)}


def stack_marks(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
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


def explain(model: Dict[str, Any], key: str) -> Dict[str, Any]:
    """The model's view of one LoRA, in numbers and sentences."""
    q = model["quality"].get(key)
    if q is None:
        return {}
    slope = model["B"] + model["slope"].get(key, 0.0)
    range_gap = slope * (IN_RANGE - OUT_RANGE)
    cks = {ck: round(model["pair"].get((key, ck), 0.0) + q, 2)
           for (lk, ck) in model["pair"] if lk == key}
    text = []
    if abs(q) >= 0.15:
        text.append(f"{'+' if q > 0 else ''}{q:.1f}★ against the average LoRA"
                    + (" - a well made LoRA" if q >= 0.4 else " - a weak LoRA" if q <= -0.4 else ""))
    else:
        text.append("about as good as the average LoRA")
    if abs(range_gap) >= 0.3:
        text.append(f"prompts like its own score {abs(range_gap):.1f}★ {'higher' if range_gap > 0 else 'lower'} than "
                    f"prompts outside its range" + (" - judge it on its own kind of prompt" if range_gap > 0 else ""))
    for ck, v in sorted(cks.items(), key=lambda kv: -kv[1]):
        if len(cks) > 1 and abs(v - q) >= 0.3:
            text.append(f"on {ck}: {'+' if v >= 0 else ''}{v:.1f}★ ({'better' if v > q else 'worse'} than elsewhere)")
    return {"quality": round(q, 2), "n": round(model["n"].get(key, 0.0), 1), "slope": round(slope, 2),
            "range_gap": round(range_gap, 2), "by_checkpoint": cks, "text": text}


_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def current(connection: sqlite3.Connection, catalog_path: str) -> Dict[str, Any]:
    """The fitted model, refitted only when the marks changed (count + newest mark time)."""
    stamp = tuple(connection.execute("select count(*), max(marked_at) from stacks").fetchone())
    cached = _cache.get(catalog_path)
    if cached and cached[0] == stamp:
        return cached[1]
    model = fit_model(observations(connection))
    _cache[catalog_path] = (stamp, model)
    return model


def update(catalog_path: str, flaw_stats=None, log=print) -> Dict[str, int]:
    """Refit and rewrite lora_observed. flaw_stats(connection) -> {key: (flaws, praise)} (stacks.tag_stats)."""
    started = time.time()
    connection = sqlite3.connect(catalog_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(TABLE)
        model = current(connection, catalog_path)
        names: Dict[str, Tuple[str, str]] = {}
        for o in observations(connection):
            if o["lora"]:
                names[o["lora"]] = (o.get("path") or "", o.get("name") or "")
        marks: Dict[str, int] = collections.Counter()
        for row in connection.execute("select loras from stacks"):
            for entry in json.loads(row["loras"] or "[]"):
                if not entry.get("panel"):
                    marks[lora_key(entry)] += 1
        tags = flaw_stats(connection) if flaw_stats else {}
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        connection.execute("DELETE FROM lora_observed")
        rows = 0
        for key, q in model["quality"].items():
            info = explain(model, key)
            flaws, praise = tags.get(key, ({}, {}))
            connection.execute("INSERT OR REPLACE INTO lora_observed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (key, names.get(key, ("", ""))[0], names.get(key, ("", ""))[1], info["quality"], info["n"],
                                marks.get(key, 0), info["slope"], json.dumps(info["by_checkpoint"]),
                                json.dumps(flaws), json.dumps(praise), now))
            rows += 1
        # stack synergy (2026-10-27)
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
    log(f"[marks] model: {rows} LoRA(s), mean {model['mu']:.2f}★, fit slope {model['B']:+.2f} "
        f"({time.time() - started:.1f}s)")
    return {"loras": rows}
