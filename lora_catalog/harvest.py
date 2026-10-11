"""
Stack harvest: which LoRAs people actually use TOGETHER, from Civitai's community images (2026-10-09).

For every LoRA you own that has a Civitai version id, fetch its top community images (most reactions, all time,
metadata only - no image is downloaded) and record the LoRAs each image used, with their weights. Then count
pairs across all harvested images.

    python run.py harvest --only Krea2              fetch (resumable: LoRAs already fetched are skipped; --force)
    python run.py harvest --only Krea2 --limit 5    a few, to test
    python run.py pairs                             recount lora_pairs from everything harvested
    python run.py pairs --show vallejo              a LoRA's best partners (owned and not owned)
    python run.py pairs --wishlist                  LoRAs you do not own that pair well with ones you do

Where a co-used LoRA comes from in an image record (meta, needs withMeta=true):
    civitaiResources   [{type, weight, modelVersionId}]           on-site generations and newer uploads
    resources          [{type, name, weight, hash}]               A1111/Forge style; hash = AutoV2 = sha256[:10]
    additionalResources [{name: "urn:air:...:civitai:<model>@<version>", strength}]
    modelVersionIds    top-level list, untyped (used only for versions we know are LoRAs)
A resource is matched to your files by version id, else by the AutoV2 hash against the catalog's sha256 (which is why
`run.py hash` matters). Identity key: "v:<versionId>", or "h:<autov2>" when only a hash is known.

Tables
    harvest_seeds    version_id, path, fetched_at, images, with_loras, status
    harvest_images   image_id, base_model, nsfw_level, reactions, created_at, username, post_id, loras (JSON
                     [{key, name, weight}]), fetched_at
    lora_pairs       a, b (keys, a < b), base_model, n (distinct users who used both), reactions (sum), lift, score,
                     weights_a / weights_b (JSON lists), examples (image ids), owned_a, owned_b
Counting is by PEOPLE, not images (2026-10-09): one user posting 20 images with the same stack is one vote, so n,
n_a, n_b and N count distinct users (username, else the post). lift = n_ab * N / (n_a * n_b) over the harvested
images of that base model (how much more often the two meet than their own frequencies predict - a detailer used with
everything gets no credit just for being common), score = log2(lift) * log(1 + n_ab), only n_ab >= 2 users.
On-site training outputs ("training_<id>-<date>", unpublished) are counted but left out of the wishlist. Reactions are kept for display, not in the score: popularity is
not your taste. Your own good / bad stacks (later) outrank all of this.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Settings
from .crawl import CivitaiClient


def person(name: str) -> str:            # same one-way hash as community.person (no import cycle)
    import hashlib
    return hashlib.sha1(("lora-catalog:" + str(name or "")).encode("utf-8")).hexdigest()[:12] if name else ""

LORA_TYPES = {"lora", "locon", "lycoris", "dora", "loha", "lokr"}
AIR = re.compile(r"civitai:(\d+)@(\d+)", re.I)
TRAINING = re.compile(r"^training_\d+-\d+", re.I)     # Civitai on-site trainer outputs, usually unpublished

TABLES = """
CREATE TABLE IF NOT EXISTS harvest_seeds (
    version_id INTEGER PRIMARY KEY, path TEXT, fetched_at TEXT, images INTEGER, with_loras INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS harvest_images (
    image_id INTEGER PRIMARY KEY, base_model TEXT, nsfw_level TEXT, reactions INTEGER, created_at TEXT,
    username TEXT, post_id INTEGER, loras TEXT, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS lora_pairs (
    a TEXT, b TEXT, base_model TEXT, n INTEGER, reactions INTEGER, lift REAL, score REAL,
    weights_a TEXT, weights_b TEXT, examples TEXT, owned_a INTEGER, owned_b INTEGER,
    PRIMARY KEY (a, b, base_model));
CREATE INDEX IF NOT EXISTS lora_pairs_a ON lora_pairs(a);
CREATE INDEX IF NOT EXISTS lora_pairs_b ON lora_pairs(b);
"""


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(TABLES)
    _hash_usernames(connection)


def _hash_usernames(connection: sqlite3.Connection) -> None:
    """Privacy (2026-10-10, LORA_STUDIO_SPEC 5.6): usernames are kept only as a short one-way hash - enough to count
    distinct people, which is all the pair counting needs. Converts rows stored in clear by earlier runs, once."""
    rows = connection.execute("select image_id, username from harvest_images where username != '' and "
                              "(length(username) != 12 or username glob '*[^0-9a-f]*')").fetchall()
    for image_id, username in rows:
        connection.execute("update harvest_images set username = ? where image_id = ?", (person(username), image_id))
    if rows:
        connection.commit()
        print(f"[harvest] {len(rows)} usernames replaced by one-way hashes", flush=True)


class LocalIndex:
    """Your LoRAs by Civitai version id and by AutoV2 hash (sha256[:10])."""

    def __init__(self, connection: sqlite3.Connection):
        self.by_version: Dict[int, sqlite3.Row] = {}
        self.by_hash: Dict[str, sqlite3.Row] = {}
        for row in connection.execute("select path, sha256, civitai_version_id, model_name, stem, base_model, "
                                      "arch_folder from loras"):
            if row["civitai_version_id"]:
                self.by_version[int(row["civitai_version_id"])] = row
            if row["sha256"]:
                self.by_hash[str(row["sha256"]).lower()[:10]] = row

    def key_for(self, version_id: Optional[int], hash10: Optional[str]) -> Optional[str]:
        if version_id:
            return f"v:{int(version_id)}"
        if hash10:
            row = self.by_hash.get(hash10.lower()[:10])
            if row is not None and row["civitai_version_id"]:
                return f"v:{int(row['civitai_version_id'])}"
            return f"h:{hash10.lower()[:10]}"
        return None

    def owned(self, key: str) -> Optional[sqlite3.Row]:
        kind, _sep, value = key.partition(":")
        if kind == "v":
            return self.by_version.get(int(value))
        return self.by_hash.get(value)


def image_loras(item: Dict[str, Any], local: LocalIndex) -> List[Dict[str, Any]]:
    """The LoRAs one image used: [{key, name, weight}], one entry per LoRA (highest weight kept)."""
    meta = item.get("meta") or {}
    found: Dict[str, Dict[str, Any]] = {}

    def add(key: Optional[str], name: str, weight: Any) -> None:
        if not key:
            return
        try:
            weight = float(weight) if weight is not None else None
        except (TypeError, ValueError):
            weight = None
        old = found.get(key)
        if old is None or (weight is not None and (old["weight"] is None or abs(weight) > abs(old["weight"]))):
            found[key] = {"key": key, "name": name or (old or {}).get("name", ""), "weight": weight}

    for res in meta.get("civitaiResources") or []:
        if str(res.get("type", "")).lower() in LORA_TYPES:
            add(local.key_for(res.get("modelVersionId"), None), str(res.get("modelVersionName") or ""), res.get("weight"))
    for res in meta.get("resources") or []:
        if str(res.get("type", "")).lower() in LORA_TYPES:
            add(local.key_for(None, str(res.get("hash") or "") or None), str(res.get("name") or ""), res.get("weight"))
    for res in meta.get("additionalResources") or []:
        match = AIR.search(str(res.get("name") or ""))
        if match and str(res.get("type", "lora")).lower() in LORA_TYPES:
            add(local.key_for(int(match.group(2)), None), "", res.get("strength", res.get("weight")))
    for version_id in item.get("modelVersionIds") or []:          # untyped: only versions we know to be LoRAs
        key = f"v:{int(version_id)}"
        if key not in found and local.owned(key) is not None:
            add(key, "", None)
    return list(found.values())


def _reactions(stats: Dict[str, Any]) -> int:
    return int(sum(int(stats.get(k) or 0) for k in ("likeCount", "heartCount", "laughCount", "cryCount")))


def harvest(settings: Settings, only: str = "", limit: int = 0, force: bool = False) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=120)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    local = LocalIndex(connection)
    where, params = "civitai_version_id is not null", []
    if only:
        where += " and rel_dir like ?"
        params.append(only.replace("\\", "/") + "%")
    seeds = list(connection.execute(f"select path, civitai_version_id from loras where {where} order by rel_dir", params))
    done = {row[0] for row in connection.execute("select version_id from harvest_seeds where status = 'ok'")}
    todo = [s for s in seeds if force or int(s["civitai_version_id"]) not in done]
    already = len(seeds) - len(todo)
    if limit:
        todo = todo[:limit]
    client = CivitaiClient(settings)
    print(f"[harvest] {len(todo)} LoRAs to fetch ({already} already done) from {client.api}, "
          f"top {settings.harvest_images} images, nsfw up to {settings.harvest_nsfw}; "
          f"API key {'found' if client.has_key else 'NOT found'}", flush=True)
    counts = {"seeds": 0, "images": 0, "images_with_2plus": 0, "failed": 0}
    started = time.time()
    for number, seed in enumerate(todo, 1):
        version_id = int(seed["civitai_version_id"])
        data = client.get_json("images", {"modelVersionId": version_id, "limit": min(100, settings.harvest_images),
                                          "sort": "Most Reactions", "period": "AllTime",
                                          "nsfw": settings.harvest_nsfw, "withMeta": "true"})
        stamp = time.strftime("%Y-%m-%d %H:%M")
        if data is None:
            counts["failed"] += 1
            connection.execute("insert or replace into harvest_seeds values (?,?,?,?,?,?)",
                               (version_id, seed["path"], stamp, 0, 0, "failed"))
            continue
        items = data.get("items") or []
        with_loras = 0
        for item in items:
            loras = image_loras(item, local)
            if len(loras) >= 2:
                with_loras += 1
            connection.execute(
                "insert or replace into harvest_images values (?,?,?,?,?,?,?,?,?)",
                (int(item["id"]), item.get("baseModel") or "", str(item.get("nsfwLevel") or ""),
                 _reactions(item.get("stats") or {}), item.get("createdAt") or "", person(item.get("username") or ""),
                 item.get("postId"), json.dumps(loras, ensure_ascii=False), stamp))
        connection.execute("insert or replace into harvest_seeds values (?,?,?,?,?,?)",
                           (version_id, seed["path"], stamp, len(items), with_loras, "ok"))
        counts["seeds"] += 1
        counts["images"] += len(items)
        counts["images_with_2plus"] += with_loras
        if number % 10 == 0 or number == len(todo):
            connection.commit()
            rate = (time.time() - started) / number
            print(f"[harvest] {number}/{len(todo)}  {counts['images']} images, {counts['images_with_2plus']} with 2+ LoRAs  "
                  f"eta {(len(todo) - number) * rate / 60:.0f} min", flush=True)
    connection.commit()
    connection.close()
    counts["live_calls"] = client.live_calls
    counts["seconds"] = int(time.time() - started)
    return counts


def _weights_summary(values: Iterable[Optional[float]]) -> List[float]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return []
    return [round(clean[0], 2), round(statistics.median(clean), 2), round(clean[-1], 2)]    # min, median, max


def count_pairs(settings: Settings, min_n: int = 2) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=120)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    local = LocalIndex(connection)
    by_base: Dict[str, List[Tuple[int, int, List[Dict[str, Any]], str]]] = defaultdict(list)
    for row in connection.execute("select image_id, base_model, reactions, loras, username, post_id from harvest_images"):
        loras = json.loads(row["loras"] or "[]")
        if loras:
            who = row["username"] or f"post:{row['post_id'] or row['image_id']}"
            by_base[row["base_model"] or ""].append((row["image_id"], row["reactions"] or 0, loras, who))
    connection.execute("delete from lora_pairs")
    written = 0
    for base, images in by_base.items():
        total = len({who for *_rest, who in images})                  # distinct users
        single: Dict[str, set] = defaultdict(set)
        pair_users: Dict[Tuple[str, str], set] = defaultdict(set)
        pair_r: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_w: Dict[Tuple[str, str], Tuple[List, List]] = defaultdict(lambda: ([], []))
        pair_ex: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for image_id, reactions, loras, who in images:
            weight = {entry["key"]: entry.get("weight") for entry in loras}
            keys = sorted(weight)
            for key in keys:
                single[key].add(who)
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    first_from_user = who not in pair_users[(a, b)]
                    pair_users[(a, b)].add(who)
                    pair_r[(a, b)] += reactions
                    if first_from_user:               # weights and examples: one per user too
                        pair_w[(a, b)][0].append(weight[a])
                        pair_w[(a, b)][1].append(weight[b])
                        if len(pair_ex[(a, b)]) < 5:
                            pair_ex[(a, b)].append(image_id)
        for (a, b), users in pair_users.items():
            n = len(users)
            if n < min_n:
                continue
            lift = n * total / (len(single[a]) * len(single[b]))
            score = math.log2(lift) * math.log(1 + n) if lift > 0 else 0.0
            connection.execute(
                "insert or replace into lora_pairs values (?,?,?,?,?,?,?,?,?,?,?,?)",
                (a, b, base, n, pair_r[(a, b)], round(lift, 3), round(score, 3),
                 json.dumps(_weights_summary(pair_w[(a, b)][0])), json.dumps(_weights_summary(pair_w[(a, b)][1])),
                 json.dumps(pair_ex[(a, b)]), int(local.owned(a) is not None), int(local.owned(b) is not None)))
            written += 1
    connection.commit()
    connection.close()
    return {"base_models": len(by_base), "images": sum(len(v) for v in by_base.values()), "pairs": written}


# ----------------------------------------------------------------------------------------------- reading
def _names(connection: sqlite3.Connection, settings: Settings, keys: Iterable[str], local: LocalIndex,
           lookup_missing: bool = False) -> Dict[str, str]:
    """Display names: your catalog for owned LoRAs, names seen in image metadata, then (optionally) the cached
    Civitai version record."""
    names: Dict[str, str] = {}
    seen: Dict[str, str] = {}
    for row in connection.execute("select loras from harvest_images"):
        for entry in json.loads(row["loras"] or "[]"):
            if entry.get("name") and entry["key"] not in seen:
                seen[entry["key"]] = entry["name"]
    client = CivitaiClient(settings) if lookup_missing else None
    for key in keys:
        row = local.owned(key)
        if row is not None:
            names[key] = row["model_name"] or row["stem"]
        elif key in seen:
            names[key] = seen[key]
        elif client is not None and key.startswith("v:"):
            version = client.get("v", int(key[2:])) or {}
            model = (version.get("model") or {}).get("name") or ""
            names[key] = f"{model} - {version.get('name') or ''}".strip(" -") or key
        else:
            names[key] = key
    return names


def show(settings: Settings, word: str, top: int = 25) -> str:
    connection = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    local = LocalIndex(connection)
    word = word.lower()
    mine = [row for row in local.by_version.values()
            if word in f"{row['model_name'] or ''} {row['stem'] or ''}".lower()]
    if not mine:
        return f"no LoRA in the catalog matches '{word}'"
    lines = []
    for row in mine[:5]:
        key = f"v:{int(row['civitai_version_id'])}"
        rows = list(connection.execute(
            "select * from lora_pairs where a = ? or b = ? order by score desc limit ?", (key, key, top)))
        partners = [(r["b"] if r["a"] == key else r["a"], r) for r in rows]
        names = _names(connection, settings, [p for p, _ in partners], local, lookup_missing=True)
        seed = connection.execute("select images, with_loras from harvest_seeds where version_id = ?",
                                  (int(row["civitai_version_id"]),)).fetchone()
        lines.append(f"\n== {row['model_name'] or row['stem']}  ({key})  harvested: "
                     + (f"{seed['images']} images, {seed['with_loras']} with 2+ LoRAs" if seed else "not yet"))
        for partner, r in partners:
            mine_w = r["weights_a"] if r["a"] == key else r["weights_b"]
            their_w = r["weights_b"] if r["a"] == key else r["weights_a"]
            owned = "owned" if local.owned(partner) is not None else "NOT OWNED"
            lines.append(f"  {r['score']:6.2f}  n={r['n']:<3} lift={r['lift']:<6} {owned:9}  {names[partner][:60]}"
                         f"   weights this {mine_w} / that {their_w}")
        if not partners:
            lines.append("  no partner used together with it by 2+ different people yet")
    connection.close()
    return "\n".join(lines)


def wishlist(settings: Settings, top: int = 40, base: str = "", plain: bool = False) -> str:
    """LoRAs you do not own that pair well with ones you do. Rank = sum of pair scores x (0.5 + community score / 100):
    a LoRA the wider audience rates at the top of its base model counts 1.5x, one at the bottom 0.5x (spec 5.4).
    The score is community.quick_scores (approval, likes per download, downloads per day; one cached call per
    version, for the 3 x top best pair sums only). base: only pairs counted for that base model (part of its name).
    plain: pair sums only, no Civitai calls."""
    connection = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    local = LocalIndex(connection)
    total: Dict[str, float] = defaultdict(float)
    partners: Dict[str, List[str]] = defaultdict(list)
    for r in connection.execute("select * from lora_pairs where owned_a + owned_b = 1 and score > 0"):
        if base and base.lower() not in (r["base_model"] or "").lower():
            continue
        theirs, ours = (r["b"], r["a"]) if r["owned_a"] else (r["a"], r["b"])
        total[theirs] += r["score"]
        partners[theirs].append(ours)
    seen = _names(connection, settings, list(total), local)
    total = {k: v for k, v in total.items() if not TRAINING.match(seen.get(k, ""))}    # unpublished on-site training
    pool = sorted(total, key=lambda k: -total[k])[:top if plain else top * 3]
    quick: Dict[int, Dict[str, Any]] = {}
    if not plain:
        from .community import quick_scores
        quick = quick_scores(settings, [int(k[2:]) for k in pool if k.startswith("v:")])

    def info(key: str) -> Dict[str, Any]:
        return quick.get(int(key[2:]), {}) if key.startswith("v:") else {}

    def rank(key: str) -> float:
        score = info(key).get("score")
        return total[key] * (0.5 + (score if score is not None else 50.0) / 100.0)

    best = sorted(pool, key=lambda k: -rank(k))[:top]
    names = _names(connection, settings, best + [p for k in best for p in partners[k][:3]], local, lookup_missing=True)
    head = "sum of pair scores" if plain else "pair sum x (0.5 + Civitai score/100)"
    lines = [f"LoRAs you do not own that pair well with ones you do ({head}){f', base model ~ {base}' if base else ''}, "
             f"top {len(best)}:"]
    for key in best:
        host = str(getattr(settings, "civitai_host", "") or "https://civitai.com").rstrip("/")
        link = f"{host}/model-versions/{key[2:]}" if key.startswith("v:") else key
        i = info(key)
        extra = ""
        if i:
            extra = (f"  [{i['base_model']}, pairs {total[key]:.2f}, Civitai "
                     + (f"{i['score']:.0f}" if i.get("score") is not None else "?")
                     + f", {i['downloads']:,} dl, {i['up']} up / {i['down']} down]")
        lines.append(f"  {(total[key] if plain else rank(key)):6.2f}  {names[key][:60]}  ({link}){extra}")
        lines.append("          with: " + "; ".join(names[p][:40] for p in partners[key][:3]))
    connection.close()
    return "\n".join(lines)
