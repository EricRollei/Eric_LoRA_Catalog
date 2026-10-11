"""
What the wider Civitai audience makes of each LoRA, and what it makes WITH it (2026-10-10).
Spec: LORA_STUDIO_SPEC.md section 5 (Eric's answers: score within base model + role; test prompts keep every NSFW
level, labelled; tag-style prompts kept and marked; built inside the catalog now).

    python run.py community --only Krea2            fetch (skips LoRAs fetched in the last 30 days; --force; --limit 5)
    python run.py community --score                 recompute the community scores only (no network)
    python run.py community --show "Mucha"          what was collected for one LoRA
    python run.py identify                          find Civitai ids for files that came without one, by sha256

Per LoRA (its Civitai version), four calls at most:
    models/{modelId}          downloads, thumbs up / down, comments, tips - model and version level (one call per model)
    images Most Reactions 50  the hits: test prompts, weights, nsfw levels
    images Newest 2 x 100     an unbiased sample: adoption (distinct posters), median reactions, dislikes, nsfw share
Only metadata; no image is downloaded. Usernames are only ever stored as a short hash (to count people).

Tables
    community_stats    one row per version: the numbers below + score, score_group, fetched_at
    community_history  version_id, fetched_at, downloads, thumbs_up, thumbs_down - one row per refresh (trends)
    community_prompts  image_id, version_id, source (creator|top|newest), prompt, cleaned, negative, weight,
                       base_model, prompt_style (tags|natural), nsfw_level, reactions, picked (the test prompts)

The numbers
    approval     (up + 1) / (up + down + 2)                  - few votes stay near neutral
    engagement   thumbs up per 1,000 downloads                - liked it enough to press the button
    momentum     version downloads per day since published    - a good new LoRA is not buried by old ones
    adoption     distinct people in the newest-200 sample      - downloads are not use
    reception    median reactions in the newest sample, minus a dislike penalty
    score        0-100: mean of the five percentiles, re-ranked, WITHIN base model + card role (your Krea2 styles are
                 compared with Krea2 styles); a group under 8 LoRAs falls back to the base model alone. A popularity
                 prior, never a verdict: your stars win everywhere it is used.
    weights      median and 25-75 % of the weights people used (civitaiResources / resources / <lora:name:w> in the
                 prompt), with the sample size - the card's community weight range when n >= 10
    nsfw_share   share of the newest sample at each level (at most 5 images per person counted in the shares and in
                 reception, so one prolific poster does not speak for everyone); explicit_unasked = Mature/X images whose prompt has no
                 nudity or sex words (a LoRA that pushes nudity by itself) - a heuristic

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Settings
from .crawl import CACHE_DIR, CivitaiClient
from .harvest import LocalIndex, image_loras, person

TABLES = """
CREATE TABLE IF NOT EXISTS community_stats (
    version_id INTEGER PRIMARY KEY, model_id INTEGER, base_model TEXT, role TEXT, fetched_at TEXT,
    published_at TEXT, downloads INTEGER, thumbs_up INTEGER, thumbs_down INTEGER, model_downloads INTEGER,
    model_thumbs_up INTEGER, model_thumbs_down INTEGER, comments INTEGER, tips INTEGER,
    images_sampled INTEGER, distinct_posters INTEGER, sample_capped INTEGER, median_reactions REAL, dislike_share REAL,
    nsfw_share TEXT, explicit_unasked REAL, weight_n INTEGER, weight_p25 REAL, weight_median REAL, weight_p75 REAL,
    approval REAL, engagement REAL, momentum REAL, score REAL, score_group TEXT, newer_version TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS community_history (
    version_id INTEGER, fetched_at TEXT, downloads INTEGER, thumbs_up INTEGER, thumbs_down INTEGER,
    PRIMARY KEY (version_id, fetched_at));
CREATE TABLE IF NOT EXISTS community_prompts (
    image_id INTEGER, version_id INTEGER, source TEXT, prompt TEXT, cleaned TEXT, negative TEXT, weight REAL,
    base_model TEXT, prompt_style TEXT, nsfw_level TEXT, reactions INTEGER, picked INTEGER DEFAULT 0,
    PRIMARY KEY (version_id, source, image_id));
CREATE INDEX IF NOT EXISTS community_prompts_v ON community_prompts(version_id, picked);
"""
LEVELS = ["None", "Soft", "Mature", "X"]
NSFW_WORDS = re.compile(r"\b(nude|nudity|naked|nsfw|topless|bottomless|nipples?|areola|breasts?|boobs?|tits|pussy|"
                        r"vagina|vulva|penis|cock|dick|genitals?|sex|sexual|intercourse|cum|orgasm|erotic|porn|"
                        r"explicit|lingerie|underwear|bikini|undress(ed|ing)?|bare (chest|skin|breasts?)|spread legs)\b", re.I)
LORA_TAG = re.compile(r"<(lora|lyco|lycoris|locon):([^:>]+)(?::([-\d.]+))?(?::[^>]*)?>", re.I)
TAG_STYLE = re.compile(r"\b(score_\d|1girl|1boy|2girls|solo|masterpiece|best quality|rating_\w+|source_\w+|"
                       r"absurdres|highres)\b", re.I)
CREATOR_PER_LORA, TOP_PER_LORA, NEWEST_PER_LORA = 3, 4, 1     # test prompts picked: up to 8


# Added 2026-10-10 (peer comparison). stats_version 2 = a row that has them; older rows are fetched again.
NEW_COLUMNS = {"images_per_day": "REAL", "images_exact": "INTEGER", "top_reactions_median": "REAL",
               "top_reactions_total": "INTEGER", "top_sampled": "INTEGER", "peer": "TEXT", "stats_version": "INTEGER"}
STATS_VERSION = 2
GONE_RETRY_DAYS = 90       # a model Civitai answered 404 for is asked again after this many days (or with --force)
# What each LoRA is compared on, within its platform (base model): (column, label)
PEER_FEATURES = [("downloads", "downloads"), ("engagement", "likes per download"),
                 ("images_per_day", "images posted per day"), ("top_reactions_median", "reactions on its top 50 images")]


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(TABLES)
    have = {r[1] for r in connection.execute("pragma table_info(community_stats)")}
    for column, kind in NEW_COLUMNS.items():
        if column not in have:
            connection.execute(f"alter table community_stats add column {column} {kind}")


def _when(item: Dict[str, Any]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(item.get("createdAt") or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def image_rate(newest: List[Dict[str, Any]], published: str) -> Tuple[Optional[float], Optional[int]]:
    """(images posted per day, exact image count or None). Under 200 in the Newest sample = we have them all: the count
    is exact and the rate is count / days since published. 200 (capped) = the rate is the sample over the days it spans,
    i.e. how busy it is NOW; the total is unknown (the API gives no count)."""
    dates = sorted(d for d in (_when(i) for i in newest) if d)
    capped = len(newest) >= 200
    if not capped:
        try:
            since = datetime.fromisoformat(published.replace("Z", "+00:00"))
            days = max(7.0, (datetime.now(timezone.utc) - since).total_seconds() / 86400)
        except ValueError:
            days = None
        return (round(len(newest) / days, 3) if days else None), len(newest)
    if len(dates) < 2:
        return None, None
    span = max(1.0, (datetime.now(timezone.utc) - dates[0]).total_seconds() / 86400)   # oldest in sample -> now
    return round(len(dates) / span, 3), None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


# -- prompts ------------------------------------------------------------------------------------------------------------
def clean_prompt(text: str) -> str:
    """Strip LoRA tags, embeddings and weight syntax; keep the words (trigger words included)."""
    text = LORA_TAG.sub(" ", str(text or ""))
    text = re.sub(r"<[^>]{1,80}>", " ", text)                       # hypernets, other angle tags
    text = re.sub(r"\bembedding:[\w.-]+", " ", text, flags=re.I)
    text = re.sub(r"\bBREAK\b", ",", text)
    for _ in range(3):                                                 # (word:1.2) -> word, ((word)) -> word, [word] -> word
        text = re.sub(r"\(([^()]*?):\s*-?[\d.]+\)", r"\1", text)
        text = re.sub(r"[()\[\]{}]", "", text)
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip(" ,\n")
    return text[:1500]


def prompt_style(text: str) -> str:
    """'tags' (Pony / Illustrious: short comma-separated tokens) or 'natural' (sentences: Flux / Krea2 / Qwen)."""
    if TAG_STYLE.search(text or ""):
        return "tags"
    parts = [p.strip() for p in re.split(r"[,\n]", text or "") if p.strip()]
    if len(parts) >= 6 and statistics.mean(len(p.split()) for p in parts) < 3.2:
        return "tags"
    return "natural"


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())}


def _similar(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) > 0.6


def _float(text: Any) -> Optional[float]:
    """A weight from a prompt tag; None for typos like '0.0.7', '..8' or '0-1'."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _weight_for(item: Dict[str, Any], version_id: int, names: List[str], local: LocalIndex) -> Optional[float]:
    """The weight this image used THIS LoRA at: from the resource lists, else a <lora:name:w> tag naming it."""
    key = f"v:{version_id}"
    for entry in image_loras(item, local):
        if entry["key"] == key and _float(entry.get("weight")) is not None:
            return _float(entry["weight"])
    tags = LORA_TAG.findall(str((item.get("meta") or {}).get("prompt") or ""))
    wanted = {n.lower() for n in names if n}
    for _kind, name, weight in tags:
        if weight and name.strip().lower() in wanted and _float(weight) is not None:
            return _float(weight)
    if len(tags) == 1 and tags[0][2] and version_id in (item.get("modelVersionIds") or []):
        return _float(tags[0][2])                                      # the only LoRA tag, and Civitai says it is this one
    return None


def _reactions(stats: Dict[str, Any]) -> int:
    return int(sum(int(stats.get(k) or 0) for k in ("likeCount", "heartCount", "laughCount", "cryCount")))


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (position - low), 3)


# -- fetching -----------------------------------------------------------------------------------------------------------
class Collector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = CivitaiClient(settings)
        self.models: Dict[int, Optional[Dict[str, Any]]] = {}
        self.missing: Dict[int, Optional[int]] = {}    # model id -> HTTP status when there was no record

    def model(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Fresh model record (stats change), written back to the crawl cache so the index sees new versions too."""
        if model_id not in self.models:
            data = self.client.get_json(f"models/{model_id}", {})
            if not data:
                self.missing[model_id] = self.client.last_status
            if data:
                try:
                    with open(os.path.join(CACHE_DIR, f"m{model_id}.json"), "w", encoding="utf-8") as handle:
                        json.dump(data, handle, ensure_ascii=False)
                except OSError:
                    pass
            self.models[model_id] = data
        return self.models[model_id]

    def images(self, version_id: int, sort: str, pages: int, per_page: int) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        params = {"modelVersionId": version_id, "limit": per_page, "sort": sort, "period": "AllTime",
                  "nsfw": self.settings.harvest_nsfw, "withMeta": "true"}
        for _page in range(pages):
            data = self.client.get_json("images", params)
            if not data:
                break
            items += data.get("items") or []
            cursor = (data.get("metadata") or {}).get("nextCursor")
            if not cursor or len(data.get("items") or []) < per_page:
                break
            params = dict(params, cursor=cursor)
        return items


def _version_facts(model: Dict[str, Any], version_id: int) -> Tuple[Dict[str, Any], str]:
    """(this version's record inside the model, the name of a newer version of the same model with the same base
    model, or '')."""
    versions = model.get("modelVersions") or []
    mine = next((v for v in versions if v.get("id") == version_id), {})
    newer, newest_at = "", ""
    if mine:
        for v in versions:
            published = v.get("publishedAt") or ""
            if (v.get("baseModel") == mine.get("baseModel") and published > (mine.get("publishedAt") or "")
                    and v.get("id") != version_id and _successor(mine.get("name"), v.get("name"))
                    and published > newest_at):
                newer, newest_at = f"{v.get('name')} ({v.get('id')})", published
    return mine, newer


VERSION_TOKEN = re.compile(r"\b(v|ver|version)?\s*\d+(\.\d+)*[a-z]?\b", re.I)


SIBLING_WORDS = {"variant", "option", "part", "alt", "set", "pack", "vol", "volume", "chapter", "strong", "soft"}
PURE_VERSION = re.compile(r"^\s*(v|ver|version)?\s*\d+(\.\d+)*\s*$", re.I)


def _version_base(name: Any) -> set:
    return set(re.findall(r"[a-z]{2,}", VERSION_TOKEN.sub(" ", str(name or "")).lower()))


def _successor(old: Any, new: Any) -> bool:
    """Is `new` a later version of the same thing, not another member of a collection (Moebius vs Ma-Ko, Strong vs
    Soft)? Same words once version numbers are removed, or one name's words inside the other's."""
    a, b = _version_base(old), _version_base(new)
    if a & SIBLING_WORDS or b & SIBLING_WORDS:
        return False                                  # Variant 1 / Variant 4, Part 1 / Part 2: siblings, not updates
    if not a and not b:                               # bare numbers: v1.0 -> v2.0 yes, 1980s -> 2010s no
        return all(PURE_VERSION.match(str(n or "")) for n in (old, new))
    if a == b:
        return True
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return bool(small) and small <= big


def refresh_newer(connection: sqlite3.Connection) -> int:
    """Recompute newer_version from the cached model records (no network). Returns how many are flagged."""
    flagged = 0
    for version_id, model_id in connection.execute(
            "select version_id, model_id from community_stats where status = 'ok' and model_id is not null").fetchall():
        try:
            with open(os.path.join(CACHE_DIR, f"m{model_id}.json"), "r", encoding="utf-8") as handle:
                model = json.load(handle)
        except (OSError, ValueError):
            continue
        _mine, newer = _version_facts(model, int(version_id))
        connection.execute("update community_stats set newer_version = ? where version_id = ?", (newer, version_id))
        flagged += bool(newer)
    return flagged


def collect_one(collector: Collector, connection: sqlite3.Connection, local: LocalIndex, row: sqlite3.Row) -> str:
    version_id, model_id = int(row["civitai_version_id"]), row["civitai_model_id"]
    model = collector.model(int(model_id)) if model_id else None
    if not model:
        # 404 = the model was deleted from Civitai: remembered, so refreshes skip it for GONE_RETRY_DAYS (2026-10-11).
        # Anything else (timeout, 5xx, rate limit) stays 'no model record' and is tried again next time.
        status = "not on civitai" if model_id and collector.missing.get(int(model_id)) == 404 else "no model record"
        connection.execute("insert or replace into community_stats (version_id, model_id, fetched_at, status) "
                           "values (?, ?, ?, ?)", (version_id, model_id, _now(), status))
        return status
    version, newer = _version_facts(model, version_id)
    vstats, mstats = version.get("stats") or {}, model.get("stats") or {}
    top = collector.images(version_id, "Most Reactions", 1, 50)
    newest = collector.images(version_id, "Newest", 2, 100)
    names = [row["stem"], row["model_name"], os.path.splitext(row["file_name"] or "")[0]]

    # the unbiased sample: adoption, reception, nsfw, unasked explicit
    people = {person(i.get("username") or f"post{i.get('postId')}") for i in newest}
    # one prolific poster must not speak for everyone: at most 5 images per person in the shares and reception
    per_person: Dict[str, int] = {}
    fair = []
    for i in newest:
        who = person(i.get("username") or f"post{i.get('postId')}")
        per_person[who] = per_person.get(who, 0) + 1
        if per_person[who] <= 5:
            fair.append(i)
    reactions = [_reactions(i.get("stats") or {}) for i in fair]
    dislikes = sum(int((i.get("stats") or {}).get("dislikeCount") or 0) for i in fair)
    likes = sum(reactions) or 0
    levels = [str(i.get("nsfwLevel") or "None") for i in fair]
    nsfw_share = {level: round(levels.count(level) / len(levels), 3) for level in LEVELS} if levels else {}
    hot = [i for i in fair if str(i.get("nsfwLevel")) in ("Mature", "X")]
    unasked = [i for i in hot if not NSFW_WORDS.search(str((i.get("meta") or {}).get("prompt") or ""))]
    weights = [w for w in (_weight_for(i, version_id, names, local) for i in top + newest) if w is not None]
    top_reactions = [_reactions(i.get("stats") or {}) for i in top]

    published = version.get("publishedAt") or version.get("createdAt") or ""
    try:
        days = max(7.0, (datetime.now(timezone.utc) - datetime.fromisoformat(published.replace("Z", "+00:00"))).days)
    except ValueError:
        days = None
    downloads = int(vstats.get("downloadCount") or 0)
    up, down = int(vstats.get("thumbsUpCount") or 0), int(vstats.get("thumbsDownCount") or 0)
    if downloads < 50:                                   # a tiny version: judge by the model as a whole
        downloads = int(mstats.get("downloadCount") or downloads)
        up, down = int(mstats.get("thumbsUpCount") or up), int(mstats.get("thumbsDownCount") or down)
    facet = connection.execute("select role from card_facets where path = ?", (row["path"],)).fetchone()
    connection.execute("""
        insert or replace into community_stats (version_id, model_id, base_model, role, fetched_at, published_at,
            downloads, thumbs_up, thumbs_down, model_downloads, model_thumbs_up, model_thumbs_down, comments, tips,
            images_sampled, distinct_posters, sample_capped, median_reactions, dislike_share, nsfw_share,
            explicit_unasked, weight_n, weight_p25, weight_median, weight_p75, approval, engagement, momentum,
            newer_version, status)
        values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (version_id, model_id, row["base_model"] or version.get("baseModel") or "", (facet[0] if facet else "") or "",
         _now(), published, int(vstats.get("downloadCount") or 0), int(vstats.get("thumbsUpCount") or 0),
         int(vstats.get("thumbsDownCount") or 0), int(mstats.get("downloadCount") or 0),
         int(mstats.get("thumbsUpCount") or 0), int(mstats.get("thumbsDownCount") or 0),
         int(mstats.get("commentCount") or 0), int(mstats.get("tippedAmountCount") or 0),
         len(newest), len(people), int(len(newest) >= 200), statistics.median(reactions) if reactions else None,
         round(dislikes / (likes + dislikes), 3) if likes + dislikes else None, json.dumps(nsfw_share),
         round(len(unasked) / len(hot), 3) if hot else None, len(weights), _pct(weights, 0.25), _pct(weights, 0.5),
         _pct(weights, 0.75), round((up + 1) / (up + down + 2), 4), round(up / (downloads / 1000), 2) if downloads else None,
         round(int(vstats.get("downloadCount") or 0) / days, 2) if days else None, newer, "ok"))
    per_day, exact = image_rate(newest, published)
    connection.execute("""update community_stats set images_per_day = ?, images_exact = ?, top_reactions_median = ?,
                          top_reactions_total = ?, top_sampled = ?, stats_version = ? where version_id = ?""",
                       (per_day, exact, statistics.median(top_reactions) if top_reactions else None,
                        sum(top_reactions), len(top), STATS_VERSION, version_id))
    connection.execute("insert or replace into community_history values (?, ?, ?, ?, ?)",
                       (version_id, _now()[:10], int(vstats.get("downloadCount") or 0),
                        int(vstats.get("thumbsUpCount") or 0), int(vstats.get("thumbsDownCount") or 0)))

    # prompts: keep the 20 most-reacted hits and 10 newest-with-reactions, then pick the test prompts
    connection.execute("delete from community_prompts where version_id = ? and source in ('top', 'newest')", (version_id,))
    for source, items, keep in (("top", top, 20), ("newest", sorted(newest, key=lambda i: -_reactions(i.get("stats") or {})), 10)):
        kept = 0
        for item in items:
            meta = item.get("meta") or {}
            prompt = str(meta.get("prompt") or "").strip()
            if len(prompt) < 12 or kept >= keep:
                continue
            cleaned = clean_prompt(prompt)
            connection.execute("""insert or replace into community_prompts (image_id, version_id, source, prompt, cleaned,
                negative, weight, base_model, prompt_style, nsfw_level, reactions, picked) values (?,?,?,?,?,?,?,?,?,?,?,0)""",
                (int(item["id"]), version_id, source, prompt[:4000], cleaned, str(meta.get("negativePrompt") or "")[:1500],
                 _weight_for(item, version_id, names, local), item.get("baseModel") or "", prompt_style(prompt),
                 str(item.get("nsfwLevel") or "None"), _reactions(item.get("stats") or {})))
            kept += 1
    _add_creator_prompts(connection, row, version_id)
    pick_test_prompts(connection, version_id)
    return "ok"


def _add_creator_prompts(connection: sqlite3.Connection, row: sqlite3.Row, version_id: int) -> None:
    """The creator's showcase prompts: the gallery the downloader saved next to the LoRA (examples table)."""
    connection.execute("delete from community_prompts where version_id = ? and source = 'creator'", (version_id,))
    for n, example in enumerate(connection.execute(
            "select prompt, negative, weight, nsfw_level, base_checkpoint from examples where path = ? and prompt != '' "
            "order by coalesce(nsfw_level, 1)", (row["path"],)).fetchall()):
        prompt = str(example["prompt"] or "").strip()
        if len(prompt) < 12:
            continue
        level = {1: "None", 2: "Soft", 4: "Mature", 8: "X", 16: "X", 32: "X"}.get(example["nsfw_level"] or 1, "None")
        connection.execute("""insert or replace into community_prompts (image_id, version_id, source, prompt, cleaned,
            negative, weight, base_model, prompt_style, nsfw_level, reactions, picked) values (?,?,?,?,?,?,?,?,?,?,?,0)""",
            (-(n + 1), version_id, "creator", prompt[:4000], clean_prompt(prompt), str(example["negative"] or "")[:1500],
             example["weight"], row["base_model"] or "", prompt_style(prompt), level, None))


def pick_test_prompts(connection: sqlite3.Connection, version_id: int) -> int:
    """Up to 3 creator + 4 top + 1 newest, no two alike; natural-language prompts before tag-style ones."""
    connection.execute("update community_prompts set picked = 0 where version_id = ?", (version_id,))
    chosen: List[str] = []
    for source, want in (("creator", CREATOR_PER_LORA), ("top", TOP_PER_LORA), ("newest", NEWEST_PER_LORA)):
        rows = connection.execute(
            "select image_id, cleaned from community_prompts where version_id = ? and source = ? "
            "order by (prompt_style = 'natural') desc, coalesce(reactions, 0) desc", (version_id, source)).fetchall()
        got = 0
        for image_id, cleaned in rows:
            if got >= want or not cleaned or any(_similar(cleaned, c) for c in chosen):
                continue
            connection.execute("update community_prompts set picked = 1 where version_id = ? and source = ? and image_id = ?",
                               (version_id, source, image_id))
            chosen.append(cleaned)
            got += 1
    return len(chosen)


def collect(settings: Settings, only: str = "", limit: int = 0, force: bool = False, refresh_days: int = 30) -> Dict[str, int]:
    connection = sqlite3.connect(settings.catalog_path, timeout=120)
    connection.row_factory = sqlite3.Row
    ensure(connection)
    local = LocalIndex(connection)
    where, params = "civitai_version_id is not null", []
    if only:
        where += " and replace(rel_dir, '\\', '/') like ?"
        params.append(only.replace("\\", "/").strip("/") + "%")
    rows = list(connection.execute(f"""select path, civitai_version_id, civitai_model_id, base_model, stem, model_name,
                                       file_name from loras where {where} order by rel_dir""", params))
    seen, unique = set(), []
    for row in rows:                                     # duplicates share a version: fetch it once
        if row["civitai_version_id"] not in seen:
            seen.add(row["civitai_version_id"])
            unique.append(row)
    fresh = {r[0] for r in connection.execute(         # recent AND has today's numbers (stats_version)
        "select version_id from community_stats where status = 'ok' and fetched_at >= ? and coalesce(stats_version, 1) >= ?",
        (time.strftime("%Y-%m-%d", time.localtime(time.time() - refresh_days * 86400)), STATS_VERSION))}
    gone = {r[0] for r in connection.execute(          # deleted from Civitai: asked again only every GONE_RETRY_DAYS
        "select version_id from community_stats where status = 'not on civitai' and fetched_at >= ?",
        (time.strftime("%Y-%m-%d", time.localtime(time.time() - GONE_RETRY_DAYS * 86400)),))}
    todo = [r for r in unique if force or int(r["civitai_version_id"]) not in fresh | gone]
    skipped_gone = sum(1 for r in unique if int(r["civitai_version_id"]) in gone) if not force else 0
    recent = len(unique) - len(todo) - skipped_gone
    if limit:
        todo = todo[:limit]
    collector = Collector(settings)
    print(f"[community] {len(todo)} LoRAs to fetch ({recent} fetched in the last {refresh_days} days, {skipped_gone} "
          f"no longer on Civitai) "
          f"from {collector.client.api}; API key {'found' if collector.client.has_key else 'NOT found'}", flush=True)
    counts = {"ok": 0, "failed": 0, "not_on_civitai": 0}
    started = time.time()
    for number, row in enumerate(todo, 1):
        try:
            status = collect_one(collector, connection, local, row)
        except Exception as error:                     # one bad record never stops the run
            status = f"{type(error).__name__}: {error}"
            print(f"[community] {row['model_name'] or row['stem']}: {status}", flush=True)
        counts["ok" if status == "ok" else "not_on_civitai" if status == "not on civitai" else "failed"] += 1
        if number % 10 == 0 or number == len(todo):
            connection.commit()
            rate = (time.time() - started) / number
            print(f"[community] {number}/{len(todo)}  eta {(len(todo) - number) * rate / 60:.0f} min", flush=True)
    connection.commit()
    counts["scored"] = score_all(connection)
    connection.commit()
    connection.close()
    counts["live_calls"] = collector.client.live_calls
    return counts


# -- the score ----------------------------------------------------------------------------------------------------------
def _percentiles(values: List[Optional[float]]) -> List[Optional[float]]:
    present = sorted(v for v in values if v is not None)
    if not present:
        return [None] * len(values)
    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        below = sum(1 for p in present if p < v)
        equal = sum(1 for p in present if p == v)
        out.append((below + 0.5 * equal) / len(present))
    return out


def score_all(connection: sqlite3.Connection) -> int:
    """Percentiles within base model + role (a group under 8 LoRAs falls back to the base model alone)."""
    cursor = connection.execute("select * from community_stats where status = 'ok'")
    names = [d[0] for d in cursor.description]
    rows = [dict(zip(names, r)) for r in cursor]
    by_pair: Dict[Tuple[str, str], List[dict]] = {}
    by_base: Dict[str, List[dict]] = {}
    for r in rows:
        by_pair.setdefault((r["base_model"] or "", r["role"] or ""), []).append(r)
        by_base.setdefault(r["base_model"] or "", []).append(r)
    for r in rows:
        group = by_pair[(r["base_model"] or "", r["role"] or "")]
        label = f"{r['base_model']} {r['role'] or 'LoRAs'}"
        if len(group) < 8:
            group, label = by_base[r["base_model"] or ""], f"{r['base_model']} LoRAs"
        r["_group"], r["_label"] = group, label
    done = 0
    cache: Dict[int, Dict[int, float]] = {}
    for r in rows:
        key = id(r["_group"])
        if key not in cache:
            group = r["_group"]
            columns = []
            for feature in ("approval", "engagement", "momentum", "distinct_posters", "median_reactions"):
                values = [g.get(feature) for g in group]
                if feature == "median_reactions":          # reception: reactions minus a dislike penalty
                    values = [None if g.get("median_reactions") is None else
                              g["median_reactions"] * (1 - 2 * (g.get("dislike_share") or 0)) for g in group]
                columns.append(_percentiles(values))
            combined = []
            for i in range(len(group)):
                parts = [c[i] for c in columns if c[i] is not None]
                combined.append(sum(parts) / len(parts) if parts else None)
            final = _percentiles(combined)
            cache[key] = {g["version_id"]: f for g, f in zip(group, final)}
        value = cache[key].get(r["version_id"])
        if len(r["_group"]) < 5:                       # 1-4 LoRAs of that base model: a rank means nothing yet
            value = None
        connection.execute("update community_stats set score = ?, score_group = ? where version_id = ?",
                           (round(value * 100, 1) if value is not None else None, r["_label"], r["version_id"]))
        done += 1
    peer_ranks(connection, rows, by_base)
    return done


def peer_ranks(connection: sqlite3.Connection, rows: List[dict], by_base: Dict[str, List[dict]]) -> None:
    """Each LoRA against the others of its platform (base model), feature by feature: percentile 0-100 (100 = most).
    Stored as JSON in `peer`: {group, n, downloads, engagement, images_per_day, top_reactions_median}. A platform with
    fewer than 5 collected LoRAs gets no ranks."""
    for base, group in by_base.items():
        ranks: Dict[str, List[Optional[float]]] = {}
        for column, _label in PEER_FEATURES:
            ranks[column] = _percentiles([g.get(column) for g in group]) if len(group) >= 5 else [None] * len(group)
        for i, g in enumerate(group):
            peer = {"group": base or "?", "n": len(group)}
            peer.update({column: (round(ranks[column][i] * 100) if ranks[column][i] is not None else None)
                         for column, _label in PEER_FEATURES})
            connection.execute("update community_stats set peer = ? where version_id = ?", (json.dumps(peer), g["version_id"]))


def peer_text(stats: Any) -> str:
    """'downloads top 12%, likes per download top 40% ... of 470 Krea 2 LoRAs' ('' when not ranked)."""
    try:
        peer = stats["peer"] or {}
        peer = json.loads(peer) if isinstance(peer, str) else peer
    except (TypeError, ValueError, KeyError, IndexError):
        return ""
    parts = []
    for column, label in PEER_FEATURES:
        value = peer.get(column)
        if value is not None:
            parts.append(f"{label} {'top ' + str(max(1, 100 - value)) if value >= 50 else 'bottom ' + str(max(1, value))}%")
    return f"{', '.join(parts)} (among {peer.get('n')} {peer.get('group')} LoRAs collected)" if parts else ""


# -- LoRAs you do not own (the wishlist) ---------------------------------------------------------------------------------
def quick_scores(settings: Settings, version_ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    """A lighter community score for versions you do not own: approval, engagement and momentum from the version
    record only (one cached call each, no image sampling), each a percentile among YOUR collected LoRAs of the same base
    model, averaged, 0-100. {version_id: {score, base_model, name, model_name, downloads, up, down}}; a version Civitai
    no longer has is left out. Version records are cached by the crawl, so the numbers can be as old as that cache."""
    client = CivitaiClient(settings)
    out: Dict[int, Dict[str, Any]] = {}
    for version_id in version_ids:
        version = client.get("v", int(version_id))
        if not version:
            continue
        stats = version.get("stats") or {}
        up, down = int(stats.get("thumbsUpCount") or 0), int(stats.get("thumbsDownCount") or 0)
        downloads = int(stats.get("downloadCount") or 0)
        published = version.get("publishedAt") or version.get("createdAt") or ""
        try:
            days = max(7.0, (datetime.now(timezone.utc) - datetime.fromisoformat(published.replace("Z", "+00:00"))).days)
        except ValueError:
            days = None
        out[int(version_id)] = {"base_model": version.get("baseModel") or "", "name": version.get("name") or "",
                                "model_name": (version.get("model") or {}).get("name") or "",
                                "downloads": downloads, "up": up, "down": down,
                                "approval": (up + 1) / (up + down + 2),
                                "engagement": up / (downloads / 1000) if downloads else None,
                                "momentum": downloads / days if days else None, "score": None}
    connection = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True, timeout=30)
    try:
        mine = connection.execute("select base_model, approval, engagement, momentum from community_stats "
                                  "where status = 'ok'").fetchall()
    except sqlite3.Error:
        mine = []
    finally:
        connection.close()
    for info in out.values():
        same = [m for m in mine if (m[0] or "") == info["base_model"]]
        group = same if len(same) >= 8 else mine
        if len(group) < 5:
            continue
        parts = []
        for column, feature in enumerate(("approval", "engagement", "momentum"), start=1):
            value = info[feature]
            present = [g[column] for g in group if g[column] is not None]
            if value is None or not present:
                continue
            below = sum(1 for p in present if p < value)
            equal = sum(1 for p in present if p == value)
            parts.append((below + 0.5 * equal) / len(present))
        if parts:
            info["score"] = round(100 * sum(parts) / len(parts), 1)
    return out


# -- identification by hash ---------------------------------------------------------------------------------------------
BY_HASH_FILE = os.path.join(CACHE_DIR, "by_hash.json")


def hashed_versions() -> Dict[str, Optional[int]]:
    """{sha256: version id or None (looked up, not on Civitai)} - read by the indexer so a re-index keeps the ids."""
    try:
        with open(BY_HASH_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def identify(settings: Settings, limit: int = 0, retry_misses: bool = False) -> Dict[str, int]:
    from .crawl import record
    connection = sqlite3.connect(settings.catalog_path, timeout=120)
    connection.row_factory = sqlite3.Row
    known = hashed_versions()
    rows = [r for r in connection.execute("select path, sha256 from loras where civitai_version_id is null and sha256 "
                                          "is not null and sha256 != ''")
            if retry_misses or r["sha256"].lower() not in known]
    if limit:
        rows = rows[:limit]
    client = CivitaiClient(settings)
    print(f"[identify] {len(rows)} files without a Civitai id to look up by sha256", flush=True)
    counts = {"found": 0, "not_on_civitai": 0}
    for number, row in enumerate(rows, 1):
        sha = row["sha256"].lower()
        data = client.get_json(f"model-versions/by-hash/{sha}", {})
        if data and data.get("id"):
            known[sha] = int(data["id"])
            with open(os.path.join(CACHE_DIR, f"v{data['id']}.json"), "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            model = client.get("m", data.get("modelId")) if data.get("modelId") else None
            update = record(data, model or {}, settings, data.get("modelId"))
            update = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in update.items()}
            update["id_source"] = "by_hash"
            update["source_kind"] = "civitai"
            connection.execute(f"update loras set {', '.join(f'{k} = ?' for k in update)} where path = ?",
                               list(update.values()) + [row["path"]])
            counts["found"] += 1
        else:
            known[sha] = None
            counts["not_on_civitai"] += 1
        if number % 20 == 0 or number == len(rows):
            connection.commit()
            with open(BY_HASH_FILE, "w", encoding="utf-8") as handle:
                json.dump(known, handle)
            print(f"[identify] {number}/{len(rows)}  found {counts['found']}", flush=True)
    connection.commit()
    connection.close()
    with open(BY_HASH_FILE, "w", encoding="utf-8") as handle:
        json.dump(known, handle)
    counts["live_calls"] = client.live_calls
    return counts


# -- reading ------------------------------------------------------------------------------------------------------------
def for_card(connection: sqlite3.Connection, version_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """What the card editor shows: the numbers, the score and every kept prompt (test prompts first)."""
    if not version_id:
        return None
    try:
        stats = connection.execute("select * from community_stats where version_id = ?", (version_id,)).fetchone()
    except sqlite3.Error:
        return None
    if stats is None or stats["status"] != "ok":
        return None
    prompts = [dict(r) for r in connection.execute(
        "select source, cleaned, weight, prompt_style, nsfw_level, reactions, picked, image_id from community_prompts "
        "where version_id = ? order by picked desc, case source when 'creator' then 0 when 'top' then 1 else 2 end, "
        "coalesce(reactions, 0) desc", (version_id,))]
    out = {k: stats[k] for k in stats.keys()}
    out["nsfw_share"] = json.loads(stats["nsfw_share"] or "{}")
    try:
        out["peer"] = json.loads(stats["peer"] or "{}") if "peer" in stats.keys() else {}
    except ValueError:
        out["peer"] = {}
    out["prompts"] = prompts
    return out


def show(settings: Settings, name: str) -> str:
    connection = sqlite3.connect(settings.catalog_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute("select path, model_name, stem, civitai_version_id from loras where (model_name like ? or "
                              "stem like ?) and civitai_version_id is not null limit 5", (f"%{name}%", f"%{name}%")).fetchall()
    lines = []
    for row in rows:
        card = for_card(connection, row["civitai_version_id"])
        lines.append(f"\n{row['model_name'] or row['stem']}  (version {row['civitai_version_id']})")
        if not card:
            lines.append("   not collected yet")
            continue
        lines.append(f"   score {card['score']} in {card['score_group']} | approval {card['approval']} | "
                     f"{card['downloads']} dl, {card['thumbs_up']} up / {card['thumbs_down']} down | "
                     f"{card['momentum']} dl/day | {card['distinct_posters']} posters in {card['images_sampled']} newest")
        lines.append(f"   weights n={card['weight_n']}  {card['weight_p25']} / {card['weight_median']} / {card['weight_p75']}"
                     f" | nsfw {card['nsfw_share']} | explicit unasked {card['explicit_unasked']}"
                     + (f" | NEWER: {card['newer_version']}" if card['newer_version'] else ""))
        if card.get("images_per_day") is not None or card.get("top_reactions_median") is not None:
            count = f"{card['images_exact']} images in all" if card.get("images_exact") is not None else "200+ images"
            lines.append(f"   {count}, {card['images_per_day']} posted per day | top {card['top_sampled']} images: median "
                         f"{card['top_reactions_median']} reactions, {card['top_reactions_total']} in all")
        peers = peer_text(card)
        if peers:
            lines.append(f"   {peers}")
        for p in [p for p in card["prompts"] if p["picked"]]:
            lines.append(f"   [{p['source']:7s} {p['nsfw_level']:6s} {p['prompt_style']:7s} w={p['weight']}] {p['cleaned'][:150]}")
    connection.close()
    return "\n".join(lines) or "no LoRA with a Civitai id matches that name"
