"""
The card editor: a small local web server (this machine only) that shows each LoRA's card with its example images
and lets you confirm or correct it with pick-list chips. Edits are saved to the catalog as you go (card_edits).

    python run.py serve                 then open http://127.0.0.1:8765
    python run.py serve --port 8770

Endpoints (all JSON unless noted)
    GET  /                       the editor page (editor.html)
    GET  /api/vocab              the pick lists
    GET  /api/shelves            shelf/category folders that have cards, with counts reviewed / total
    GET  /api/queue?shelf=&order=priority|name|folder|rating&status=todo|all|confirmed|edited|unsure
                       &rating=any|unrated|low|high      (low = 1-2 stars, high = 3+)
                       &q=fashion photo   search (search.py): returns {items, matched, hidden, mode}
    GET  /api/card?path=         one LoRA: facts, card, model prefill, your edit, image count
    GET  /img?path=&i=           example image i as a JPEG thumbnail (only for files in the catalog)
    GET  /api/artists?q=         artist name suggestions
    GET  /api/creators?q=        LoRA creator (Civitai username) suggestions
    POST /api/add_term           {facet, word, parent}  make your own word a button for every card (vocab.yaml)
    POST /api/fetch_images       {path}  download this LoRA's Civitai example images next to it (civitai_images)
    GET  /api/folders            the organising folders with LoRA counts (organise.py)
    POST /api/organise/plan      {action: move|rename|new, ...}  what would happen; nothing touches the disk
    POST /api/organise/apply     {id}  do a plan          POST /api/organise/undo  {id}  undo a batch (newest first)
    GET  /api/organise/batches   past batches             GET  /api/organise/references?id=  saved files naming them
    GET  /api/organise/status    LoRA Manager rescan state
    GET  /sweeps                 the sweep ratings page (sweeps.html, stacks.py): 1-5 stars, flaw / praise tags
    GET  /api/sweeps             sweep folders        GET /api/sweep?id=   one sweep's cells and your marks
    GET  /sweepimg?id=&file=&side=512   a cell as JPEG    POST /api/stacks/mark  {id, file, verdict, notes}
    GET  /api/stacks/summary     marked stacks and LoRA pairs with good / bad counts
    POST /api/stacks/mark also takes stars, tags, push (stronger|weaker|'')     POST /api/stacks/update  backfill + refit
    Style search (style.py; optional, settings.json style_enabled):
    GET  /api/style_status       {enabled, image_search, reason}
    GET  /api/similar?path=&n=8&scope=base|all&kinds=look|all&nsfw=0|1   closest looks to one LoRA, + {info}
    GET  /api/queue?...&q=like:mignola   (likeall: across base models) the list in look order
    GET  /api/queue?looks=1&shelf=&min=0.85   look-alike pairs (style LoRAs whose previews look the same)
    POST /api/style_search?<queue filters>   body = an image file: the list in order of closeness to its look
    POST /api/save               {path, facets, artists, creator, triggers, trigger_states, weight, weight_low,
                                  weight_high, rating, notes, unsure, other_changed}
    Prompt words (trigger_picks.py, 2026-10-15):
    GET  /api/prompt_words?path=  recipe variants, trigger-test scores, your picks, known checkpoints, current words
    POST /api/prompt_words       {path, checkpoint ('' = all), variant, text}  your pick;  {..., clear: true} removes it
    POST /api/prompt_words/cell  {id, file, scope: checkpoint|all}  pick the variant a trigger-test cell used
    POST /api/word_options       {path, options: [{name, text, default}]} or {path, reset: true}  (2026-10-20)
    POST /api/share_sheets       {id, mode, background, group_by, ratings}  labelled strips / singles for posting
                                 (Eric_Krea2's _sweep_sheets.py) into <sweep>/share/    GET /sharefile?id=&file=

Nothing here writes to the generated cards; the overnight card run can keep writing while you edit (sqlite WAL).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import io
import json
import os
import re
import socket
import sqlite3
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from . import community, creators, edits, facets
from .artists import fold
from .cards import VIDEO_EXTENSIONS, _video_frame, artist_candidates, preview_images
from .config import Settings, load_settings
from .organise import Organiser
from .search import SearchIndex
from . import style
from .stacks import Sweeps
from .vocab import NSFW_TERMS, _civitai_nsfw, load_vocab, prefill

HERE = os.path.dirname(os.path.abspath(__file__))


def code_stamp() -> float:
    """Newest change time of this package's Python files. The ComfyUI panel compares it with the running editor's (from
    /api/health) to see that the editor runs older code than is on disk, and restarts it (2026-10-17). Pages (.html)
    are read fresh on every request, so they do not count."""
    stamps = [os.path.getmtime(os.path.join(HERE, name)) for name in os.listdir(HERE) if name.endswith(".py")]
    return max(stamps) if stamps else 0.0


STARTED = time.time()
CODE_STAMP = code_stamp()


def process_alive(pid: int) -> bool:
    """Is that process still running? (Windows: OpenProcess + exit code - os.kill would end it there.)"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, int(pid))         # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259                    # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
IMAGES_PER_CARD = 4          # one page of the editor's 2x2 grid
MAX_IMAGES = 24              # how many it can page through
THUMB_SIDE = 720
STYLE_TOP = 200              # like: / image search list at most this many LoRAs, closest first
LIKE = re.compile(r'\b(like|likeall):(?:"([^"]+)"|(\S+))', re.IGNORECASE)


def _json_list(value: Any) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def _shelf(rel_dir: str, depth: int = 2) -> str:
    parts = (rel_dir or "").replace("\\", "/").split("/")
    return "/".join(parts[:depth])


class Editor:
    """State shared by request threads: settings, caches. Each request opens its own sqlite connection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.prefill_cache: Dict[tuple, Dict[str, List[str]]] = {}
        self.image_cache: Dict[tuple, bytes] = {}
        self.image_list_cache: Dict[str, List[str]] = {}
        self.artist_names: Optional[List[tuple]] = None
        self.creator_cache: Dict[Any, str] = {}
        self.search = SearchIndex(settings)
        self.organiser = Organiser(settings)
        self.sweeps = Sweeps(settings)
        self.style_index = style.StyleIndex(settings)     # stored vectors; no torch needed
        self.style_model = style.ModelHolder(settings)    # loaded on the first image search, unloaded when idle
        threading.Thread(target=self.organiser.folders, daemon=True).start()   # folder panel ready before it opens
        with self.connect() as connection:
            edits.ensure(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.catalog_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    # -- data ----------------------------------------------------------------------------------------------------
    def rows(self, connection: sqlite3.Connection, shelf: str = "", path: str = "") -> List[sqlite3.Row]:
        sql = """select l.*, c.kind, c.scope, c.effect, c.triggers as card_triggers, c.weight_typical, c.weight_low,
                        c.weight_high, c.overrides, c.nsfw_level as card_nsfw, c.confidence, c.artists as card_artists,
                        c.medium, c.movement_era, c.palette, c.line_quality, c.suited_subjects, c.refined_by, c.refined_at,
                        c.genre, c.mood,
                        e.status as edit_status, e.rating as edit_rating
                 from loras l join cards c on c.path = l.path
                 left join card_edits e on e.key = coalesce(lower(l.sha256), l.path)"""
        if path:
            return list(connection.execute(sql + " where l.path = ?", (path,)))
        if shelf:
            # the folder itself and everything under it ('Flux2' no longer takes in 'Flux2klein', 2026-10-16)
            shelf = shelf.replace("\\", "/").strip("/")
            return list(connection.execute(sql + " where replace(l.rel_dir, '\\', '/') = ? or "
                                           "replace(l.rel_dir, '\\', '/') like ?", (shelf, shelf + "/%")))
        return list(connection.execute(sql))

    def prefill_for(self, row: sqlite3.Row) -> Dict[str, List[str]]:
        vocab = load_vocab()
        key = (row["path"], row["refined_at"], id(vocab))
        if key not in self.prefill_cache:
            self.prefill_cache[key] = prefill(vocab, row, row)
        return self.prefill_cache[key]

    def facet_terms(self, connection: sqlite3.Connection) -> Dict[str, Dict[str, list]]:
        """{path: facet terms} from card_facets (built by run.py facets, kept current by every save). Reading these is
        one quick query; working the terms out again per card (prefill) took ~14 s for the whole library on the
        first list after each editor start (2026-10-16)."""
        out: Dict[str, Dict[str, list]] = {}
        try:
            for path, text in connection.execute("select path, facets from card_facets"):
                try:
                    terms = json.loads(text or "{}")
                except (TypeError, ValueError):
                    continue
                if isinstance(terms, dict):
                    out[path] = terms
        except sqlite3.Error:
            pass
        return out

    def priority(self, row: sqlite3.Row, terms: Optional[Dict[str, list]] = None) -> float:
        """Higher = look at it sooner: low confidence, sparse cards, and an artist hinted by name but not on the card.
        `terms`: the card's facet terms when the caller has them (card_facets), else they are worked out here."""
        confidence = row["confidence"] or 0.0
        filled = terms if terms is not None else self.prefill_for(row)
        empty = sum(1 for facet in ("use", "medium", "period", "rendering", "line") if not filled.get(facet))
        score = (1.0 - confidence) * 2.0 + empty * 0.15
        if row["kind"] == "style_medium" and filled.get("role") == ["style"] and not _json_list(row["card_artists"]):
            score += 0.2
        return round(score, 3)

    def duplicates(self) -> List[Dict[str, Any]]:
        """Every LoRA that exists more than once (same sha256), copies next to each other."""
        with self.connect() as connection:
            rows = self.rows(connection)
        groups: Dict[str, list] = {}
        for row in rows:
            if row["sha256"]:
                groups.setdefault(row["sha256"].lower(), []).append(row)
        items = []
        for sha, copies in sorted(groups.items(), key=lambda kv: (kv[1][0]["model_name"] or kv[1][0]["stem"] or "").lower()):
            if len(copies) < 2:
                continue
            for n, row in enumerate(sorted(copies, key=lambda r: r["rel_dir"].lower())):
                items.append({"rating": row["edit_rating"], "path": row["path"], "name": row["model_name"] or row["stem"],
                              "version": row["version_name"] or "", "folder": row["rel_dir"], "kind": row["kind"],
                              "confidence": row["confidence"], "status": row["edit_status"] or "todo",
                              "priority": 0, "group": sha[:10], "copy": f"{n + 1}/{len(copies)}"})
        return items

    # -- style search (style.py) ---------------------------------------------------------------------------------
    def style_status(self) -> Dict[str, Any]:
        off = style.available(self.settings, need_model=False)
        if not off:
            self.style_index.ensure()
        return {"enabled": not off, "image_search": not style.available(self.settings), "reason": off or
                style.available(self.settings), "vectors": len(self.style_index.items) if not off else 0}

    def similar(self, path: str, n: int, scope: str, kinds: str, nsfw: bool) -> Dict[str, Any]:
        off = style.available(self.settings, need_model=False)
        if off:
            return {"off": off}
        info = self.style_index.info(path)
        if info is None:
            return {"info": None, "items": [], "note": "no style vector: this LoRA has no readable preview images"}
        return {"info": info, "items": self.style_index.similar(path, n=n, scope=scope, kinds=kinds, nsfw=nsfw)}

    def _top(self, scores: Dict[str, float], arch: str = "") -> Dict[str, float]:
        if arch:
            keep = {item["path"] for item in self.style_index.items if item["arch"].lower() == arch.lower()}
            scores = {p: s for p, s in scores.items() if p in keep}
        return dict(sorted(scores.items(), key=lambda kv: -kv[1])[:STYLE_TOP])

    def like_hits(self, words: str, every_base: bool) -> tuple:
        """like:<name> -> ({path: cosine}, anchor name). The anchor is the best name match that has a style vector."""
        self.style_index.ensure()
        candidates = self.search.search(f'name:"{words}"') or self.search.search(words)
        ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
        anchor = next((p for p, _s in ranked if self.style_index.vector(p) is not None), None)
        if anchor is None:
            lowered = words.lower()
            anchor = next((i["path"] for i in self.style_index.items if lowered in i["name"].lower()), None)
        if anchor is None:
            return {}, ""
        item = self.style_index.items[self.style_index.by_path[anchor]]
        scores = self._top(self.style_index.rank(item["vec"]), "" if every_base else item["arch"])
        return scores, item["name"]

    def lookalikes(self, shelf: str, minimum: float) -> List[Dict[str, Any]]:
        """Look-alike pairs as list items: a, b next to each other, grouped like the duplicates list."""
        arch = (shelf or "").replace("\\", "/").split("/")[0]
        pairs = self.style_index.twins(minimum=minimum, arch=arch)
        with self.connect() as connection:
            rows = {r["path"]: r for r in self.rows(connection)}
        items = []
        for n, pair in enumerate(pairs):
            for side in ("a", "b"):
                row = rows.get(pair[side]["path"])
                if row is None:
                    continue
                items.append({"rating": row["edit_rating"], "path": row["path"], "name": row["model_name"] or row["stem"],
                              "version": row["version_name"] or "", "folder": row["rel_dir"], "kind": row["kind"],
                              "confidence": row["confidence"], "status": row["edit_status"] or "todo", "priority": 0,
                              "group": f"p{n}", "score": pair["score"]})
        return items

    def queue(self, shelf: str, order: str, status: str, rating: str = "any", q: str = "", folder: str = "",
              style_hits: Optional[Dict[str, float]] = None, style_mode: str = "", anchor: str = "") -> Any:
        """The list on the left. With a search `q`: {items, matched, mode, hidden} instead of a bare list, items in
        match order unless another order is picked. `folder` (a relative folder from the folder panel, any depth)
        replaces the shelf. like:<name> in q, or style_hits from an image search: the list in order of look."""
        like = LIKE.search(q or "") if style_hits is None else None
        if like:
            off = style.available(self.settings, need_model=False)
            if off:
                return {"items": [], "matched": 0, "hidden": 0, "mode": "style-off", "reason": off}
            style_hits, anchor = self.like_hits(like.group(2) or like.group(3), like.group(1).lower() == "likeall")
            style_mode = "like"
            q = (q[:like.start()] + q[like.end():]).strip()
            if q:                                                   # like:mucha poster -> look order, text filter
                text = self.search.search(q)
                style_hits = {p: s for p, s in style_hits.items() if p in text}
        if style_hits is not None:
            hits = style_hits
        else:
            hits = self.search.search(q) if q.strip() else None
        extra = self.look_extras(hits) if hits and not style_mode else {}
        folder = folder.replace("\\", "/").strip("/")
        with self.queue_lock:                       # one list at a time: changing the view quickly no longer piles
            with self.connect() as connection:      # up several whole-library passes fighting each other
                rows = self.rows(connection, shelf="" if folder else shelf)
                terms_by_path = self.facet_terms(connection) if order == "priority" and hits is None else {}
        items = []
        extra_items = []
        for row in rows:
            is_extra = row["path"] in extra
            if hits is not None and row["path"] not in hits and not is_extra:
                continue
            if folder:
                rel = (row["rel_dir"] or "").replace("\\", "/")
                if rel.lower() != folder.lower() and not rel.lower().startswith(folder.lower() + "/"):
                    continue
            state = row["edit_status"] or "todo"
            if status != "all" and state != status:
                continue
            stars = row["edit_rating"]
            if (rating == "unrated" and stars) or (rating == "low" and not (stars and stars <= 2)) or \
                    (rating == "high" and not (stars and stars >= 3)):
                continue
            items.append({"rating": stars, "path": row["path"], "name": row["model_name"] or row["stem"], "version": row["version_name"] or "",
                          "folder": row["rel_dir"], "kind": row["kind"], "confidence": row["confidence"],
                          "status": state,
                          "priority": self.priority(row, terms_by_path.get(row["path"])) if order == "priority" and hits is None else 0,
                          "score": hits.get(row["path"]) if hits is not None else None})
            if is_extra:
                items[-1]["score"], items[-1]["extra"] = extra[row["path"]], True
                extra_items.append(items.pop())
        if hits is not None and order == "priority":
            items.sort(key=lambda item: -(item["score"] or 0))
        elif order == "name":
            items.sort(key=lambda item: item["name"].lower())
        elif order == "rating":
            items.sort(key=lambda item: (-(item["rating"] or 0), item["name"].lower()))
        elif order == "folder":
            items.sort(key=lambda item: (item["folder"].lower(), item["name"].lower()))
        else:
            items.sort(key=lambda item: -item["priority"])
        extra_items.sort(key=lambda item: -(item["score"] or 0))
        items += extra_items
        if hits is None:
            return items
        if style_mode:
            return {"items": items, "matched": len(hits), "hidden": len(hits) - len(items), "mode": style_mode,
                    "anchor": anchor}
        return {"items": items, "matched": len(hits), "hidden": len(hits) - (len(items) - len(extra_items)),
                "mode": self.search.last.get("mode", "none"), "extra": len(extra_items)}

    def look_extras(self, hits: Dict[str, float], anchors: int = 3, per_anchor: int = 6) -> Dict[str, float]:
        """Style search, widened text search: LoRAs that LOOK like the best text hits (style-type hits only) but that
        the text search did not find, e.g. a Frazetta search also showing Brom whose card never names Frazetta.
        {path: look similarity}; {} when style search is off."""
        if style.available(self.settings, need_model=False):
            return {}
        self.style_index.ensure()
        found: Dict[str, float] = {}
        used = 0
        for path, _score in sorted(hits.items(), key=lambda kv: -kv[1]):
            info = self.style_index.info(path)
            if not info or not info["look"]:
                continue
            for near in self.style_index.similar(path, n=per_anchor + 4):
                if near["path"] not in hits and near["score"] >= 0.65:     # 0.60-0.65 was mostly noise (frazetta test)
                    found[near["path"]] = max(found.get(near["path"], 0.0), near["score"])
            used += 1
            if used >= anchors:
                break
        return dict(sorted(found.items(), key=lambda kv: -kv[1])[:anchors * per_anchor])

    def style_search(self, data: bytes, shelf: str, order: str, status: str, rating: str, folder: str) -> Any:
        from . import image_meta
        try:                               # the LoRAs the image's own metadata names (works with style search off too)
            made_with = image_meta.match(self.settings.catalog_path, image_meta.read_bytes(data))
        except Exception as error:       # never fail the search over metadata
            print(f"[editor] image metadata not read: {error}")
            made_with = []
        off = style.available(self.settings)
        if off:
            return {"items": [], "matched": 0, "hidden": 0, "mode": "style-off", "reason": off, "made_with": made_with}
        from PIL import Image, UnidentifiedImageError
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError):
            return {"error": "that file is not an image this tool can read (jpg, png, webp...)"}
        vector = self.style_model.embed_image(image)
        arch = "" if folder else (shelf or "").replace("\\", "/").split("/")[0]
        hits = self._top(self.style_index.rank(vector), arch)
        result = self.queue(shelf, order, status, rating, "", folder, style_hits=hits, style_mode="image")
        if isinstance(result, dict):
            result["made_with"] = made_with
        return result

    def shelves(self) -> List[Dict[str, Any]]:
        counts: Dict[str, List[int]] = {}
        with self.connect() as connection:
            for row in connection.execute("""select l.rel_dir, e.status from loras l join cards c on c.path = l.path
                                             left join card_edits e on e.key = coalesce(lower(l.sha256), l.path)"""):
                shelf = _shelf(row["rel_dir"])
                counts.setdefault(shelf, [0, 0])
                counts[shelf][1] += 1
                if row["status"]:
                    counts[shelf][0] += 1
        return [{"shelf": shelf, "reviewed": done, "total": total} for shelf, (done, total) in sorted(counts.items())]

    def start_view(self) -> Dict[str, str]:
        """What the editor opens on (settings.json editor_start_shelf / editor_start_status)."""
        fresh = load_settings()                     # read now, so a settings change needs no editor restart
        status = str(getattr(fresh, "editor_start_status", "all") or "all")
        return {"shelf": str(getattr(fresh, "editor_start_shelf", "Krea2") or ""),
                "status": status if status in ("todo", "unsure", "edited", "confirmed", "all") else "all"}

    def images(self, path: str) -> List[str]:
        if path not in self.image_list_cache:
            chosen, sizes = [], set()          # LoRA Manager's .preview copy duplicates a gallery image: drop same-size files
            for candidate in preview_images(path, MAX_IMAGES * 2):
                try:
                    size = os.path.getsize(candidate)
                except OSError:
                    continue
                if size not in sizes:
                    sizes.add(size)
                    chosen.append(candidate)
            self.image_list_cache[path] = chosen[:MAX_IMAGES]
        return self.image_list_cache[path]

    def card(self, path: str) -> Optional[Dict[str, Any]]:
        with self.connect() as connection:
            rows = self.rows(connection, path=path)
            if not rows:
                return None
            row = rows[0]
            examples = list(connection.execute(
                "select prompt, weight, nsfw_level from examples where path = ? and prompt is not null and prompt != ''"
                " order by coalesce(nsfw_level, 1), length(prompt) limit 4", (path,)))
            edit = edits.get_edit(connection, row["sha256"], path)
            try:                                       # Civitai audience data (community.py), when collected
                community_block = community.for_card(connection, row["civitai_version_id"])
            except Exception:
                community_block = None
        try:
            candidates = [c["name"] for c in artist_candidates(row, self.settings)]
        except Exception:
            candidates = []
        card_artists = [a.get("name", "") if isinstance(a, dict) else str(a) for a in _json_list(row["card_artists"])]
        card_triggers = [t.get("text", "") if isinstance(t, dict) else str(t) for t in _json_list(row["card_triggers"])]
        civitai_url = ""
        if row["civitai_model_id"]:
            civitai_url = f"https://civitai.com/models/{row['civitai_model_id']}"
            if row["civitai_version_id"]:
                civitai_url += f"?modelVersionId={row['civitai_version_id']}"
        return {
            "path": path, "name": row["model_name"] or row["stem"], "version": row["version_name"] or "",
            "file": row["file_name"], "folder": row["rel_dir"], "base_model": row["base_model"] or "",
            "civitai_url": civitai_url, "source": row["source_kind"] or "",
            "civitai_tags": _json_list(row["tags"]), "trained_words": _json_list(row["trained_words"]),
            "description": (row["description"] or "")[:1200],
            "examples": [{"prompt": (e["prompt"] or "")[:400], "weight": e["weight"]} for e in examples],
            "card": {"kind": row["kind"], "scope": row["scope"], "effect": row["effect"] or "",
                     "confidence": row["confidence"], "nsfw": row["card_nsfw"], "overrides": _json_list(row["overrides"]),
                     "weight": row["weight_typical"], "weight_low": row["weight_low"], "weight_high": row["weight_high"],
                     "artists": card_artists, "triggers": card_triggers, "model": row["refined_by"] or "",
                     "free": {"medium": row["medium"], "era": row["movement_era"], "palette": row["palette"],
                              "genre": row["genre"], "mood": row["mood"],
                              "lines": row["line_quality"], "subjects": ", ".join(map(str, _json_list(row["suited_subjects"])))}},
            "prefill": self.prefill_for(row),
            "artist_candidates": candidates,
            "civitai_nsfw": NSFW_TERMS[_civitai_nsfw(row)] if row["nsfw_level"] else "",
            "creator": self.creator_for(row),
            "edit": edit,
            "lm_favorite": bool(row["lm_favorite"]), "lm_exclude": bool(row["lm_exclude"]),
            "image_count": len(self.images(path)), "has_civitai": bool(row["civitai_version_id"]),
            "community": community_block,
        }

    def mirror(self) -> Optional[sqlite3.Connection]:
        """LoRA Manager's offline Civitai mirror, read-only (it has a username per model)."""
        path = getattr(self.settings, "lm_civitai_db", "")
        if not path or not os.path.exists(path):
            return None
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)

    def creator_for(self, row: sqlite3.Row) -> str:
        """Who made it: our Civitai cache, then LoRA Manager's mirror, then 'self-trained' for your own training jobs."""
        model_id = row["civitai_model_id"]
        key = model_id or row["path"]
        if key in self.creator_cache:
            return self.creator_cache[key]
        name = ""
        if model_id:
            with self.connect() as connection:
                creators.ensure(connection)
                name = creators.lookup(connection, model_id)
        if not name and model_id:
            from .crawl import CACHE_DIR
            cached = os.path.join(CACHE_DIR, f"m{model_id}.json")
            try:
                with open(cached, "r", encoding="utf-8") as handle:
                    name = ((json.load(handle).get("creator") or {}).get("username") or "")
            except (OSError, ValueError):
                name = ""
            if not name:
                try:
                    connection = self.mirror()
                    if connection is not None:
                        found = connection.execute("select username from models where id = ?", (model_id,)).fetchone()
                        connection.close()
                        name = (found[0] or "") if found else ""
                except sqlite3.Error:
                    name = ""
        if not name and (row["source_kind"] == "self_trained" or row["training_job"]):
            name = "self-trained"
        self.creator_cache[key] = name
        return name

    def creators(self, query: str) -> List[str]:
        wanted = (query or "").strip()
        if len(wanted) < 2:
            return []
        names: List[str] = []
        with self.connect() as connection:
            for (value,) in connection.execute("select distinct creator from card_edits where creator like ? limit 12",
                                               (wanted + "%",)):
                if value:
                    names.append(value)
        try:
            connection = self.mirror()
            if connection is not None:
                for (value,) in connection.execute("select distinct username from models where username like ? limit 20",
                                                   (wanted + "%",)):
                    if value and value not in names:
                        names.append(value)
                connection.close()
        except sqlite3.Error:
            pass
        return names[:12]

    def thumbnail(self, path: str, index: int) -> Optional[bytes]:
        key = (path, index)
        if key in self.image_cache:
            return self.image_cache[key]
        images = self.images(path)
        if index < 0 or index >= len(images):
            return None
        source = images[index]
        try:
            from PIL import Image
            if source.lower().endswith(VIDEO_EXTENSIONS):
                image = _video_frame(source)
                if image is None:
                    return None
            else:
                with Image.open(source) as opened:
                    image = opened.convert("RGB")
            image.thumbnail((THUMB_SIDE, THUMB_SIDE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=84)
            data = buffer.getvalue()
        except Exception:
            return None
        with self.lock:
            if len(self.image_cache) > 600:
                self.image_cache.clear()
            self.image_cache[key] = data
        return data

    def artists(self, query: str) -> List[str]:
        if self.artist_names is None:
            names = set()
            try:
                with open(self.settings.artist_names_file, "r", encoding="utf-8") as handle:
                    names.update((json.load(handle).get("names") or {}).keys())
            except (OSError, ValueError):
                pass
            with self.connect() as connection:
                for (value,) in connection.execute("select artists from card_edits"):
                    names.update(str(n) for n in _json_list(value))
            self.artist_names = sorted((fold(name), name) for name in names)
        wanted = fold(query)
        if len(wanted) < 2:
            return []
        starts = [name for folded, name in self.artist_names if folded.startswith(wanted)
                  or any(part.startswith(wanted) for part in folded.split())]
        inside = [name for folded, name in self.artist_names if wanted in folded and name not in starts]
        return (starts + inside)[:12]

    def fetch_images(self, path: str) -> Dict[str, Any]:
        from .civitai_images import fetch_for
        from .crawl import CivitaiClient
        with self.connect() as connection:
            row = connection.execute("select civitai_version_id, civitai_model_id from loras where path = ?", (path,)).fetchone()
        if row is None:
            return {"ok": False, "error": "not in catalog"}
        if not row["civitai_version_id"]:
            return {"ok": False, "error": "no Civitai version id for this LoRA"}
        result = fetch_for(path, row["civitai_version_id"], row["civitai_model_id"], CivitaiClient(self.settings))
        with self.lock:
            self.image_list_cache.pop(path, None)
            for key in [k for k in self.image_cache if k[0] == path]:
                self.image_cache.pop(key, None)
        return {"ok": True, "saved": result["saved"], "available": result["available"], "image_count": len(self.images(path))}

    # -- file organising (organise.py) ---------------------------------------------------------------------------
    def organise_plan(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """{action: move, paths, target} | {action: rename, folder, new_path} | {action: new, parent, name}.
        Folders may be given relative to the first library root (as the folder panel shows them)."""
        def full(value: Any) -> str:
            value = str(value or "").replace("/", os.sep)
            return value if os.path.isabs(value) else os.path.join(self.organiser.roots[0], value)
        action = body.get("action")
        if action == "move":
            return self.organiser.plan_move([str(p) for p in body.get("paths") or []], full(body.get("target")))
        if action == "rename":
            return self.organiser.plan_rename_folder(full(body.get("folder")), full(body.get("new_path")))
        if action == "new":
            return self.organiser.plan_new_folder(full(body.get("parent")), str(body.get("name") or ""))
        if action == "delete":
            return self.organiser.plan_delete([str(p) for p in body.get("paths") or []])
        return {"error": f"unknown action {action!r}"}

    def organise_done(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Files moved: every cache keyed by path is stale."""
        with self.lock:
            self.image_cache.clear()
            self.image_list_cache.clear()
            self.prefill_cache.clear()
        if len(result.get("mapping") or {}) > 2000:              # a big folder move: the page does not need it all
            result.pop("mapping", None)
        return result

    # -- prompt words: which recipe variant a LoRA's prompt gets (trigger_picks.py, 2026-10-15) ------------------
    def prompt_words(self, path: str) -> Dict[str, Any]:
        from . import trigger_picks
        with self.connect() as connection:
            facets.ensure(connection)
            return trigger_picks.card_view(connection, path, self.settings.sweeps_dir)

    def set_prompt_words(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from . import trigger_picks
        path = str(body.get("path") or "")
        checkpoint = str(body.get("checkpoint") or "").strip()
        with self.connect() as connection:
            facets.ensure(connection)
            if not connection.execute("select 1 from loras where path = ?", (path,)).fetchone():
                return {"ok": False, "error": "no such LoRA"}
            if body.get("clear"):
                trigger_picks.clear_pick(connection, path, checkpoint)
            else:
                trigger_picks.set_pick(connection, path, checkpoint, str(body.get("variant") or "custom"),
                                       str(body.get("text") or "").strip())
            connection.commit()
            facets.refresh_one(connection, path)
            return {"ok": True, **trigger_picks.card_view(connection, path, self.settings.sweeps_dir)}

    def set_word_options(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Your word options (word_options.py, 2026-10-20): {path, options: [{name, text, default}]} replaces them
        ([] = none at all); {path, reset: true} forgets yours so the automatic ones come back."""
        from . import trigger_picks, word_options
        path = str(body.get("path") or "")
        with self.connect() as connection:
            facets.ensure(connection)
            if not connection.execute("select 1 from loras where path = ?", (path,)).fetchone():
                return {"ok": False, "error": "no such LoRA"}
            if body.get("reset"):
                word_options.reset(connection, path)
            else:
                word_options.set_options(connection, path, list(body.get("options") or []))
            connection.commit()
            facets.refresh_one(connection, path)
            return {"ok": True, **trigger_picks.card_view(connection, path, self.settings.sweeps_dir)}

    def share_sheets(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make share sheets for one sweep with Eric_Krea2's _sweep_sheets.py (2026-10-16), loaded by file path so
        the catalog keeps no copy of it. Stars / notes come from this catalog's marks."""
        import importlib.util
        from .config import COMFY_DIR
        sweep_id = str(body.get("id") or "")
        folder = self.sweeps.folder(sweep_id)
        if not folder or not os.path.isdir(folder):
            return {"ok": False, "error": "no such sweep"}
        module_path = os.path.join(COMFY_DIR or "", "custom_nodes", "Eric_Krea2", "_sweep_sheets.py")
        if not os.path.isfile(module_path):
            return {"ok": False, "error": f"Eric_Krea2 not found ({module_path})"}
        spec = importlib.util.spec_from_file_location("eric_krea2_sweep_sheets", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pick = lambda key, allowed, default: str(body.get(key)) if str(body.get(key)) in allowed else default
        result = module.make_sheets(
            folder, mode=pick("mode", ("strips", "singles", "strips + singles"), "strips"),
            background=pick("background", ("black", "white"), "black"),
            group_by=pick("group_by", ("auto", "prompt", "lora", "seed", "none"), "auto"),
            ratings=pick("ratings", ("none", "stars", "stars + notes"), "none"),
            catalog=self.settings.catalog_path, log=lambda *a: print("[editor]", *a))
        if result.get("ok"):
            result["folder"] = os.path.join(folder, "share")
        return result

    def share_file(self, sweep_id: str, file_name: str) -> Optional[bytes]:
        """One share sheet JPEG (only files directly inside <sweep>/share/)."""
        folder = self.sweeps.folder(sweep_id)
        name = os.path.basename(file_name or "")
        if not folder or not name.lower().endswith(".jpg"):
            return None
        path = os.path.join(folder, "share", name)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as handle:
            return handle.read()

    def sweep_record(self, path: str) -> Dict[str, Any]:
        """The card's Sweep record panel (stacks.lora_record): your marks on this LoRA, summed up, plus offers."""
        weight = None
        with self.connect() as connection:
            try:
                row = connection.execute("select weight_typical from card_facets where path = ?", (path,)).fetchone()
                weight = row[0] if row else None
            except sqlite3.Error:
                weight = None
        return self.sweeps.lora_record(path, weight)

    def edit_recipe(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Your correction of one recipe field (recipe_edits.py): {path, field, value}; value null = the model's."""
        from . import recipe_edits, trigger_picks
        path = str(body.get("path") or "")
        value = body.get("value")
        with self.connect() as connection:
            facets.ensure(connection)
            if not connection.execute("select 1 from loras where path = ?", (path,)).fetchone():
                return {"ok": False, "error": "no such LoRA"}
            recipe_edits.set_field(connection, path, str(body.get("field") or ""),
                                   None if value is None else str(value))
            recipe_edits.refresh(connection, path)
            connection.commit()
            facets.refresh_one(connection, path)
            return {"ok": True, **trigger_picks.card_view(connection, path, self.settings.sweeps_dir)}

    def pick_from_cell(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Sweeps page: 'use these words' on a trigger-test cell -> that cell's variant becomes your pick."""
        from . import trigger_picks
        from .style import _find_lora
        sweep_id, file_name = str(body.get("id") or ""), str(body.get("file") or "")
        manifest = self.sweeps.manifest(sweep_id)
        cell = next((c for c in manifest.get("cells") or [] if c.get("file") == file_name), None)
        if cell is None:
            return {"ok": False, "error": "no such cell"}
        over = cell.get("overrides") or {}
        if not over.get("lora_file"):
            return {"ok": False, "error": "this cell has no LoRA (a words-only or plain cell)"}
        prompt = str(over.get("prompt") or "")
        with self.connect() as connection:
            facets.ensure(connection)
            found = _find_lora(connection, str(over["lora_file"]))
            if not found:
                return {"ok": False, "error": f"{over['lora_file']} is not in the catalog"}
            path = found[0]
            variants = trigger_picks.variants(connection, path)
            match = next((v for v in variants if v.get("text")
                          and prompt.startswith(v["text"].strip().rstrip(",.") + ".")), None)
            if match is None:
                match = next((v for v in variants if not v.get("text")), None)      # the subject alone = 'none'
            if match is None:
                return {"ok": False, "error": "this cell's words match none of the LoRA's recipe variants"}
            checkpoint = "" if body.get("scope") == "all" else str(manifest.get("checkpoint") or "")
            trigger_picks.set_pick(connection, path, checkpoint, match["name"], match.get("text") or "",
                                   sweep=sweep_id)
            connection.commit()
            facets.refresh_one(connection, path)
        return {"ok": True, "path": path, "variant": match["name"], "text": match.get("text") or "",
                "checkpoint": checkpoint or "all checkpoints"}

    def save(self, body: Dict[str, Any]) -> Dict[str, Any]:
        path = body.get("path") or ""
        with self.connect() as connection:
            rows = self.rows(connection, path=path)
            if not rows:
                return {"ok": False, "error": "not in catalog"}
            row = rows[0]
            def number(key: str) -> Optional[float]:
                value = body.get(key)
                try:
                    value = float(value) if value not in (None, "") else None
                except (TypeError, ValueError):
                    return None
                return None if value is None else max(-10.0, min(10.0, value))
            weight, weight_low, weight_high = number("weight"), number("weight_low"), number("weight_high")
            if weight_low is not None and weight_high is not None and weight_low > weight_high:
                weight_low, weight_high = weight_high, weight_low
            triggers = [str(t).strip() for t in body.get("triggers") or [] if str(t).strip()]
            status = edits.save_edit(
                connection, sha256=row["sha256"], path=path, facets=body.get("facets") or {},
                artists=[str(a).strip() for a in body.get("artists") or [] if str(a).strip()],
                triggers=triggers, weight=weight, notes=body.get("notes") or "", unsure=bool(body.get("unsure")),
                prefill=self.prefill_for(row), other_changed=bool(body.get("other_changed")),
                creator=str(body.get("creator") or ""), rating=body.get("rating"), weight_low=weight_low,
                weight_high=weight_high, trigger_states=body.get("trigger_states") or {})
            try:
                facets.refresh_one(connection, path)      # keep the suggester's view in step with the edit
            except Exception as error:
                print(f"[editor] card_facets refresh failed for {path}: {error}")
        for artist in body.get("artists") or []:                 # new names become suggestions straight away
            if self.artist_names is not None and str(artist).strip():
                pair = (fold(str(artist)), str(artist).strip())
                if pair not in self.artist_names:
                    self.artist_names.append(pair)
                    self.artist_names.sort()
        return {"ok": True, "status": status, "rating": body.get("rating") or None}


def make_handler(editor: Editor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):          # quiet console
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if kind.startswith("image/") else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, code: int = 200) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                elif url.path in ("/", "/index.html"):
                    with open(os.path.join(HERE, "editor.html"), "rb") as handle:
                        self._send(200, handle.read(), "text/html; charset=utf-8")
                elif url.path == "/api/vocab":
                    self._json(load_vocab().to_json())
                elif url.path == "/api/shelves":
                    self._json(editor.shelves())
                elif url.path == "/api/start":
                    self._json(editor.start_view())
                elif url.path == "/api/queue" and query.get("looks") == "1":
                    off = style.available(editor.settings, need_model=False)
                    self._json({"items": [] if off else editor.lookalikes(query.get("shelf", ""), float(query.get("min", "0.85"))),
                                "looks": True, "reason": off})
                elif url.path == "/api/discover":                 # 2026-10-29, discover.py
                    from . import discover
                    self._json(discover.candidates(editor.settings, query.get("shelf", ""), query.get("all") == "1",
                                                   int(query.get("top", "100"))))
                elif url.path == "/api/style_status":
                    self._json(editor.style_status())
                elif url.path == "/api/similar":
                    self._json(editor.similar(query.get("path", ""), int(query.get("n", "8")), query.get("scope", "base"),
                                              query.get("kinds", "look"), query.get("nsfw", "0") == "1"))
                elif url.path == "/api/queue" and query.get("dupes") == "1":
                    self._json({"items": editor.duplicates(), "dupes": True})
                elif url.path == "/api/queue":
                    self._json(editor.queue(query.get("shelf", ""), query.get("order", "priority"), query.get("status", "todo"),
                                            query.get("rating", "any"), query.get("q", ""), query.get("folder", "")))
                elif url.path in ("/sweeps", "/sweeps.html"):
                    with open(os.path.join(HERE, "sweeps.html"), "rb") as handle:
                        self._send(200, handle.read(), "text/html; charset=utf-8")
                elif url.path == "/api/health":
                    # 2026-10-17: who is running, since when, on which code (the ComfyUI panel's editor switch)
                    self._json({"ok": True, "pid": os.getpid(), "started": STARTED, "code": CODE_STAMP,
                                "parent": getattr(editor, "parent_pid", 0)})
                elif url.path == "/api/sweeps":
                    self._json(editor.sweeps.list())
                elif url.path == "/api/sweep":
                    self._json(editor.sweeps.cells(query.get("id", "")))
                elif url.path == "/api/stacks/summary":
                    self._json({"stacks": editor.sweeps.summary(), "pairs": editor.sweeps.pairs()})
                elif url.path == "/sharefile":
                    data = editor.share_file(query.get("id", ""), query.get("file", ""))
                    if data is None:
                        self._send(404, b"", "text/plain")
                    else:
                        self._send(200, data, "image/jpeg")
                elif url.path == "/sweepimg":
                    data = editor.sweeps.image(query.get("id", ""), query.get("file", ""), int(query.get("side", "512")))
                    if data is None:
                        self._send(404, b"", "text/plain")
                    else:
                        self._send(200, data, "image/jpeg")
                elif url.path == "/api/folders":
                    self._json(editor.organiser.folders())
                elif url.path == "/api/organise/batches":
                    self._json(editor.organiser.batches())
                elif url.path == "/api/organise/references":
                    self._json(editor.organiser.references(query.get("id", "")))
                elif url.path == "/api/organise/status":
                    self._json({"lora_manager_scan": editor.organiser.scan_state})
                elif url.path == "/api/prompt_words":
                    self._json(editor.prompt_words(query.get("path", "")))
                elif url.path == "/api/sweep_record":
                    self._json(editor.sweep_record(query.get("path", "")))
                elif url.path == "/api/card":
                    card = editor.card(query.get("path", ""))
                    self._json(card if card else {"error": "not found"}, 200 if card else 404)
                elif url.path == "/api/artists":
                    self._json(editor.artists(query.get("q", "")))
                elif url.path == "/api/creators":
                    self._json(editor.creators(query.get("q", "")))
                elif url.path == "/img":
                    data = editor.thumbnail(query.get("path", ""), int(query.get("i", "0")))
                    if data is None:
                        self._send(404, b"", "text/plain")
                    else:
                        self._send(200, data, "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as error:                      # keep the server up; show the error in the page
                self._json({"error": f"{type(error).__name__}: {error}"}, 500)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if url.path == "/api/style_search":               # the body is an image file, not JSON
                    if length > 40 * 1024 * 1024:
                        self._json({"error": "image too large (40 MB at most)"}, 413)
                        return
                    query = {k: v[0] for k, v in parse_qs(url.query).items()}
                    self._json(editor.style_search(self.rfile.read(length), query.get("shelf", ""),
                                                   query.get("order", "priority"), query.get("status", "all"),
                                                   query.get("rating", "any"), query.get("folder", "")))
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/api/save":
                    self._json(editor.save(body))
                elif url.path == "/api/add_term":
                    from .vocab import add_term
                    term = add_term(str(body.get("facet") or ""), str(body.get("word") or ""), str(body.get("parent") or ""))
                    self._json({"ok": True, "id": term})
                elif url.path == "/api/fetch_images":
                    self._json(editor.fetch_images(str(body.get("path") or "")))
                elif url.path == "/api/prompt_words":
                    self._json(editor.set_prompt_words(body))
                elif url.path == "/api/recipe_edit":
                    self._json(editor.edit_recipe(body))
                elif url.path == "/api/word_options":
                    self._json(editor.set_word_options(body))
                elif url.path == "/api/prompt_words/cell":
                    self._json(editor.pick_from_cell(body))
                elif url.path == "/api/stacks/mark":
                    # 2026-10-16: stars and tags too; a field left out keeps the cell's value, clear removes the mark
                    self._json(editor.sweeps.mark(
                        str(body.get("id") or ""), str(body.get("file") or ""),
                        body.get("verdict") if body.get("verdict") is not None else None, body.get("notes"),
                        stars=body.get("stars"), tags=body.get("tags"), clear=bool(body.get("clear")),
                        push=body.get("push")))     # 2026-10-17: stronger / weaker / '' (none)
                elif url.path == "/api/discover/dismiss":         # 2026-10-29: Not interested / undo
                    from . import discover
                    self._json(discover.dismiss(editor.settings, str(body.get("key") or ""), str(body.get("name") or ""),
                                                bool(body.get("undo"))))
                elif url.path == "/api/shutdown":
                    # 2026-10-17: the ComfyUI panel's Stop. Answer first, then stop serving (this machine only: the
                    # server listens on 127.0.0.1)
                    self._json({"ok": True, "pid": os.getpid()})
                    if getattr(editor, "httpd", None) is not None:
                        threading.Thread(target=editor.httpd.shutdown, daemon=True).start()
                elif url.path == "/api/stacks/update":
                    # 2026-10-17: prompt fit / type for older marks, then refit the marks model (markmodel.py)
                    filled = editor.sweeps.backfill()
                    self._json({"ok": True, **filled, **editor.sweeps.update_model()})
                elif url.path == "/api/stacks/import":
                    self._json(editor.sweeps.import_workbook(str(body.get("id") or "")))
                elif url.path == "/api/share_sheets":
                    self._json(editor.share_sheets(body))
                elif url.path == "/api/organise/plan":
                    self._json(editor.organise_plan(body))
                elif url.path == "/api/organise/apply":
                    self._json(editor.organise_done(editor.organiser.apply(str(body.get("id") or ""))))
                elif url.path == "/api/organise/undo":
                    self._json(editor.organise_done(editor.organiser.undo(str(body.get("id") or ""))))
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as error:
                self._json({"ok": False, "error": f"{type(error).__name__}: {error}"}, 500)

    return Handler


class EditorServer(ThreadingHTTPServer):
    """Owns its port outright. Windows otherwise lets a second server bind the same port, and requests keep going
    to whichever started first (an old, forgotten one)."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):            # Windows
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(port: int = 8765, open_browser: bool = True, parent_pid: int = 0) -> None:
    """parent_pid (2026-10-17): started from the ComfyUI panel, the editor stops by itself within ~10 s of ComfyUI
    closing, so no invisible editor is left running old code. 0 = runs until stopped (launch.bat, a terminal)."""
    settings = load_settings()
    editor = Editor(settings)
    editor.parent_pid = int(parent_pid or 0)
    try:
        server = EditorServer(("127.0.0.1", port), make_handler(editor))
    except OSError as error:
        print(f"[editor] port {port} is already in use ({error}).\n"
              f"[editor] Another card editor is probably still running: close it, or run card_editor.bat, which stops"
              f" old ones first.")
        raise SystemExit(1)
    editor.httpd = server
    if editor.parent_pid:
        def watch():
            while True:
                time.sleep(10)
                if not process_alive(editor.parent_pid):
                    print(f"[editor] ComfyUI (pid {editor.parent_pid}) has closed - stopping the card editor")
                    server.shutdown()
                    return
        threading.Thread(target=watch, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"[editor] {url}   (Ctrl+C to stop)  catalog: {settings.catalog_path}"
          + (f"  - stops when ComfyUI (pid {editor.parent_pid}) closes" if editor.parent_pid else ""))
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
