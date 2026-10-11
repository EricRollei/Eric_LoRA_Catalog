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
    GET  /sweeps                 the sweep verdict page (sweeps.html, stacks.py): mark cells good / bad / unsure
    GET  /api/sweeps             sweep folders        GET /api/sweep?id=   one sweep's cells and your marks
    GET  /sweepimg?id=&file=&side=512   a cell as JPEG    POST /api/stacks/mark  {id, file, verdict, notes}
    GET  /api/stacks/summary     marked stacks and LoRA pairs with good / bad counts
    POST /api/save               {path, facets, artists, creator, triggers, trigger_states, weight, weight_low,
                                  weight_high, rating, notes, unsure, other_changed}

Nothing here writes to the generated cards; the overnight card run can keep writing while you edit (sqlite WAL).

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sqlite3
import threading
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
from .stacks import Sweeps
from .vocab import NSFW_TERMS, _civitai_nsfw, load_vocab, prefill

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_PER_CARD = 4          # one page of the editor's 2x2 grid
MAX_IMAGES = 24              # how many it can page through
THUMB_SIDE = 720


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
        self.prefill_cache: Dict[tuple, Dict[str, List[str]]] = {}
        self.image_cache: Dict[tuple, bytes] = {}
        self.image_list_cache: Dict[str, List[str]] = {}
        self.artist_names: Optional[List[tuple]] = None
        self.creator_cache: Dict[Any, str] = {}
        self.search = SearchIndex(settings)
        self.organiser = Organiser(settings)
        self.sweeps = Sweeps(settings)
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
            return list(connection.execute(sql + " where replace(l.rel_dir, '\\', '/') like ?", (shelf + "%",)))
        return list(connection.execute(sql))

    def prefill_for(self, row: sqlite3.Row) -> Dict[str, List[str]]:
        vocab = load_vocab()
        key = (row["path"], row["refined_at"], id(vocab))
        if key not in self.prefill_cache:
            self.prefill_cache[key] = prefill(vocab, row, row)
        return self.prefill_cache[key]

    def priority(self, row: sqlite3.Row) -> float:
        """Higher = look at it sooner: low confidence, sparse cards, and an artist hinted by name but not on the card."""
        confidence = row["confidence"] or 0.0
        filled = self.prefill_for(row)
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

    def queue(self, shelf: str, order: str, status: str, rating: str = "any", q: str = "", folder: str = "") -> Any:
        """The list on the left. With a search `q`: {items, matched, mode, hidden} instead of a bare list, items in
        match order unless another order is picked. `folder` (a relative folder from the folder panel, any depth)
        replaces the shelf."""
        hits = self.search.search(q) if q.strip() else None
        folder = folder.replace("\\", "/").strip("/")
        with self.connect() as connection:
            rows = self.rows(connection, shelf="" if folder else shelf)
        items = []
        for row in rows:
            if hits is not None and row["path"] not in hits:
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
                          "status": state, "priority": self.priority(row),
                          "score": hits.get(row["path"]) if hits is not None else None})
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
        if hits is None:
            return items
        return {"items": items, "matched": len(hits), "hidden": len(hits) - len(items),
                "mode": self.search.last.get("mode", "none")}

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
                elif url.path == "/api/queue" and query.get("dupes") == "1":
                    self._json({"items": editor.duplicates(), "dupes": True})
                elif url.path == "/api/queue":
                    self._json(editor.queue(query.get("shelf", ""), query.get("order", "priority"), query.get("status", "todo"),
                                            query.get("rating", "any"), query.get("q", ""), query.get("folder", "")))
                elif url.path in ("/sweeps", "/sweeps.html"):
                    with open(os.path.join(HERE, "sweeps.html"), "rb") as handle:
                        self._send(200, handle.read(), "text/html; charset=utf-8")
                elif url.path == "/api/sweeps":
                    self._json(editor.sweeps.list())
                elif url.path == "/api/sweep":
                    self._json(editor.sweeps.cells(query.get("id", "")))
                elif url.path == "/api/stacks/summary":
                    self._json({"stacks": editor.sweeps.summary(), "pairs": editor.sweeps.pairs()})
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
                body = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/api/save":
                    self._json(editor.save(body))
                elif url.path == "/api/add_term":
                    from .vocab import add_term
                    term = add_term(str(body.get("facet") or ""), str(body.get("word") or ""), str(body.get("parent") or ""))
                    self._json({"ok": True, "id": term})
                elif url.path == "/api/fetch_images":
                    self._json(editor.fetch_images(str(body.get("path") or "")))
                elif url.path == "/api/stacks/mark":
                    self._json(editor.sweeps.mark(str(body.get("id") or ""), str(body.get("file") or ""),
                                                  body.get("verdict") or None, body.get("notes")))
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


def serve(port: int = 8765, open_browser: bool = True) -> None:
    settings = load_settings()
    editor = Editor(settings)
    try:
        server = EditorServer(("127.0.0.1", port), make_handler(editor))
    except OSError as error:
        print(f"[editor] port {port} is already in use ({error}).\n"
              f"[editor] Another card editor is probably still running: close it, or run card_editor.bat, which stops"
              f" old ones first.")
        raise SystemExit(1)
    url = f"http://127.0.0.1:{port}/"
    print(f"[editor] {url}   (Ctrl+C to stop)  catalog: {settings.catalog_path}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
