"""
Style search: what each LoRA LOOKS like, from its preview images, as a CSD style vector (docs/design/STYLE_SEARCH.md).

CSD - Contrastive Style Descriptors (Somepalli et al., ECCV 2024, arXiv 2404.01292; code MIT, github.com/learn2phoenix/
CSD; weights CC-BY-4.0, huggingface.co/tomg-group-umd/CSD-ViT-L): a CLIP ViT-L/14 fine-tuned so that images in one
style land close together whatever they show. A LoRA's style vector is the mean of its previews' vectors.

Optional and off by default. Needs torch (ComfyUI has it) and the weights (python run.py style-setup), and
settings.json "style_enabled": true. Without them every function here says why and does nothing; the rest of the
tool never imports torch.

    python run.py style-setup [--from pytorch_model.bin]   download (2.4 GB) and convert to safetensors (1.2 GB)
    python run.py style [--limit N] [--force]               vectors for LoRAs whose previews are new or changed
    python run.py style --show mignola                      the closest looks to one LoRA

Where things are
    style_vectors (catalog table)            one row per LoRA file (key = sha256): vector, previews used, agreement
    DATA_DIR/cache/style_images.sqlite       one vector per preview image, so changed galleries re-embed only new images
    DATA_DIR/models/csd_vit_l.safetensors    the weights (settings.json "style_model" points elsewhere)

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .config import DATA_DIR, Settings

# style_device "cuda:N" counts GPUs the way nvidia-smi does (torch's own default order is fastest-first). Set before
# torch is first imported in this process; style.py is only imported where style search is used.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

MODEL_ID = "csd-vit-l"
DIM = 768
PREVIEWS = 12                     # preview images per LoRA (the editor shows the same ones, least NSFW first)
LOOK_KINDS = ("style_medium", "photo_look", "lighting_colour")
NSFW_HIDDEN_FROM = 2              # card nsfw level (0-3) hidden in "similar" unless asked for (the editor's badge level)
HF_URL = "https://huggingface.co/tomg-group-umd/CSD-ViT-L/resolve/main/pytorch_model.bin"
IMAGE_CACHE = os.path.join(DATA_DIR, "cache", "style_images.sqlite")
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

SCHEMA = """
CREATE TABLE IF NOT EXISTS style_vectors (
    key        TEXT PRIMARY KEY,
    path       TEXT,
    model      TEXT,
    source     TEXT,
    n_images   INTEGER,
    agreement  REAL,
    vec        BLOB,
    images_sig TEXT,
    updated_at TEXT
);
"""


# -- set up? -------------------------------------------------------------------------------------------------------
def weights_path(settings: Settings) -> str:
    return str(getattr(settings, "style_model", "") or "").strip() or os.path.join(DATA_DIR, "models", "csd_vit_l.safetensors")


def available(settings: Settings, need_model: bool = True) -> str:
    """'' when style search can run, else the reason in plain words. need_model=False: only the stored vectors are
    needed (similar / twins work without torch)."""
    if not getattr(settings, "style_enabled", False):
        return "style search is off (settings.json \"style_enabled\": true turns it on)"
    if not need_model:
        return ""
    if not os.path.isfile(weights_path(settings)):
        return f"style model not set up ({weights_path(settings)} missing): run  python run.py style-setup"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "style search needs torch (ComfyUI's Python has it; a standalone install does not)"
    return ""


# Renders (2026-10-13): how each LoRA's sweep renders compare with its own preview look. One row per rendered cell;
# style_vectors.render_verdict sums them up per LoRA (RENDER_VERDICTS).
RENDER_SCHEMA = """
CREATE TABLE IF NOT EXISTS style_renders (
    key TEXT, sweep TEXT, prompt TEXT, strength REAL, kind TEXT, image TEXT, match REAL, base_match REAL, lift REAL,
    rank INTEGER, pool INTEGER, shows INTEGER, checked_at TEXT,
    PRIMARY KEY (key, sweep, prompt, strength)
);
"""
RENDER_COLUMNS = {"render_verdict": "TEXT", "render_note": "TEXT"}
# A render 'shows the look' when its own LoRA's preview signature is among the SHOWS_RANK closest of the platform's
# look LoRAs to it, and it is clearly closer to that signature than the no-LoRA render of the same prompt was.
SHOWS_RANK = 10
SHOWS_LIFT = 0.05
RENDER_VERDICTS = {
    "shows": "shows its look on a plain prompt",
    "needs_words": "shows its look only with its own style words in the prompt",
    "weak": "does not show its preview look in renders, even with its creator's prompt",
    "unclear": "did not show its look on a plain prompt - try its creator's prompt (second pass)",
}


def ensure(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA + RENDER_SCHEMA)
    have = {row[1] for row in connection.execute("pragma table_info(style_vectors)")}
    for column, kind in RENDER_COLUMNS.items():
        if column not in have:
            connection.execute(f"alter table style_vectors add column {column} {kind}")


# -- vectors as bytes (float16, no numpy needed to read) -----------------------------------------------------------
def pack(values) -> bytes:
    return struct.pack(f"<{DIM}e", *[float(v) for v in values])


def unpack(blob: bytes) -> List[float]:
    return list(struct.unpack(f"<{DIM}e", blob))


# -- the model -----------------------------------------------------------------------------------------------------
def _build_visual():
    """CLIP ViT-L/14 visual tower in OpenAI's layout (QuickGELU, class token, no output projection): the state-dict
    keys match CSD's `backbone.*` without renaming. Plain torch, no open_clip."""
    from collections import OrderedDict

    import torch
    from torch import nn

    class QuickGELU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)

    class Block(nn.Module):
        def __init__(self, width, heads):
            super().__init__()
            self.ln_1 = nn.LayerNorm(width)
            self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.ln_2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(width, width * 4)), ("gelu", QuickGELU()),
                                                  ("c_proj", nn.Linear(width * 4, width))]))

        def forward(self, x):
            y = self.ln_1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
            return x + self.mlp(self.ln_2(x))

    class Transformer(nn.Module):
        def __init__(self, width, layers, heads):
            super().__init__()
            self.resblocks = nn.Sequential(*[Block(width, heads) for _ in range(layers)])

        def forward(self, x):
            return self.resblocks(x)

    class Visual(nn.Module):
        def __init__(self, width=1024, layers=24, heads=16, patch=14, size=224):
            super().__init__()
            self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
            self.class_embedding = nn.Parameter(torch.zeros(width))
            self.positional_embedding = nn.Parameter(torch.zeros((size // patch) ** 2 + 1, width))
            self.ln_pre = nn.LayerNorm(width)
            self.transformer = Transformer(width, layers, heads)
            self.ln_post = nn.LayerNorm(width)

        def forward(self, x):
            x = self.conv1(x).flatten(2).transpose(1, 2)
            cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
            x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
            x = self.transformer(self.ln_pre(x))
            return self.ln_post(x[:, 0])

    return Visual()


def _device(settings: Settings) -> str:
    import torch
    wanted = str(getattr(settings, "style_device", "") or "auto").strip().lower()
    if wanted in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return wanted


class StyleModel:
    """The CSD image encoder. embed_files(paths) / embed_image(PIL image) -> unit 768-d vectors (lists of floats)."""

    def __init__(self, settings: Settings, device: Optional[str] = None):
        import torch
        from safetensors.torch import load_file

        self.torch = torch
        self.device = device or _device(settings)           # inside ComfyUI: ComfyUI's own device
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        state = load_file(weights_path(settings))
        backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.") and k != "backbone.proj"}
        self.visual = _build_visual()
        missing, unexpected = self.visual.load_state_dict(backbone, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"style weights do not fit the model: missing {missing[:4]}, unexpected {unexpected[:4]}")
        self.visual = self.visual.to(self.device, self.dtype).eval()
        self.style_proj = state["last_layer_style"].to(self.device, self.dtype)
        del state

    def tensor(self, image):
        """CSD's preprocessing in PIL: shorter side to 224 (bicubic), centre crop 224, CLIP mean/std."""
        torch = self.torch
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            size = (224, int(224 * height / width))
        else:
            size = (int(224 * width / height), 224)
        image = image.resize(size, 3)                                   # 3 = BICUBIC
        left, top = int(round((size[0] - 224) / 2.0)), int(round((size[1] - 224) / 2.0))
        image = image.crop((left, top, left + 224, top + 224))
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(224, 224, 3).permute(2, 0, 1).float() / 255
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        return (x - mean) / std

    def load_file(self, path: str):
        """A preview as a model input, or None when it cannot be read. A preview video gives one frame (the frame the
        editor shows; needs OpenCV, like the editor)."""
        from PIL import Image
        try:
            from .cards import VIDEO_EXTENSIONS, _video_frame
            if path.lower().endswith(VIDEO_EXTENSIONS):
                frame = _video_frame(path)
                return None if frame is None else self.tensor(frame)
            image = Image.open(path)
            image.draft("RGB", (448, 448))                              # fast JPEG decode at reduced size
            return self.tensor(image)
        except Exception:
            return None

    def embed_tensors(self, tensors, batch: int = 32) -> List[List[float]]:
        torch = self.torch
        out: List[List[float]] = []
        for start in range(0, len(tensors), batch):
            x = torch.stack(tensors[start:start + batch]).to(self.device, self.dtype)
            with torch.no_grad():
                style = torch.nn.functional.normalize((self.visual(x) @ self.style_proj).float(), dim=1)
            out.extend(style.cpu().tolist())
        return out

    def embed_image(self, image) -> List[float]:
        return self.embed_tensors([self.tensor(image)])[0]

    def close(self) -> None:
        del self.visual, self.style_proj
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


# -- per-image cache -----------------------------------------------------------------------------------------------
def _image_key(path: str) -> str:
    stat = os.stat(path)
    return f"{path}|{stat.st_mtime:.0f}|{stat.st_size}"


class ImageCache:
    def __init__(self):
        os.makedirs(os.path.dirname(IMAGE_CACHE), exist_ok=True)
        self.db = sqlite3.connect(IMAGE_CACHE)
        self.db.execute("CREATE TABLE IF NOT EXISTS img (key TEXT PRIMARY KEY, model TEXT, vec BLOB)")

    def get(self, keys: List[str]) -> Dict[str, bytes]:
        found = {}
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            marks = ",".join("?" * len(part))
            for key, blob in self.db.execute(f"SELECT key, vec FROM img WHERE model = ? AND key IN ({marks})",
                                             [MODEL_ID, *part]):
                found[key] = blob
        return found

    def put(self, items: List[Tuple[str, bytes]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO img (key, model, vec) VALUES (?, ?, ?)",
                            [(k, MODEL_ID, v) for k, v in items])
        self.db.commit()

    def close(self) -> None:
        self.db.close()


# -- computing LoRA vectors (refresh step) -------------------------------------------------------------------------
def _signature(vectors: List[List[float]]) -> Tuple[List[float], Optional[float]]:
    """Unit mean of unit vectors, and the mean pairwise cosine among them (|sum|^2 = n + 2*sum of pairs)."""
    n = len(vectors)
    total = [sum(column) for column in zip(*vectors)]
    norm2 = sum(v * v for v in total)
    norm = norm2 ** 0.5 or 1.0
    agreement = (norm2 - n) / (n * (n - 1)) if n > 1 else None
    return [v / norm for v in total], agreement


def compute(settings: Settings, limit: int = 0, force: bool = False, log=print) -> Dict[str, int]:
    """Vectors for every LoRA whose preview images changed since last time (all of them with force)."""
    from .cards import preview_images

    counts = {"loras": 0, "unchanged": 0, "no_previews": 0, "images_embedded": 0, "written": 0, "unreadable": 0}
    reason = available(settings)
    if reason:
        log(f"[style] skipped: {reason}")
        counts["skipped"] = 1
        return counts
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    ensure(connection)
    known = {key: sig for key, sig in connection.execute("SELECT key, images_sig FROM style_vectors WHERE model = ?",
                                                         (MODEL_ID,))}
    rows = connection.execute("SELECT path, coalesce(lower(sha256), path) FROM loras ORDER BY path").fetchall()
    cache = ImageCache()
    todo: List[Tuple[str, str, List[str], List[str], str]] = []      # path, key, images, image keys, sig
    for path, key in rows:
        counts["loras"] += 1
        try:
            images = preview_images(path, PREVIEWS)
            image_keys = [_image_key(p) for p in images]
        except OSError:
            images, image_keys = [], []
        if not images:
            counts["no_previews"] += 1
            continue
        sig = hashlib.sha1("\n".join(image_keys).encode("utf-8")).hexdigest()
        if not force and known.get(key) == sig:
            counts["unchanged"] += 1
            continue
        todo.append((path, key, images, image_keys, sig))
        if limit and len(todo) >= limit:
            break
    have = cache.get([k for item in todo for k in item[3]])
    need = sorted({(p, k) for item in todo for p, k in zip(item[2], item[3]) if k not in have})
    if need:
        started = time.time()
        log(f"[style] loading the style model for {len(need)} new images ({len(todo)} LoRAs)")
        model = StyleModel(settings)
        log(f"[style] model on {model.device} in {time.time() - started:.0f}s")
        step = 256
        with ThreadPoolExecutor(8) as pool:
            for start in range(0, len(need), step):
                chunk = need[start:start + step]
                tensors = list(pool.map(model.load_file, [p for p, _k in chunk]))
                good = [(k, t) for (_p, k), t in zip(chunk, tensors) if t is not None]
                counts["unreadable"] += len(chunk) - len(good)
                vectors = model.embed_tensors([t for _k, t in good])
                items = [(k, pack(v)) for (k, _t), v in zip(good, vectors)]
                cache.put(items)
                have.update(items)
                counts["images_embedded"] += len(items)
                log(f"[style] images {min(start + step, len(need))}/{len(need)}  {time.time() - started:.0f}s")
        model.close()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    writes = []
    for path, key, _images, image_keys, sig in todo:
        vectors = [unpack(have[k]) for k in image_keys if k in have]
        if not vectors:
            continue
        signature, agreement = _signature(vectors)
        writes.append((key, path, MODEL_ID, "previews", len(vectors), agreement, pack(signature), sig, now))
    # an upsert, not INSERT OR REPLACE: a LoRA's render verdict (render_check) survives new preview images
    connection.executemany("INSERT INTO style_vectors (key, path, model, source, n_images, agreement, vec, images_sig, "
                           "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                           "path = excluded.path, model = excluded.model, source = excluded.source, "
                           "n_images = excluded.n_images, agreement = excluded.agreement, vec = excluded.vec, "
                           "images_sig = excluded.images_sig, updated_at = excluded.updated_at", writes)
    counts["written"] = len(writes)
    connection.commit()
    connection.close()
    cache.close()
    return counts


# -- reading: similar, twins, image search (no torch needed) -------------------------------------------------------
def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class StyleIndex:
    """The stored vectors plus what the editor needs to show them, reloaded when the catalog's vectors or cards change."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.stamp: Optional[tuple] = None
        self.items: List[Dict[str, Any]] = []
        self.by_path: Dict[str, int] = {}
        self.matrix = None            # numpy array when numpy is there, else None (pure-Python fallback)

    def _load(self) -> None:
        connection = sqlite3.connect(f"file:{self.settings.catalog_path}?mode=ro", uri=True, timeout=15)
        try:
            ensure_ro = connection.execute("SELECT name FROM sqlite_master WHERE name = 'style_vectors'").fetchone()
            if not ensure_ro:
                self.items, self.by_path, self.matrix, self.stamp = [], {}, None, None
                return
            stamp = tuple(connection.execute("SELECT count(*), max(updated_at) FROM style_vectors").fetchone()) + \
                tuple(connection.execute("SELECT count(*), max(updated_at) FROM card_facets").fetchone())
            have = {row[1] for row in connection.execute("pragma table_info(style_vectors)")}
            renders = "s.render_verdict, s.render_note" if "render_verdict" in have else "NULL, NULL"
            stamp += tuple(connection.execute("SELECT count(*), max(checked_at) FROM style_renders").fetchone()) \
                if renders != "NULL, NULL" and connection.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'style_renders'").fetchone() else ()
            if stamp == self.stamp:
                return
            rows = connection.execute(f"""
                SELECT l.path, l.model_name, l.version_name, l.stem, l.arch_folder, l.base_model,
                       f.name AS card_name, f.kind, f.nsfw, s.vec, s.n_images, s.agreement, s.images_sig,
                       l.civitai_model_id, {renders}
                FROM style_vectors s JOIN loras l ON coalesce(lower(l.sha256), l.path) = s.key
                LEFT JOIN card_facets f ON f.path = l.path
                WHERE s.model = ? AND coalesce(f.exclude, 0) = 0""", (MODEL_ID,)).fetchall()
            # 2026-10-14: LoRA Manager 'exclude' (VAEs, presets, tools, other-arch files on the shelf) = not a look
        finally:
            connection.close()
        items = []
        for r in rows:
            name = r[6] or " - ".join(x for x in (r[1], r[2]) if x) or r[3] or os.path.basename(r[0])
            items.append({"path": r[0], "name": name, "arch": r[4] or "", "base": r[5] or "", "kind": r[7] or "",
                          "nsfw": int(r[8] or 0), "vec": unpack(r[9]), "n": r[10], "agreement": r[11], "sig": r[12],
                          "model_id": r[13], "verdict": r[14] or "", "verdict_note": r[15] or ""})
        sig_count: Dict[str, int] = {}
        for item in items:
            sig_count[item["sig"]] = sig_count.get(item["sig"], 0) + 1
        for item in items:
            item["shared"] = sig_count[item["sig"]] > 1
        try:
            import numpy
            matrix = numpy.array([item["vec"] for item in items], dtype="float32") if items else None
        except ImportError:
            matrix = None
        self.items, self.matrix, self.stamp = items, matrix, stamp
        self.by_path = {item["path"]: i for i, item in enumerate(items)}

    def ensure(self) -> None:
        with self.lock:
            self._load()

    def _scores(self, vector: List[float]) -> List[float]:
        if self.matrix is not None:
            import numpy
            return (self.matrix @ numpy.asarray(vector, dtype="float32")).tolist()
        return [_dot(item["vec"], vector) for item in self.items]

    @staticmethod
    def _public(item: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {"path": item["path"], "name": item["name"], "kind": item["kind"], "arch": item["arch"],
                "base": item["base"], "nsfw": item["nsfw"], "score": round(score, 3), "look": item["kind"] in LOOK_KINDS}

    def info(self, path: str) -> Optional[Dict[str, Any]]:
        self.ensure()
        i = self.by_path.get(path)
        if i is None:
            return None
        item = self.items[i]
        same = [other["name"] for other in self.items if other["sig"] == item["sig"] and other["path"] != path]
        return {"n_images": item["n"], "agreement": None if item["agreement"] is None else round(item["agreement"], 3),
                "look": item["kind"] in LOOK_KINDS, "same_previews": same[:6],
                "render_verdict": item["verdict"], "render_label": RENDER_VERDICTS.get(item["verdict"], ""),
                "render_note": item["verdict_note"]}

    def similar(self, path: str, n: int = 8, scope: str = "base", kinds: str = "look", nsfw: bool = False) -> List[Dict[str, Any]]:
        """Closest looks to one LoRA. scope base = same arch folder (Krea2 with Krea2); kinds look = only style /
        photo-look / lighting LoRAs; nsfw False hides cards rated nudity and up. LoRAs with the very same previews
        are left out (previews cannot tell them apart; info() lists them)."""
        self.ensure()
        i = self.by_path.get(path)
        if i is None:
            return []
        me = self.items[i]
        scores = self._scores(me["vec"])
        out = []
        for j in sorted(range(len(self.items)), key=lambda k: -scores[k]):
            other = self.items[j]
            if j == i or other["sig"] == me["sig"]:
                continue
            if scope == "base" and other["arch"].lower() != me["arch"].lower():
                continue
            if kinds == "look" and other["kind"] not in LOOK_KINDS:
                continue
            if not nsfw and other["nsfw"] >= NSFW_HIDDEN_FROM:
                continue
            out.append(self._public(other, scores[j]))
            if len(out) >= n:
                break
        return out

    def rank(self, vector: List[float]) -> Dict[str, float]:
        """{path: cosine} for every LoRA with a vector (image search, like:)."""
        self.ensure()
        return {item["path"]: round(s, 4) for item, s in zip(self.items, self._scores(vector))}

    def vector(self, path: str) -> Optional[List[float]]:
        self.ensure()
        i = self.by_path.get(path)
        return None if i is None else self.items[i]["vec"]

    def twins(self, minimum: float = 0.85, arch: str = "", nsfw: bool = False, limit: int = 300) -> List[Dict[str, Any]]:
        """Pairs of look-type LoRAs at or above `minimum`, highest first (needs numpy). Versions of the same Civitai
        model and files with the same previews are left out."""
        self.ensure()
        if self.matrix is None:
            return []
        import numpy
        keep = [i for i, item in enumerate(self.items) if item["kind"] in LOOK_KINDS
                and (not arch or item["arch"].lower() == arch.lower()) and (nsfw or item["nsfw"] < NSFW_HIDDEN_FROM)]
        if len(keep) < 2:
            return []
        sub = self.matrix[keep]
        sims = numpy.triu(sub @ sub.T, 1)
        pairs = []
        for a, b in zip(*numpy.where(sims >= minimum)):
            x, y = self.items[keep[a]], self.items[keep[b]]
            if x["sig"] == y["sig"] or x["arch"].lower() != y["arch"].lower():
                continue
            if x["model_id"] and x["model_id"] == y["model_id"]:     # versions of one Civitai model: alike by design
                continue
            pairs.append((float(sims[a, b]), x, y))
        pairs.sort(key=lambda p: -p[0])
        return [{"score": round(s, 3), "a": self._public(x, s), "b": self._public(y, s)} for s, x, y in pairs[:limit]]


# -- the model inside a long-running process (the editor): load on first use, unload when idle ---------------------
class ModelHolder:
    """device None = settings style_device (the card editor); the Suggester (Advanced) node passes ComfyUI's device."""

    def __init__(self, settings: Settings, device: Optional[str] = None):
        self.settings = settings
        self.device = device
        self.lock = threading.Lock()
        self.model: Optional[StyleModel] = None
        self.last_used = 0.0
        self.timer: Optional[threading.Timer] = None
        self.seen: Dict[str, List[float]] = {}            # image content hash -> vector (re-runs cost nothing)

    def embed_image(self, image) -> List[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images) -> List[List[float]]:
        keys = [hashlib.sha1(image.convert("RGB").tobytes()).hexdigest() + f"{image.size}" for image in images]
        with self.lock:
            todo = [(k, im) for k, im in zip(keys, images) if k not in self.seen]
            if todo:
                if self.model is None:
                    self.model = StyleModel(self.settings, device=self.device)
                vectors = self.model.embed_tensors([self.model.tensor(im) for _k, im in todo])
                if len(self.seen) > 256:
                    self.seen.clear()
                self.seen.update({k: v for (k, _im), v in zip(todo, vectors)})
            self.last_used = time.time()
            out = [self.seen[k] for k in keys]
        self._schedule()
        return out

    def _schedule(self) -> None:
        minutes = float(getattr(self.settings, "style_idle_unload_min", 10) or 10)
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(minutes * 60, self._idle)
        self.timer.daemon = True
        self.timer.start()

    def _idle(self) -> None:
        with self.lock:
            if self.model is not None:
                self.model.close()
                self.model = None


# -- setup: download and convert the weights -----------------------------------------------------------------------
def _recycle(path: str) -> bool:
    """Send a file to the Windows Recycle Bin. False elsewhere (the file is left in place)."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
    operation = SHFILEOPSTRUCTW(None, 3, os.path.abspath(path) + "\0", None, 0x0040 | 0x0010 | 0x0004 | 0x0400,
                                False, None, None)                   # FO_DELETE, ALLOWUNDO|NOCONFIRMATION|SILENT|NOERRORUI
    return ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation)) == 0


def convert(source: str, target: str) -> str:
    """CSD's training checkpoint (a pickle) -> safetensors with only the weights style search uses. Loaded with torch's
    safe unpickler, allowing just the plain types the checkpoint holds (its saved arguments and numpy scalars)."""
    import argparse

    import numpy
    import torch
    from safetensors.torch import save_file

    allowed: List[Any] = [argparse.Namespace, numpy.dtype]
    scalar = getattr(getattr(numpy, "_core", None), "multiarray", None)
    if scalar is not None:
        allowed.append((scalar.scalar, "numpy.core.multiarray.scalar"))
    else:
        allowed.append(numpy.core.multiarray.scalar)
    allowed += [type(numpy.dtype(t)) for t in ("float64", "float32", "int64", "int32", "bool")]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items() if torch.is_tensor(v)}
    keep = {k: v.contiguous() for k, v in state.items() if k.startswith("backbone.") or k == "last_layer_style"}
    if "last_layer_style" not in keep or not any(k.startswith("backbone.transformer") for k in keep):
        raise RuntimeError(f"{source} does not look like a CSD checkpoint")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    save_file(keep, target)
    return target


def setup(settings: Settings, from_file: str = "", log=print) -> int:
    """Download (unless from_file) and convert the weights to weights_path(settings). Nothing runs without being asked:
    the refresh never downloads."""
    try:
        import torch  # noqa: F401
    except ImportError:
        log("[style-setup] needs torch - run it with ComfyUI's Python")
        return 1
    target = weights_path(settings)
    if os.path.isfile(target):
        log(f"[style-setup] already set up: {target}")
        return 0
    source = from_file
    downloaded = False
    if not source:
        import requests
        source = os.path.join(DATA_DIR, "models", "csd_vit_l_checkpoint.bin")
        os.makedirs(os.path.dirname(source), exist_ok=True)
        have = os.path.getsize(source) if os.path.isfile(source) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        log(f"[style-setup] downloading CSD ViT-L (2.4 GB, CC-BY-4.0, tomg-group-umd) to {source}")
        with requests.get(HF_URL, headers=headers, stream=True, timeout=60) as response:
            if response.status_code == 416:
                pass                                                   # already complete
            else:
                response.raise_for_status()
                mode = "ab" if have and response.status_code == 206 else "wb"
                done = have if mode == "ab" else 0
                total = done + int(response.headers.get("Content-Length") or 0)
                last = time.time()
                with open(source, mode) as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            log(f"[style-setup] {done >> 20} / {total >> 20} MB")
                            last = time.time()
        downloaded = True
    log(f"[style-setup] converting to {target}")
    convert(source, target)
    log(f"[style-setup] done ({os.path.getsize(target) >> 20} MB). Turn it on with settings.json \"style_enabled\": true, "
        f"then run  python run.py style")
    if downloaded:
        log("[style-setup] the download was moved to the Recycle Bin" if _recycle(source)
            else f"[style-setup] you can delete the download: {source}")
    return 0


# -- renders: does a LoRA show its preview look in our own sweep renders? -----------------------------------------
def _sweep_cells(sweep_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from .sweepfiles import manifest_path
    path = manifest_path(sweep_dir)                # sweep_manifest.json or sweep_manifest_<name>.json
    if not path:
        raise ValueError(f"no sweep manifest in {sweep_dir}")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest, [c for c in manifest.get("cells") or [] if c.get("file")]


def _find_lora(connection: sqlite3.Connection, name: str) -> Optional[Tuple[str, str, str]]:
    """A sweep's lora_file (bare file name, stem, loras-folder path or full path) -> (catalog path, key, arch)."""
    wanted = name.replace("\\", "/").strip().lower()
    base = wanted.rsplit("/", 1)[-1]
    stem = os.path.splitext(base)[0] if base.endswith((".safetensors", ".ckpt", ".pt")) else base
    rows = connection.execute(
        "SELECT path, coalesce(lower(sha256), path), arch_folder FROM loras WHERE lower(file_name) = ? OR lower(stem) = ?",
        (base, stem)).fetchall()
    if not rows:
        return None
    rows.sort(key=lambda r: (not r[0].replace("\\", "/").lower().endswith(wanted), r[0]))
    return rows[0][0], rows[0][1], rows[0][2] or ""


def _render_verdict(rows: List[Tuple]) -> Tuple[str, str]:
    """rows: (kind, match, lift, rank, pool, shows, sweep) for one LoRA -> (verdict, note)."""
    def best(kind: str):
        mine = [r for r in rows if r[0] == kind]
        if not mine:
            return None, ""
        top = min(mine, key=lambda r: (r[3], -(r[2] or 0)))
        sweeps = len({r[6] for r in mine})
        lift = "" if top[2] is None else f", lift {top[2]:+.2f}"
        return mine, (f"{kind}: best rank {top[3]} of {top[4]}{lift} ({len(mine)} render"
                      f"{'s' if len(mine) != 1 else ''}, {sweeps} sweep{'s' if sweeps != 1 else ''})")
    neutral, n_note = best("neutral")
    creator, c_note = best("creator")
    note = "; ".join(x for x in (n_note, c_note) if x)
    if neutral and any(r[5] for r in neutral):
        return "shows", note
    if creator and any(r[5] for r in creator):
        return "needs_words", note
    if creator:
        return "weak", note
    return "unclear", note


def render_check(settings: Settings, sweep_dir: str, kind: str = "auto", log=print) -> Dict[str, Any]:
    """Compare each LoRA cell of a finished sweep with the LoRA's own preview look, store one style_renders row per
    cell and update the LoRA's render_verdict. kind: neutral = the sweep used plain prompts (the overnight check);
    creator = each LoRA ran its creator's prompt (the second pass); auto = creator for combos sweeps, else neutral.
    Stack cells are skipped (a stack's look is not one LoRA's look)."""
    reason = available(settings)
    if reason:
        raise ValueError(reason)
    manifest, cells = _sweep_cells(sweep_dir)
    sweep_id = manifest.get("sweep_id") or os.path.basename(os.path.normpath(sweep_dir))
    if kind == "auto":
        kind = "creator" if (manifest.get("plan") or {}).get("mode") == "combos" else "neutral"
    baselines: Dict[str, str] = {}
    lora_cells = []
    for cell in cells:
        over = cell.get("overrides") or {}
        image = os.path.join(sweep_dir, cell["file"])
        if not os.path.isfile(image):
            continue
        prompt = str(over.get("prompt") or "")
        if over.get("lora_baseline") and not over.get("lora_file"):
            baselines.setdefault(prompt, image)
        elif over.get("lora_file"):
            lora_cells.append((str(over["lora_file"]), float(over.get("lora_strength") or 1.0), prompt, image))
    if not lora_cells:
        raise ValueError(f"{sweep_dir}: no single-LoRA cells with images (stack cells are not checked)")

    index = StyleIndex(settings)
    index.ensure()
    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    ensure(connection)
    found: Dict[str, Optional[Tuple[str, str, str]]] = {}
    for name, *_rest in lora_cells:
        if name not in found:
            found[name] = _find_lora(connection, name)

    # embed every render once (the per-image cache makes a re-check free)
    images = sorted({c[3] for c in lora_cells} | set(baselines.values()))
    cache = ImageCache()
    keys = {p: _image_key(p) for p in images}
    have = cache.get(list(keys.values()))
    need = [p for p in images if keys[p] not in have]
    if need:
        log(f"[style] embedding {len(need)} renders")
        model = StyleModel(settings)
        tensors = [model.load_file(p) for p in need]
        good = [(p, t) for p, t in zip(need, tensors) if t is not None]
        items = [(keys[p], pack(v)) for (p, _t), v in zip(good, model.embed_tensors([t for _p, t in good]))]
        cache.put(items)
        have.update(items)
        model.close()
    cache.close()
    vec = {p: unpack(have[k]) for p, k in keys.items() if k in have}

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    report: List[Dict[str, Any]] = []
    writes = []
    skipped: Dict[str, str] = {}
    for name, strength, prompt, image in lora_cells:
        hit = found.get(name)
        if hit is None:
            skipped[name] = "not in the catalog"
            continue
        path, key, arch = hit
        own_i = index.by_path.get(path)
        if own_i is None:
            skipped[name] = "no style vector (no previews)"
            continue
        if image not in vec:
            skipped[name] = "render could not be read"
            continue
        me = index.items[own_i]
        render = vec[image]
        scores = index._scores(render)
        match = scores[own_i]
        base_image = baselines.get(prompt)
        base_match = _dot(vec[base_image], me["vec"]) if base_image in vec else None
        lift = None if base_match is None else match - base_match
        pool_sigs: Dict[str, float] = {}                   # one entry per preview set (shared previews count once)
        for j, other in enumerate(index.items):
            if other["sig"] == me["sig"] or other["kind"] not in LOOK_KINDS:
                continue
            if other["arch"].lower() != (me["arch"] or arch).lower():
                continue
            pool_sigs[other["sig"]] = max(pool_sigs.get(other["sig"], -2.0), scores[j])
        rank = 1 + sum(1 for s in pool_sigs.values() if s > match)
        pool = len(pool_sigs) + 1
        shows = rank <= SHOWS_RANK and (lift is None or lift >= SHOWS_LIFT)
        writes.append((key, sweep_id, prompt, strength, kind, os.path.basename(image), round(match, 4),
                       None if base_match is None else round(base_match, 4), None if lift is None else round(lift, 4),
                       rank, pool, int(shows), now))
        report.append({"name": me["name"], "path": path, "key": key, "strength": strength, "match": match,
                       "base_match": base_match, "lift": lift, "rank": rank, "pool": pool, "shows": shows})
    connection.executemany("INSERT OR REPLACE INTO style_renders (key, sweep, prompt, strength, kind, image, match, "
                           "base_match, lift, rank, pool, shows, checked_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", writes)
    verdicts: Dict[str, Tuple[str, str]] = {}
    for key in {w[0] for w in writes}:
        rows = connection.execute("SELECT kind, match, lift, rank, pool, shows, sweep FROM style_renders WHERE key = ?",
                                  (key,)).fetchall()
        verdicts[key] = _render_verdict(rows)
        connection.execute("UPDATE style_vectors SET render_verdict = ?, render_note = ? WHERE key = ?",
                           (verdicts[key][0], verdicts[key][1], key))
    connection.commit()
    connection.close()
    for item in report:
        item["verdict"] = verdicts[item["key"]][0]
    return {"sweep": sweep_id, "kind": kind, "cells": report, "skipped": skipped,
            "baselines": len(baselines)}


def render_report(result: Dict[str, Any]) -> str:
    lines = [f"{result['sweep']}  ({result['kind']} prompts, {len(result['cells'])} LoRA cells, "
             f"{result['baselines']} no-LoRA reference{'s' if result['baselines'] != 1 else ''})",
             f"  shows = own look among the {SHOWS_RANK} closest of the platform's look LoRAs and "
             f">= {SHOWS_LIFT:+.2f} closer than the no-LoRA render", ""]
    for c in sorted(result["cells"], key=lambda c: (c["rank"], -(c["lift"] or 0))):
        lift = "   n/a" if c["lift"] is None else f"{c['lift']:+.2f}"
        lines.append(f"  {'SHOWS' if c['shows'] else '  -  '}  rank {c['rank']:>4}/{c['pool']:<4} match {c['match']:.2f} "
                     f"lift {lift}  @{c['strength']:g}  {c['name'][:60]}  -> {c['verdict']}")
    for name, why in result["skipped"].items():
        lines.append(f"  skipped  {name}: {why}")
    counts: Dict[str, int] = {}
    for key, verdict in {c["key"]: c["verdict"] for c in result["cells"]}.items():
        counts[verdict] = counts.get(verdict, 0) + 1
    lines.append("")
    lines.append("  verdicts: " + ", ".join(f"{v} {n}" for v, n in sorted(counts.items())))
    return "\n".join(lines)


# -- the second pass: LoRAs that did not show their look on a plain prompt, with their creator's prompt -------------
LORA_TAG = re.compile(r"<(?:lora|lyco|hypernet):[^>]*>", re.I)
COMMUNITY_LEVEL = {"None": 0, "Soft": 1, "Mature": 2, "X": 3}
PROMPT_MAX = 1200                                    # characters; Krea 2 reads long prompts, but not essays


def _clean_prompt(text: str) -> str:
    from .sweepplan import MOJIBAKE
    text = LORA_TAG.sub(" ", (text or "").replace("\n", " "))
    if MOJIBAKE.search(text):
        return ""
    text = re.sub(r"\*{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" ,.")
    if len(text) > PROMPT_MAX:                         # long creator prompts: keep whole sentences up to the limit
        cut = text[:PROMPT_MAX]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:end + 1] if end > PROMPT_MAX // 2 else cut.rsplit(" ", 1)[0].rstrip(" ,")
    return text if len(text.split()) >= 6 else ""


def creator_prompts(connection: sqlite3.Connection, path: str, version_id: Optional[int], n: int = 1,
                    nsfw: bool = False) -> List[str]:
    """The creator's own prompts for a LoRA: its Civitai example images first (least NSFW first), then the creator's
    harvested prompts (run.py community). R and X prompts only with nsfw."""
    out: List[str] = []
    for prompt, level in connection.execute(
            "SELECT prompt, coalesce(nsfw_level, 0) FROM examples WHERE path = ? AND prompt IS NOT NULL AND prompt <> '' "
            "ORDER BY coalesce(nsfw_level, 0), media_file", (path,)):
        if level >= 4 and not nsfw:                    # Civitai levels: 1 PG, 2 PG-13, 4 R, 8 X
            continue
        text = _clean_prompt(prompt)
        if text and text not in out:
            out.append(text)
        if len(out) >= n:
            return out
    if version_id:
        try:
            rows = connection.execute("SELECT cleaned, prompt, nsfw_level, picked FROM community_prompts "
                                      "WHERE version_id = ? AND source = 'creator'", (version_id,)).fetchall()
        except sqlite3.Error:
            rows = []
        rows.sort(key=lambda r: (COMMUNITY_LEVEL.get(r[2], 0), -(r[3] or 0)))
        for cleaned, prompt, level, _picked in rows:
            if COMMUNITY_LEVEL.get(level, 0) >= 2 and not nsfw:
                continue
            text = _clean_prompt(cleaned or prompt)
            if text and text not in out:
                out.append(text)
            if len(out) >= n:
                break
    return out


def second_pass(settings: Settings, arch: str = "Krea2", limit: int = 0, prompts: int = 1, nsfw: bool = False,
                name: str = "") -> Dict[str, Any]:
    """A combos sweep preset for the LoRAs whose verdict is 'unclear': each runs its creator's prompt, beside a no-LoRA
    render of that same prompt. Run it, then  run.py style --renders <the sweep folder>  turns 'unclear' into
    needs_words or weak. Returns {name, preset, cells, notes, loras, missing} like sweepplan.plan."""
    from datetime import datetime

    from .sweepplan import GRID_FIELDS_BLANK

    connection = sqlite3.connect(settings.catalog_path, timeout=30)
    ensure(connection)
    try:
        stem_counts: Dict[str, int] = {}
        for (file_name,) in connection.execute("SELECT lower(file_name) FROM loras"):
            stem_counts[file_name] = stem_counts.get(file_name, 0) + 1
        rows = connection.execute("""
            SELECT l.path, l.file_name, l.root, l.civitai_version_id, l.arch_folder,
                   coalesce(f.name, l.model_name, l.stem), f.weight_typical, coalesce(f.nsfw, 0)
            FROM style_vectors s JOIN loras l ON coalesce(lower(l.sha256), l.path) = s.key
            LEFT JOIN card_facets f ON f.path = l.path
            WHERE s.render_verdict = 'unclear' AND coalesce(f.exclude, 0) = 0 ORDER BY l.path""").fetchall()
        combos, loras, missing, seen = [], [], [], set()
        for path, file_name, root, version_id, folder, title, weight, card_nsfw in rows:
            if arch and (folder or "").lower() != arch.lower():
                continue
            if card_nsfw >= NSFW_HIDDEN_FROM and not nsfw:
                continue
            if path in seen:
                continue
            seen.add(path)
            texts = creator_prompts(connection, path, version_id, prompts, nsfw)
            if not texts:
                missing.append(title)
                continue
            line_name = file_name if stem_counts.get(file_name.lower(), 0) == 1 else \
                (os.path.relpath(path, root) if root else path)
            strength = round(float(weight), 2) if weight not in (None, 0) else 1.0
            for text in texts:
                combos.append({"prompt": text, "lora_baseline": True})
                combos.append({"prompt": text, "lora_file": line_name, "lora_strength": strength})
            loras.append((title, strength, len(texts)))
            if limit and len(loras) >= limit:
                break
    finally:
        connection.close()
    if not combos:
        raise ValueError("no 'unclear' LoRAs with a creator prompt" + (f" under {arch}" if arch else "") +
                         (f" ({len(missing)} unclear ones have no creator prompt yet)" if missing else "") +
                         " - run  run.py style --renders <overnight sweep folder>  first")
    notes = (f"Style check, second pass ({datetime.now():%Y-%m-%d}): {len(loras)} LoRAs that did not show their "
             f"preview look on the plain portrait prompt, each with its creator's own prompt beside a no-LoRA render of "
             f"that prompt ({len(combos)} cells). When it is done:  run.py style --renders <this sweep folder>.")
    sweep = {field: "" for field in GRID_FIELDS_BLANK}
    sweep.update({"mode": "combos", "stage_scope": "s1", "lora_mode": "none", "lora_files": "",
                  "lora_strengths": "", "include_baseline": True, "keep_panel_stack": False, "lora_triggers": "catalog",
                  "prompts": "", "combos_json": "\n".join(json.dumps(c, ensure_ascii=False) for c in combos),
                  "seed_policy": "fixed", "seeds": "", "quick_screen": False, "score_metrics": True,
                  "sort_by": "plan_order", "max_combos": len(combos), "preset_notes": notes})
    return {"name": name or f"Style check - second pass (creator prompts) {datetime.now():%m-%d}",
            "preset": {"sweep": sweep}, "cells": len(combos), "notes": notes, "loras": loras, "missing": missing}


def show(settings: Settings, words: str, n: int = 10, scope: str = "base") -> str:
    """Text report: the closest looks to the LoRA whose name best matches `words`."""
    index = StyleIndex(settings)
    index.ensure()
    words_l = words.lower()
    hits = [item for item in index.items if words_l in item["name"].lower() or words_l in os.path.basename(item["path"]).lower()]
    if not hits:
        return f"no LoRA with a style vector matches '{words}'"
    lines = []
    for item in hits[:3]:
        info = index.info(item["path"]) or {}
        lines.append(f"{item['name']}  [{item['arch']}, {item['kind'] or 'no card'}, {item['n']} previews, "
                     f"agreement {info.get('agreement')}]")
        if info.get("same_previews"):
            lines.append(f"   same previews as: {', '.join(info['same_previews'])}")
        for other in index.similar(item["path"], n=n, scope=scope, kinds="all", nsfw=True):
            lines.append(f"   {other['score']:.2f}  {other['name']}  ({other['kind']}{', nsfw' if other['nsfw'] >= NSFW_HIDDEN_FROM else ''})")
    if len(hits) > 3:
        lines.append(f"... {len(hits) - 3} more match '{words}'")
    return "\n".join(lines)
