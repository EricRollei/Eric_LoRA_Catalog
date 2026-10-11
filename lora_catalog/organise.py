"""
Moving LoRAs and folders around the library from the card editor, with the catalog kept in step (2026-10-09).
Spec: CARD_EDITOR_ORGANISE_SPEC.md.

    org = Organiser(settings)
    org.folders()                                  -> the organising folders under each library root, with counts
    plan = org.plan_move([lora paths], target_dir) -> what would happen; nothing touches the disk
    plan = org.plan_rename_folder(folder, new_folder_path)      (a rename and a move of a folder are the same thing)
    plan = org.plan_new_folder(parent, name)
    plan = org.plan_delete([lora paths])         -> move them (with their files) to the trash folder beside the library
    result = org.apply(plan["id"])                 -> does it, writes the undo manifest, updates the catalog
    org.undo(batch_id)                             -> plays a manifest backwards
    org.references(batch_id)                       -> saved workflows / notes that still name a moved LoRA

Rules (all from the spec)
    Plan first, apply second. Nothing is ever deleted except an empty downloader model folder whose last version left.
    Delete (2026-10-09, Eric's choice) = move to <library root's parent>/_lora_trash/<date_time>/<same relative path>:
    out of ComfyUI's lists, undoable from the editor, emptied by hand. Its catalog rows are removed and kept in the
    batch manifest so undo puts them back exactly; your card edit (keyed by sha256) stays for any other copy.
    Moves stay inside one library root (same drive, so every move is a rename: instant, never half-copied).
    A Civitai-downloader LoRA moves as its whole version folder ({modelId}-{Name}\\{versionId}-{Version}); a LoRA in a
    flat folder moves with its <stem>.* companions and images\\<stem>.* / images\\<stem>_*.
    A name already taken at the target = that LoRA is skipped, never overwritten or renamed.
    A file Windows will not let go of (ComfyUI has it loaded) = that LoRA is skipped; its files that already moved go
    back, the rest of the batch carries on.
    LoRA Manager's <stem>.metadata.json gets the new file_path / preview_url, and LoRA Manager is asked to rescan when
    ComfyUI is running.
    Catalog: every path-keyed table is rewritten, card_edits.path too (edits are keyed by sha256 and need nothing
    else), card_facets is refreshed, and path_history keeps old path -> new path for anything that still holds an old
    one.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
DOWNLOADER_FOLDER = re.compile(r"^\d{3,}-")          # {modelId}-{Name} and {versionId}-{Version}
BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
PATH_TABLES = ["loras", "examples", "cards", "card_facets", "slider_cards", "file_hashes", "harvest_seeds"]
HISTORY = """
CREATE TABLE IF NOT EXISTS path_history (
    old_path TEXT, new_path TEXT, sha256 TEXT, batch TEXT, moved_at TEXT, undone INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS path_history_old ON path_history(old_path);
"""
COMPANION_DIRS = {"images", "videos", "workflows"}   # a LoRA's own gallery folders, not places to organise into
REFERENCE_EXTENSIONS = (".json", ".txt", ".yaml", ".yml", ".csv", ".md")


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _inside(path: str, folder: str) -> bool:
    """True when `path` is `folder` itself or anything below it."""
    path, folder = _norm(path), _norm(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def check_name(name: str) -> Optional[str]:
    """Why a folder name is not allowed on Windows, or None when it is fine."""
    if not name or not name.strip():
        return "the name is empty"
    if BAD_NAME.search(name):
        return 'a folder name cannot contain < > : " / \\ | ? *'
    if name != name.strip() or name.endswith("."):
        return "a folder name cannot start or end with a space, or end with a dot"
    if name.split(".")[0].upper() in RESERVED:
        return f"{name} is a reserved name on Windows"
    if len(name) > 120:
        return "the name is longer than 120 characters"
    return None


class Organiser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.roots = [os.path.normpath(root) for root in settings.library_roots if os.path.isdir(root)]
        self.plans: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.scan_state: Dict[str, Any] = {"state": "idle"}
        self._folders: Optional[List[Dict[str, Any]]] = None
        self._folders_at = 0.0
        os.makedirs(settings.moves_dir, exist_ok=True)
        try:                                   # once here, so reads never wait for a write lock
            connection = sqlite3.connect(settings.catalog_path, timeout=30)
            connection.executescript(HISTORY)
            connection.close()
        except sqlite3.Error as error:
            print(f"[organise] path_history not created yet: {error}")

    # -- where things are -------------------------------------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.catalog_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def root_of(self, path: str) -> Optional[str]:
        for root in self.roots:
            if _inside(path, root) and _norm(path) != _norm(root):
                return root
        return None

    def skipped_dir(self, path: str) -> bool:
        root = self.root_of(path)
        if root is None:
            return True
        parts = os.path.relpath(path, root).split(os.sep)
        return any(part in self.settings.skip_dirs for part in parts)

    def organising_folder(self, path: str) -> Optional[str]:
        """Why `path` cannot be used as a folder to organise into, or None when it can."""
        root = self.root_of(path) if _norm(path) not in {_norm(r) for r in self.roots} else path
        if root is None:
            return "that folder is not inside a LoRA library"
        if self.skipped_dir(path) and _norm(path) != _norm(root):
            return "that folder is one the catalog leaves alone (settings skip_dirs)"
        relative = os.path.relpath(path, root)
        if relative != "." and any(DOWNLOADER_FOLDER.match(part) for part in relative.split(os.sep)):
            return "that is a downloader model / version folder, not an organising folder"
        if relative != "." and any(part.lower() in COMPANION_DIRS for part in relative.split(os.sep)):
            return "that is a LoRA's own images / videos / workflows folder"
        return None

    def folders(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Every organising folder (downloader model/version folders and skipped folders left out), with the number of
        catalogued LoRAs below it (all depths) and directly in it. Kept for 10 minutes; every change made here drops
        it (forget_folders)."""
        with self.lock:
            if not refresh and self._folders is not None and time.time() - self._folders_at < 600:
                return self._folders
        folders = self._build_folders()
        with self.lock:
            self._folders, self._folders_at = folders, time.time()
        return folders

    def forget_folders(self) -> None:
        with self.lock:
            self._folders = None

    def _build_folders(self) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            paths = [row[0] for row in connection.execute("select path from loras")]
        below: Dict[str, int] = {}
        here: Dict[str, int] = {}
        for path in paths:
            folder = os.path.dirname(path)
            here[_norm(folder)] = here.get(_norm(folder), 0) + 1
            while True:
                below[_norm(folder)] = below.get(_norm(folder), 0) + 1
                parent = os.path.dirname(folder)
                if parent == folder or self.root_of(folder) is None:
                    break
                folder = parent
        out = []
        for root in self.roots:
            for folder, dirs, _files in os.walk(root):
                keep = []
                for name in sorted(dirs, key=str.lower):
                    if name in self.settings.skip_dirs or name.startswith(".") or DOWNLOADER_FOLDER.match(name) \
                            or name.lower() in COMPANION_DIRS:
                        continue
                    keep.append(name)
                dirs[:] = keep
                relative = os.path.relpath(folder, root)
                out.append({"path": folder, "root": root, "rel": "" if relative == "." else relative.replace(os.sep, "/"),
                            "depth": 0 if relative == "." else relative.count(os.sep) + 1,
                            "loras": below.get(_norm(folder), 0), "direct": here.get(_norm(folder), 0)})
        return out

    # -- what moves with a LoRA -------------------------------------------------------------------------------------
    @staticmethod
    def _model_files(folder: str) -> List[str]:
        try:
            return [name for name in os.listdir(folder) if name.lower().endswith(MODEL_EXTENSIONS)]
        except OSError:
            return []

    def unit_for(self, path: str) -> Dict[str, Any]:
        """The files / folder that travel with one LoRA. kind 'version_folder' (the whole downloader folder) or 'flat'."""
        folder = os.path.dirname(path)
        name = os.path.basename(folder)
        if DOWNLOADER_FOLDER.match(name) and len(self._model_files(folder)) == 1:
            parent = os.path.dirname(folder)
            model_folder = os.path.basename(parent) if DOWNLOADER_FOLDER.match(os.path.basename(parent)) else ""
            return {"kind": "version_folder", "folder": folder, "model_folder": model_folder, "files": [folder]}
        stem = os.path.splitext(os.path.basename(path))[0]
        others = [os.path.splitext(n)[0] for n in self._model_files(folder) if n != os.path.basename(path)]
        # another model whose stem starts with ours plus a dot ("foo" vs "foo.v2") owns its own "foo.v2.*" files
        longer = [other for other in others if other.lower().startswith(stem.lower() + ".")]

        def ours(file_name: str, separators: Tuple[str, ...]) -> bool:
            lowered = file_name.lower()
            if not any(lowered.startswith(stem.lower() + sep) for sep in separators):
                return False
            if lowered.endswith(MODEL_EXTENSIONS) and lowered != os.path.basename(path).lower():
                return False
            return not any(lowered.startswith(other.lower() + ".") or lowered.startswith(other.lower() + "_")
                           for other in longer)

        files = [path]
        for file_name in sorted(os.listdir(folder)):
            full = os.path.join(folder, file_name)
            if full != path and os.path.isfile(full) and ours(file_name, (".",)):
                files.append(full)
        images = os.path.join(folder, "images")
        if os.path.isdir(images):
            for file_name in sorted(os.listdir(images)):
                full = os.path.join(images, file_name)
                if os.path.isfile(full) and ours(file_name, (".", "_")):
                    files.append(full)
        return {"kind": "flat", "folder": folder, "model_folder": "", "files": files}

    # -- planning -------------------------------------------------------------------------------------------------------
    def _catalog_rows(self, paths: List[str]) -> Dict[str, sqlite3.Row]:
        with self.connect() as connection:
            rows = {}
            for path in paths:
                row = connection.execute("select path, sha256, model_name, stem, rel_dir from loras where path = ?",
                                         (path,)).fetchone()
                if row is not None:
                    rows[path] = row
            return rows

    def _store(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # microseconds in the id: undo orders batches by id, and two batches made in the same second (a fast script, or
        # the test) otherwise sorted by their random suffix (2026-10-11)
        plan["id"] = datetime.now().strftime("%Y-%m-%d_%H%M%S%f_") + uuid.uuid4().hex[:4]
        plan["created"] = _now()
        plan["busy"] = self.busy_jobs()
        plan["counts"] = {"loras": sum(len(u["mapping"]) for u in plan["units"]),
                          "files": sum(u.get("file_count", len(u["ops"])) for u in plan["units"]),
                          "skipped": len(plan["skipped"]), "new_folders": len(plan["mkdirs"])}
        with self.lock:
            self.plans = {k: v for k, v in self.plans.items() if time.time() - v["_t"] < 3600}
            plan["_t"] = time.time()
            self.plans[plan["id"]] = plan
        return {k: v for k, v in plan.items() if not k.startswith("_")}

    def plan_move(self, paths: List[str], target: str) -> Dict[str, Any]:
        target = os.path.normpath(target)
        plan: Dict[str, Any] = {"kind": "move", "target": target, "units": [], "skipped": [], "mkdirs": [],
                                "cleanup": [], "warnings": []}
        problem = self.organising_folder(target)
        if problem:
            plan["error"] = problem
            return self._store(plan)
        target_root = self.root_of(target) or target
        if not os.path.isdir(target):
            plan["mkdirs"].append(target)
        rows = self._catalog_rows(paths)
        claimed: Dict[str, str] = {}          # destination -> label, so two LoRAs in one batch cannot collide
        seen_units = set()
        for path in paths:
            label = os.path.basename(path)
            row = rows.get(path)
            if row is not None:
                label = row["model_name"] or row["stem"] or label
            if not os.path.isfile(path):
                plan["skipped"].append({"path": path, "label": label, "reason": "the file is not there any more"})
                continue
            root = self.root_of(path)
            if root is None or _norm(root) != _norm(target_root):
                plan["skipped"].append({"path": path, "label": label,
                                        "reason": "moves stay inside one library root (this LoRA is in another)"})
                continue
            unit = self.unit_for(path)
            if _norm(unit["folder"]) in seen_units and unit["kind"] == "version_folder":
                continue
            seen_units.add(_norm(unit["folder"]))
            ops: List[List[str]] = []
            mkdirs: List[str] = []
            if unit["kind"] == "version_folder":
                parent = os.path.join(target, unit["model_folder"]) if unit["model_folder"] else target
                destination = os.path.join(parent, os.path.basename(unit["folder"]))
                if _norm(os.path.dirname(unit["folder"])) == _norm(parent):
                    plan["skipped"].append({"path": path, "label": label, "reason": "already in that folder"})
                    continue
                if _inside(target, unit["folder"]):
                    plan["skipped"].append({"path": path, "label": label, "reason": "cannot move a folder into itself"})
                    continue
                if unit["model_folder"] and not os.path.isdir(parent) and _norm(parent) not in claimed:
                    mkdirs.append(parent)
                ops.append([unit["folder"], destination])
                new_path = os.path.join(destination, os.path.basename(path))
                file_count = sum(len(files) for _f, _d, files in os.walk(unit["folder"]))
                old_model_folder = os.path.dirname(unit["folder"]) if unit["model_folder"] else ""
            else:
                if _norm(unit["folder"]) == _norm(target):
                    plan["skipped"].append({"path": path, "label": label, "reason": "already in that folder"})
                    continue
                for source in unit["files"]:
                    relative = os.path.relpath(source, unit["folder"])
                    ops.append([source, os.path.join(target, relative)])
                    sub = os.path.dirname(os.path.join(target, relative))
                    if _norm(sub) != _norm(target) and not os.path.isdir(sub) and sub not in mkdirs:
                        mkdirs.append(sub)
                new_path = os.path.join(target, os.path.basename(path))
                file_count = len(ops)
                old_model_folder = ""
            clash = next((dst for _src, dst in ops if os.path.exists(dst) or _norm(dst) in claimed), None)
            if clash:
                plan["skipped"].append({"path": path, "label": label,
                                        "reason": f"the target already has {os.path.basename(clash)}"})
                continue
            for _src, dst in ops:
                claimed[_norm(dst)] = label
            for folder in mkdirs:
                claimed[_norm(folder)] = label
            plan["units"].append({"label": label, "kind": unit["kind"], "ops": ops, "mkdirs": mkdirs,
                                  "file_count": file_count, "mapping": {path: new_path},
                                  "sha": {path: row["sha256"] if row is not None else None},
                                  "old_model_folder": old_model_folder})
        plan["mkdirs"] += [folder for unit in plan["units"] for folder in unit["mkdirs"]]
        self._cleanup_candidates(plan)
        return self._store(plan)

    @staticmethod
    def _cleanup_candidates(plan: Dict[str, Any]) -> None:
        """Downloader model folders that will be empty once their versions in this batch have left."""
        leaving: Dict[str, int] = {}
        for unit in plan["units"]:
            if unit.get("old_model_folder"):
                leaving[unit["old_model_folder"]] = leaving.get(unit["old_model_folder"], 0) + 1
        for folder, count in leaving.items():
            try:
                if len(os.listdir(folder)) == count:
                    plan["cleanup"].append(folder)
            except OSError:
                pass

    def plan_rename_folder(self, folder: str, new_folder: str) -> Dict[str, Any]:
        folder, new_folder = os.path.normpath(folder), os.path.normpath(new_folder)
        plan: Dict[str, Any] = {"kind": "rename_folder", "target": new_folder, "source": folder, "units": [],
                                "skipped": [], "mkdirs": [], "cleanup": [], "warnings": []}
        error = self.organising_folder(folder) or self.organising_folder(os.path.dirname(new_folder)) \
            or check_name(os.path.basename(new_folder))
        root = self.root_of(folder)
        if not error and not os.path.isdir(folder):
            error = "that folder is not there any more"
        if not error and (root is None or self.root_of(new_folder) is None
                          or _norm(self.root_of(new_folder)) != _norm(root)):
            error = "folders stay inside one library root"
        if not error and _norm(os.path.dirname(folder)) == _norm(root):
            error = ("that is a top-level architecture folder: every saved workflow names it, so it is not renamed "
                     "from here")
        if not error and _inside(new_folder, folder):
            error = "cannot move a folder into itself"
        if not error and os.path.exists(new_folder) and _norm(new_folder) != _norm(folder):
            error = f"{new_folder} already exists"
        if not error and _norm(new_folder) == _norm(folder) and new_folder == folder:
            error = "that is the same name"
        if error:
            plan["error"] = error
            return self._store(plan)
        if not os.path.isdir(os.path.dirname(new_folder)):
            plan["mkdirs"].append(os.path.dirname(new_folder))
        with self.connect() as connection:
            rows = [row for row in connection.execute("select path, sha256 from loras") if _inside(row["path"], folder)]
        mapping = {row["path"]: os.path.join(new_folder, os.path.relpath(row["path"], folder)) for row in rows}
        file_count = sum(len(files) for _f, _d, files in os.walk(folder))
        plan["units"].append({"label": os.path.basename(folder), "kind": "folder", "ops": [[folder, new_folder]],
                              "mkdirs": [], "file_count": file_count, "mapping": mapping,
                              "sha": {row["path"]: row["sha256"] for row in rows}, "old_model_folder": ""})
        return self._store(plan)

    def trash_root(self, root: str) -> str:
        return os.path.join(os.path.dirname(os.path.normpath(root)), "_lora_trash")

    def plan_delete(self, paths: List[str]) -> Dict[str, Any]:
        """Move LoRAs (each with everything that travels with it) to the trash folder. Same unit rules as a move."""
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        plan: Dict[str, Any] = {"kind": "delete", "target": "", "units": [], "skipped": [], "mkdirs": [],
                                "cleanup": [], "warnings": []}
        rows = self._catalog_rows(paths)
        seen_units = set()
        for path in paths:
            row = rows.get(path)
            label = (row["model_name"] or row["stem"]) if row is not None else os.path.basename(path)
            if not os.path.isfile(path):
                plan["skipped"].append({"path": path, "label": label, "reason": "the file is not there any more"})
                continue
            root = self.root_of(path)
            if root is None:
                plan["skipped"].append({"path": path, "label": label, "reason": "not inside a LoRA library"})
                continue
            trash = os.path.join(self.trash_root(root), stamp)
            plan["target"] = plan["target"] or trash
            unit = self.unit_for(path)
            if _norm(unit["folder"]) in seen_units and unit["kind"] == "version_folder":
                continue
            seen_units.add(_norm(unit["folder"]))
            sources = [unit["folder"]] if unit["kind"] == "version_folder" else unit["files"]
            ops = [[source, os.path.join(trash, os.path.relpath(source, root))] for source in sources]
            clash = next((dst for _src, dst in ops if os.path.exists(dst)), None)
            if clash:
                plan["skipped"].append({"path": path, "label": label, "reason": f"already in the trash: {clash}"})
                continue
            new_path = os.path.join(trash, os.path.relpath(path, root))
            file_count = (sum(len(files) for _f, _d, files in os.walk(unit["folder"]))
                          if unit["kind"] == "version_folder" else len(ops))
            plan["units"].append({"label": label, "kind": unit["kind"], "ops": ops, "mkdirs": [],
                                  "file_count": file_count, "mapping": {path: new_path},
                                  "sha": {path: row["sha256"] if row is not None else None},
                                  "old_model_folder": os.path.dirname(unit["folder"]) if unit["model_folder"] else ""})
        self._cleanup_candidates(plan)
        return self._store(plan)

    def forget_rows(self, mapping: Dict[str, str], sha: Dict[str, Optional[str]], batch: str) -> Dict[str, Any]:
        """Delete: take the trashed LoRAs out of every path-keyed table; return the rows so undo can put them back."""
        saved: Dict[str, Any] = {"rows": {}, "edits": []}
        connection = self.connect()
        try:
            tables = {row[0] for row in connection.execute("select name from sqlite_master where type = 'table'")}
            with connection:
                for old, new in mapping.items():
                    for table in PATH_TABLES:
                        if table not in tables:
                            continue
                        rows = [dict(row) for row in connection.execute(f"select * from {table} where path = ?", (old,))]
                        for row in rows:          # BLOBs (card embeddings) as hex so the manifest stays JSON
                            for key, value in row.items():
                                if isinstance(value, bytes):
                                    row[key] = {"hex": value.hex()}
                        if rows:
                            saved["rows"].setdefault(table, []).extend(rows)
                        connection.execute(f"delete from {table} where path = ?", (old,))
                    key = (sha.get(old) or "").lower()
                    if key:                       # the edit stays (keyed by sha); point it at a copy that remains
                        other = connection.execute("select path from loras where lower(sha256) = ? limit 1", (key,)).fetchone()
                        edit = connection.execute("select path from card_edits where key = ?", (key,)).fetchone()
                        if edit and edit[0] == old:
                            saved["edits"].append([key, old])
                            connection.execute("update card_edits set path = ? where key = ?",
                                               (other[0] if other else None, key))
                    connection.execute("insert into path_history (old_path, new_path, sha256, batch, moved_at) "
                                       "values (?, ?, ?, ?, ?)", (old, new, key or None, batch, _now()))
        finally:
            connection.close()
        return saved

    def restore_rows(self, saved: Dict[str, Any], batch: str, paths: Optional[set] = None) -> None:
        """Undo of a delete: put the saved rows back - only for `paths` (the LoRAs that actually came back)."""
        connection = self.connect()
        try:
            with connection:
                for table, rows in (saved.get("rows") or {}).items():
                    for row in rows:
                        if paths is not None and row.get("path") not in paths:
                            continue
                        values = [bytes.fromhex(v["hex"]) if isinstance(v, dict) and "hex" in v else v
                                  for v in row.values()]
                        connection.execute(f"insert or replace into {table} ({','.join(row)}) "
                                           f"values ({','.join('?' * len(row))})", values)
                for key, old in saved.get("edits") or []:
                    if paths is None or old in paths:
                        connection.execute("update card_edits set path = ? where key = ?", (old, key))
                for old in (paths if paths is not None else []):
                    connection.execute("update path_history set undone = 1 where batch = ? and old_path = ?", (batch, old))
                if paths is None:
                    connection.execute("update path_history set undone = 1 where batch = ?", (batch,))
        finally:
            connection.close()

    def plan_new_folder(self, parent: str, name: str) -> Dict[str, Any]:
        parent = os.path.normpath(parent)
        plan: Dict[str, Any] = {"kind": "new_folder", "target": os.path.join(parent, name.strip()), "units": [],
                                "skipped": [], "mkdirs": [], "cleanup": [], "warnings": []}
        error = self.organising_folder(parent) or check_name(name)
        if not error and os.path.exists(plan["target"]):
            error = "that folder already exists"
        if error:
            plan["error"] = error
        else:
            plan["mkdirs"].append(plan["target"])
        return self._store(plan)

    # -- doing it -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _makedirs(folder: str, created: List[str]) -> None:
        """os.makedirs that records EVERY level it creates (top first), so undo can take each one away again."""
        missing = []
        while folder and not os.path.isdir(folder):
            missing.append(folder)
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
        for level in reversed(missing):
            os.mkdir(level)
            created.append(level)

    @staticmethod
    def _rename(source: str, destination: str) -> None:
        if os.path.exists(destination) and _norm(source) != _norm(destination):     # a case-only rename is fine
            raise FileExistsError(f"{destination} appeared since the plan was made")
        os.rename(source, destination)

    def apply(self, plan_id: str) -> Dict[str, Any]:
        with self.lock:
            plan = self.plans.pop(plan_id, None)
        if plan is None:
            return {"ok": False, "error": "that plan has expired or was already applied - make it again"}
        if plan.get("error"):
            return {"ok": False, "error": plan["error"]}
        manifest: Dict[str, Any] = {"id": plan["id"], "kind": plan["kind"], "target": plan.get("target"),
                                    "source": plan.get("source"), "applied_at": _now(), "created_dirs": [],
                                    "units": [], "removed_dirs": [], "undone_at": None}
        created = manifest["created_dirs"]
        failed: List[Dict[str, str]] = []
        for folder in plan["mkdirs"]:
            if not os.path.isdir(folder):
                try:
                    self._makedirs(folder, created)
                except OSError as error:
                    failed.append({"label": folder, "reason": f"could not create the folder: {error}"})
        for unit in plan["units"]:
            done: List[List[str]] = []
            try:
                for source, destination in unit["ops"]:
                    self._makedirs(os.path.dirname(destination), created)
                    self._rename(source, destination)
                    done.append([source, destination])
            except OSError as error:
                for source, destination in reversed(done):          # put this LoRA's moved files back
                    try:
                        os.rename(destination, source)
                    except OSError:
                        pass
                reason = "Windows would not let go of a file - is ComfyUI using this LoRA?" \
                    if isinstance(error, PermissionError) else str(error)
                failed.append({"label": unit["label"], "reason": reason})
                continue
            manifest["units"].append({"label": unit["label"], "ops": done, "mapping": unit["mapping"],
                                      "sha": unit["sha"]})
        keep = {_norm(plan["target"])} if plan["kind"] in ("move", "new_folder") else set()
        for folder in reversed(list(created)):                    # made for a LoRA that then failed: take it away
            try:
                if _norm(folder) not in keep and os.path.isdir(folder) and not os.listdir(folder):
                    os.rmdir(folder)
                    created.remove(folder)
            except OSError:
                pass
        for folder in plan.get("cleanup", []):                      # empty downloader model folders only
            try:
                if os.path.isdir(folder) and not os.listdir(folder):
                    os.rmdir(folder)
                    manifest["removed_dirs"].append(folder)
            except OSError:
                pass
        mapping = {old: new for unit in manifest["units"] for old, new in unit["mapping"].items()}
        sha = {old: value for unit in manifest["units"] for old, value in unit["sha"].items()}
        catalog_error = None
        lm_fixed: List[str] = []
        try:
            if plan["kind"] == "delete":
                manifest["deleted_rows"] = self.forget_rows(mapping, sha, plan["id"])
            else:
                self.rewrite_catalog(mapping, sha, plan["id"])
        except Exception as error:                                   # the files moved; say so loudly, keep the manifest
            catalog_error = f"{type(error).__name__}: {error}"
        if plan["kind"] != "delete":                                 # trashed LoRAs keep their sidecars as they were
            lm_fixed = self.fix_lora_manager(mapping)
        self.forget_folders()
        manifest["lm_metadata"] = lm_fixed
        manifest["failed"] = failed
        manifest["catalog_error"] = catalog_error
        self._write_manifest(manifest)
        if mapping:
            self.ask_lora_manager_to_scan()
        return {"ok": catalog_error is None, "id": plan["id"], "kind": plan["kind"], "target": plan.get("target"),
                "moved": len(mapping),
                "units": len(manifest["units"]), "failed": failed + plan["skipped"], "created_dirs": created,
                "removed_dirs": manifest["removed_dirs"], "lm_metadata": len(lm_fixed),
                "error": f"files moved but the catalog update failed: {catalog_error}" if catalog_error else None,
                "mapping": mapping}

    def _manifest_path(self, batch_id: str) -> str:
        return os.path.join(self.settings.moves_dir, f"{batch_id}.json")

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        with open(self._manifest_path(manifest["id"]), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=1)

    def batches(self, limit: int = 30) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(os.listdir(self.settings.moves_dir), reverse=True):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.settings.moves_dir, name), "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, ValueError):
                continue
            moved = sum(len(unit["mapping"]) for unit in manifest.get("units", []))
            out.append({"id": manifest["id"], "kind": manifest["kind"], "applied_at": manifest.get("applied_at"),
                        "target": manifest.get("target"), "source": manifest.get("source"), "moved": moved,
                        "created_dirs": len(manifest.get("created_dirs", [])), "undone_at": manifest.get("undone_at")})
            if len(out) >= limit:
                break
        return out

    def undo(self, batch_id: str) -> Dict[str, Any]:
        path = self._manifest_path(batch_id)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            return {"ok": False, "error": "no such batch"}
        if manifest.get("undone_at"):
            return {"ok": False, "error": f"already undone at {manifest['undone_at']}"}
        later = [b for b in self.batches(200) if b["id"] > batch_id and not b["undone_at"] and (b["moved"] or b["created_dirs"])]
        if later:
            return {"ok": False, "error": f"undo the later batch {later[-1]['id']} first (batches undo newest first)"}
        for folder in manifest.get("removed_dirs", []):
            os.makedirs(folder, exist_ok=True)
        failed, mapping, sha = [], {}, {}
        for unit in reversed(manifest.get("units", [])):
            done = []
            try:
                for source, destination in reversed(unit["ops"]):
                    os.makedirs(os.path.dirname(source), exist_ok=True)
                    self._rename(destination, source)
                    done.append([source, destination])
            except OSError as error:
                for source, destination in reversed(done):
                    try:
                        os.rename(source, destination)
                    except OSError:
                        pass
                failed.append({"label": unit["label"], "reason": str(error)})
                continue
            for old, new in unit["mapping"].items():
                mapping[new] = old
                sha[new] = unit["sha"].get(old)
        removed = []
        for folder in reversed(manifest.get("created_dirs", [])):
            try:
                if os.path.isdir(folder) and not os.listdir(folder):
                    os.rmdir(folder)
                    removed.append(folder)
            except OSError:
                pass
        catalog_error = None
        try:
            if manifest.get("kind") == "delete":
                if mapping:
                    self.restore_rows(manifest.get("deleted_rows") or {}, manifest["id"], set(mapping.values()))
            else:
                self.rewrite_catalog(mapping, sha, manifest["id"], undo=True)
        except Exception as error:
            catalog_error = f"{type(error).__name__}: {error}"
        if manifest.get("kind") != "delete":
            self.fix_lora_manager(mapping)
        self.forget_folders()
        if not failed:
            manifest["undone_at"] = _now()
        manifest.setdefault("undo_log", []).append({"at": _now(), "restored": len(mapping), "failed": failed,
                                                    "catalog_error": catalog_error})
        self._write_manifest(manifest)
        if mapping:
            self.ask_lora_manager_to_scan()
        return {"ok": not failed and catalog_error is None, "restored": len(mapping), "failed": failed,
                "removed_dirs": removed, "error": catalog_error}

    # -- keeping the catalog and LoRA Manager in step -------------------------------------------------------------------
    def rewrite_catalog(self, mapping: Dict[str, str], sha: Dict[str, Optional[str]], batch: str,
                        undo: bool = False) -> None:
        if not mapping:
            return
        from . import facets                                      # late: facets pulls in the vocab
        connection = self.connect()
        try:
            tables = {row[0] for row in connection.execute("select name from sqlite_master where type = 'table'")}
            with connection:
                for old, new in mapping.items():
                    for table in PATH_TABLES:
                        if table in tables:
                            connection.execute(f"update or replace {table} set path = ? where path = ?", (new, old))
                    root = self.root_of(new) or ""
                    relative = os.path.relpath(os.path.dirname(new), root) if root else ""
                    parts = [] if relative in ("", ".") else relative.split(os.sep)
                    arch = parts[0] if parts else ""
                    rel_dir = "/".join(parts)
                    connection.execute("update loras set root = ?, arch_folder = ?, rel_dir = ? where path = ?",
                                       (root, arch, rel_dir, new))
                    if "card_facets" in tables:
                        connection.execute("update card_facets set arch_folder = ?, rel_dir = ? where path = ?",
                                           (arch, rel_dir, new))
                    key = (sha.get(old) or "").lower()
                    if key:
                        connection.execute("update card_edits set path = ? where key = ?", (new, key))
                    else:
                        connection.execute("update card_edits set path = ?, key = ? where key = ?", (new, new, old))
                    if undo:
                        connection.execute("update path_history set undone = 1 where batch = ? and new_path = ?",
                                           (batch, old))
                    else:
                        connection.execute("insert into path_history (old_path, new_path, sha256, batch, moved_at) "
                                           "values (?, ?, ?, ?, ?)", (old, new, key or None, batch, _now()))
            for new in mapping.values():
                try:
                    facets.refresh_one(connection, new)
                except Exception as error:
                    print(f"[organise] card_facets refresh failed for {new}: {error}")
        finally:
            connection.close()

    @staticmethod
    def fix_lora_manager(mapping: Dict[str, str]) -> List[str]:
        """Point each moved LoRA's <stem>.metadata.json at its new place, the way LoRA Manager's own move does."""
        fixed = []
        for new in mapping.values():
            metadata_path = os.path.splitext(new)[0] + ".metadata.json"
            if not os.path.isfile(metadata_path):
                continue
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                data["file_path"] = new.replace(os.sep, "/")
                data["file_name"] = os.path.splitext(os.path.basename(new))[0]
                if data.get("preview_url"):
                    preview = os.path.join(os.path.dirname(new), os.path.basename(data["preview_url"].replace("/", os.sep)))
                    data["preview_url"] = preview.replace(os.sep, "/")
                temporary = metadata_path + ".tmp"
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=2)
                os.replace(temporary, metadata_path)
                fixed.append(metadata_path)
            except (OSError, ValueError) as error:
                print(f"[organise] LoRA Manager metadata not updated for {new}: {error}")
        return fixed

    def ask_lora_manager_to_scan(self) -> None:
        """LoRA Manager has no file watcher; when ComfyUI is up, its scan reconciles its list with the disk."""
        def run():
            url = self.settings.comfy_url.rstrip("/") + "/api/lm/loras/scan"
            self.scan_state = {"state": "running", "since": _now()}
            try:
                with urllib.request.urlopen(url, timeout=900) as response:
                    self.scan_state = {"state": "done", "at": _now(), "code": response.status}
            except Exception as error:
                self.scan_state = {"state": "not run", "at": _now(),
                                   "why": f"ComfyUI not reachable ({type(error).__name__}) - LoRA Manager will catch "
                                          f"up when it next starts, or press its refresh"}
        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def busy_jobs() -> List[str]:
        """Other LoRA Catalog runs (cards, sliders, harvest, index...) that write by path: moving under them is unsafe.
        From the lock files every run.py command leaves while it runs (jobs.py), so it works on any OS."""
        from .jobs import running
        return [info.get("command", "?") for info in running() if info.get("command") != "serve"]

    # -- saved files that still name a moved LoRA -----------------------------------------------------------------------
    def references(self, batch_id: str, limit: int = 300) -> Dict[str, Any]:
        try:
            with open(self._manifest_path(batch_id), "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            return {"ok": False, "error": "no such batch"}
        names = {}
        for unit in manifest.get("units", []):
            for old in unit["mapping"]:
                names[os.path.basename(old).lower()] = old
        if not names:
            return {"ok": True, "files": [], "searched": 0}
        pattern = re.compile("|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)), re.I)
        found, searched = [], 0
        for folder in self.settings.reference_dirs:
            if not os.path.isdir(folder):
                continue
            for directory, _dirs, files in os.walk(folder):
                for file_name in files:
                    if not file_name.lower().endswith(REFERENCE_EXTENSIONS):
                        continue
                    full = os.path.join(directory, file_name)
                    searched += 1
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as handle:
                            text = handle.read()
                    except OSError:
                        continue
                    hits = sorted({match.group(0).lower() for match in pattern.finditer(text)})
                    if hits:
                        found.append({"file": full, "loras": hits})
                        if len(found) >= limit:
                            return {"ok": True, "files": found, "searched": searched, "truncated": True}
        return {"ok": True, "files": found, "searched": searched}
