"""
Where things are. Every path and endpoint the tool uses is here; settings.json (next to run.py, never in git) overrides
any of them with the same keys. Copy settings.example.json to settings.json to start, or let the first-run setup
page write it.

Three folders:
    TOOL_DIR      the code (this repo; inside ComfyUI it is custom_nodes/Eric_LoRA_Catalog)
    SHIPPED_DIR   TOOL_DIR/data: the pick lists that ship with the code (vocab.yaml, axes.yaml, axes_ladders.yaml)
    DATA_DIR      everything the tool writes: catalog, Civitai cache, review pages, move manifests, logs, locks, and
                  your own copies of the pick lists (made on first use, so your edits never touch the repo).
                  settings.json "data_dir", else the LORA_CATALOG_DATA environment variable, else
                  ComfyUI/user/lora_catalog inside ComfyUI, else TOOL_DIR/user_data.

Blank paths mean "not used" (LoRA Manager's database, training folders, sweep folders...) or "detect it" when the tool
runs inside ComfyUI (custom_nodes/<this folder>): sweeps, Krea2 sweep presets, LoRA Manager's database and the saved
workflows folder are then found next to it when they exist.

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(PACKAGE_DIR)
SHIPPED_DIR = os.path.join(TOOL_DIR, "data")
SETTINGS_FILE = os.environ.get("LORA_CATALOG_SETTINGS") or os.path.join(TOOL_DIR, "settings.json")
EXAMPLE_SETTINGS_FILE = os.path.join(TOOL_DIR, "settings.example.json")


def _comfy_dir() -> str:
    """ComfyUI's folder when this tool sits in its custom_nodes, else ''."""
    parent = os.path.dirname(TOOL_DIR)
    if os.path.basename(parent).lower() == "custom_nodes":
        return os.path.dirname(parent)
    return ""


COMFY_DIR = _comfy_dir()


def _raw_settings() -> Dict[str, Any]:
    if not os.path.isfile(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as error:
        print(f"[LoRA Catalog] settings.json ignored: {error}")
        return {}


def _data_dir() -> str:
    chosen = str(_raw_settings().get("data_dir") or os.environ.get("LORA_CATALOG_DATA") or "").strip()
    if not chosen:
        chosen = os.path.join(COMFY_DIR, "user", "lora_catalog") if COMFY_DIR else os.path.join(TOOL_DIR, "user_data")
    return os.path.abspath(os.path.expanduser(chosen))


# Read once at import: module-level paths (cache, reviews...) are built from it.
DATA_DIR = _data_dir()


def data_path(*parts: str) -> str:
    """A path under DATA_DIR (the folder is created)."""
    path = os.path.join(DATA_DIR, *parts)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


def user_file(name: str) -> str:
    """Your copy of a shipped pick list (vocab.yaml, axes.yaml, axes_ladders.yaml): DATA_DIR/name, copied from
    SHIPPED_DIR the first time it is needed. Edits made by the tool (new vocab terms, accepted ladders) go here."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        shipped = os.path.join(SHIPPED_DIR, name)
        if os.path.exists(shipped):
            os.makedirs(DATA_DIR, exist_ok=True)
            shutil.copy2(shipped, path)
    return path


def _detect(*parts: str) -> str:
    """A path under ComfyUI when it exists there, else ''."""
    if not COMFY_DIR:
        return ""
    path = os.path.join(COMFY_DIR, *parts)
    return path if os.path.exists(path) else ""


@dataclass
class Settings:
    # LoRA libraries to index (folders holding .safetensors files, sorted into any sub-folders you like).
    library_roots: List[str] = field(default_factory=list)
    # The catalog database. Blank: DATA_DIR/lora_catalog.sqlite.
    catalog_path: str = ""
    data_dir: str = ""                     # informational here; DATA_DIR above is what the tool uses
    # LoRA Manager's offline Civitai database (optional; detected inside ComfyUI; skipped when missing).
    lm_civitai_db: str = ""
    # ai-toolkit output folders: each job folder has config.yaml and its checkpoints (optional).
    training_roots: List[str] = field(default_factory=list)
    # Folder names under a library root that are not LoRAs or should be left alone.
    skip_dirs: List[str] = field(default_factory=lambda: ["example_images", "recipes", "Workflows", "New folder",
                                                          "__pycache__", ".git"])
    max_examples_per_lora: int = 12        # gallery prompts kept per file
    max_caption_samples: int = 15          # training captions kept per self-trained LoRA
    max_description_chars: int = 4000
    # Civitai API key, in this order: the CIVITAI_API_KEY environment variable, civitai_api_key here, then the
    # api_key in civitai_settings_file (another tool's settings file, read at run time, never copied). Never printed.
    civitai_api_key: str = ""
    civitai_settings_file: str = ""
    # Civitai site for API calls. civitai.red serves the same API with every rating included.
    civitai_host: str = "https://civitai.com"
    # Stack harvest (run.py harvest): top community images per LoRA, and the highest rating fetched
    # (None | Soft | Mature | X). Metadata only - no images are downloaded.
    harvest_images: int = 50
    harvest_nsfw: str = "X"
    # The LLM: any OpenAI-compatible server. LM Studio http://localhost:1234/v1, Ollama http://localhost:11434/v1,
    # llama.cpp server http://localhost:8080/v1. llm_backend: auto | lmstudio | ollama | other (used to unload models).
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = ""
    llm_backend: str = "auto"
    llm_endpoint: str = ""                 # old setting (full .../chat/completions URL); used when given
    card_model: str = "qwen/qwen3-vl-8b"   # vision model that writes the cards
    genre_model: str = ""                  # text model for run.py genre-mood; blank: card_model
    slider_model: str = ""                 # text model for run.py sliders / ladders; blank: genre_model
    card_images: int = 2                   # preview images shown to the vision model per LoRA
    # Artist names (a JSON list or {"names": [...]}), used to tie a LoRA to the artist it emulates by its name.
    # Optional: without it cards are written without name matches.
    artist_names_file: str = ""
    # File organising in the card editor: ComfyUI's address (LoRA Manager is told to rescan after a move; skipped when
    # not running), folders searched for saved files that still name a moved LoRA (report only), undo manifests.
    comfy_url: str = "http://127.0.0.1:8188"
    reference_dirs: List[str] = field(default_factory=list)
    moves_dir: str = ""                    # blank: DATA_DIR/moves
    # Eric_Krea2 sweeps (optional): sweep output folder and the SweepPlan node's preset file. Detected inside ComfyUI.
    sweeps_dir: str = ""
    krea2_sweep_presets: str = ""
    # LoRA Suggester options (see suggest/settings.py).
    suggest: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> "Settings":
        """Fill the blanks that have a sensible default."""
        self.data_dir = DATA_DIR
        if not self.catalog_path:
            self.catalog_path = os.path.join(DATA_DIR, "lora_catalog.sqlite")
        if not self.moves_dir:
            self.moves_dir = os.path.join(DATA_DIR, "moves")
        if not self.lm_civitai_db:
            self.lm_civitai_db = _detect("custom_nodes", "comfyui-lora-manager", "civitai", "civitai.sqlite")
        if not self.sweeps_dir:
            self.sweeps_dir = _detect("output", "sweeps")
        if not self.krea2_sweep_presets:
            self.krea2_sweep_presets = _detect("custom_nodes", "Eric_Krea2", "sweep_presets.json")
        stacks = os.path.join(DATA_DIR, "stacks")
        workflows = _detect("user", "default", "workflows")
        for folder in (workflows, stacks):
            if folder and folder not in self.reference_dirs:
                self.reference_dirs.append(folder)
        self.genre_model = self.genre_model or self.card_model
        self.slider_model = self.slider_model or self.genre_model
        self.library_roots = [os.path.expanduser(r) for r in self.library_roots if str(r).strip()]
        return self

    def chat_url(self) -> str:
        if self.llm_endpoint:
            return self.llm_endpoint
        return self.llm_base_url.rstrip("/") + "/chat/completions"

    def civitai_key(self) -> str:
        """The Civitai API key, or ''. Never print or log it."""
        key = os.environ.get("CIVITAI_API_KEY", "").strip() or str(self.civitai_api_key or "").strip()
        if key or not self.civitai_settings_file:
            return key
        try:
            with open(self.civitai_settings_file, "r", encoding="utf-8") as handle:
                return str(json.load(handle).get("api_key") or "").strip()
        except (OSError, ValueError):
            return ""

    def problems(self) -> List[str]:
        """What stops the tool from working, in plain words (empty = ready). The setup page and the panel show these."""
        found = []
        if not self.library_roots:
            found.append("No LoRA folders set (library_roots).")
        for root in self.library_roots:
            if not os.path.isdir(root):
                found.append(f"LoRA folder not found: {root}")
        return found


def load_settings() -> Settings:
    settings = Settings()
    for key, value in _raw_settings().items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings.finish()


def save_settings(values: Dict[str, Any]) -> str:
    """Merge values into settings.json (keys the Settings class knows only). Returns the file path."""
    data = _raw_settings()
    known = set(Settings.__dataclass_fields__)
    for key, value in values.items():
        if key in known:
            data[key] = value
    folder = os.path.dirname(SETTINGS_FILE)
    os.makedirs(folder, exist_ok=True)
    temp = SETTINGS_FILE + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temp, SETTINGS_FILE)
    return SETTINGS_FILE
