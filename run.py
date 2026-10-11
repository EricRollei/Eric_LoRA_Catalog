"""
Launcher for the LoRA Catalog commands (python run.py --help). ComfyUI's embedded Python ignores the current folder
(its ._pth file), so `-m lora_catalog` cannot find the package; this puts the tool folder on the path first.

    python run.py check                  settings, folders and the LLM server
    python run.py serve                  the card editor in your browser
    python run.py refresh                every step after adding LoRAs

(c) Eric Hiss. Dual license, see LICENSE.
"""
import os
import sys

if sys.version_info < (3, 12):
    sys.exit(f"The LoRA Catalog needs Python 3.12 or newer; this is {sys.version.split()[0]} ({sys.executable}).\n"
             "ComfyUI's portable build ships 3.12 in python_embeded.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lora_catalog.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
