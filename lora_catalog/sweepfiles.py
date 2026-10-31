"""
Where a sweep folder keeps its manifest (2026-10-15).

Eric_Krea2 sweeps since 2026-10-15 name their files after the sweep (sweep_manifest_<name>.json, ...); older
sweeps use sweep_manifest.json. Both are found here. (Mirrors Eric_Krea2/_sweep_files.py - no import between
the packs.)

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

OLD = "sweep_manifest.json"


def manifest_path(sweep_dir: str) -> Optional[str]:
    """The sweep's manifest file, or None."""
    old = os.path.join(sweep_dir, OLD)
    if os.path.isfile(old):
        return old
    hits = sorted(glob.glob(os.path.join(glob.escape(sweep_dir), "sweep_manifest_*.json")))
    return hits[0] if hits else None
