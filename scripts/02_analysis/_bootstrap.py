"""Put analysis layer folders on ``sys.path``.

Call ``ensure_analysis_path()`` before local imports in any entry script.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_ROOT = Path(__file__).resolve().parent
_LAYER_DIRS = ("core", "features", "summaries", "cross")


def ensure_analysis_path() -> Path:
    """Insert layer folders then analysis root onto ``sys.path``."""
    root = _ANALYSIS_ROOT
    dirs = [root / name for name in _LAYER_DIRS if (root / name).is_dir()]
    dirs.append(root)
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


ANALYSIS_ROOT = _ANALYSIS_ROOT
