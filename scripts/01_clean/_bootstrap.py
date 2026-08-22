"""Put ``01_clean`` shared + step folders on ``sys.path``.

Call ``ensure_clean_path()`` before local imports in any entry script under
``01_0N_*``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CLEAN_ROOT = Path(__file__).resolve().parent


def ensure_clean_path() -> Path:
    """Insert clean root and each ``01_*`` step folder onto ``sys.path``."""
    root = _CLEAN_ROOT
    dirs = [p for p in sorted(root.glob("01_*")) if p.is_dir()]
    dirs.append(root)
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


CLEAN_ROOT = _CLEAN_ROOT
