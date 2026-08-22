"""Put ``02_analysis`` shared + theme folders on ``sys.path``.

Call ``ensure_analysis_path()`` before local imports in any entry script under
``02_0N_*``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_ROOT = Path(__file__).resolve().parent


def ensure_analysis_path() -> Path:
    """Insert analysis root and each ``02_*`` theme folder onto ``sys.path``."""
    root = _ANALYSIS_ROOT
    # Theme folders first so same-named modules resolve to the intended package,
    # then root for shared helpers (_paths, gait_onset, gait_smooth_plot).
    dirs = [p for p in sorted(root.glob("02_*")) if p.is_dir()]
    dirs.append(root)
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


ANALYSIS_ROOT = _ANALYSIS_ROOT
