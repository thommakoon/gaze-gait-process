"""Shared pastel styling for finalize paper figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

# Locomotion (grouped bars)
LOCO = {
    "standing": {"label": "Standing", "color": "#A7C4E0"},  # soft blue
    "walking": {"label": "Walking", "color": "#F2B8A8"},  # soft coral
}

# Cursor modality (lines / gait bars)
MODALITY = {
    "HeadPinch": {"label": "Head", "color": "#6FA8DC", "marker": "o"},
    "HandPinch": {"label": "Hand", "color": "#F4A261", "marker": "s"},
    "EyePinch": {"label": "Eye", "color": "#82C49A", "marker": "^"},
}

F2_COLOR = "#B39CD0"  # soft purple
RF_IC_COLOR = "#9E9E9E"
EDGE = "#5C6B73"
GRID_ALPHA = 0.28

LAYOUTS = (("ring", "Ring"), ("rect", "Rectangle"))
INTERACTIONS = ("HeadPinch", "HandPinch", "EyePinch")
X_GROUPS = ("standing", "walking")


def apply_base_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.edgecolor": EDGE,
            "axes.labelcolor": "#2F3A40",
            "xtick.color": "#2F3A40",
            "ytick.color": "#2F3A40",
            "grid.color": "#C5CED4",
            "grid.alpha": GRID_ALPHA,
        }
    )
