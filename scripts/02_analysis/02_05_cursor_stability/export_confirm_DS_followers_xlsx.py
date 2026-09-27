#!/usr/bin/env python3
"""List person × bout cells that do NOT follow confirm-in-double-support.

Rule: enrichment E for double support <= 1
  (confirms in DS no more often than time-in-DS predicts).

Reads existing:
  confirm_count_gait_support/confirm_support_enrichment_person.csv

Writes (new file only):
  confirm_count_gait_support/confirm_DS_followers.xlsx
    - sheet "all_cells": every person × layout × modality
    - sheet "does_not_follow": E_double <= 1 only
    - sheet "person_summary": fraction of cells that follow

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/export_confirm_DS_followers_xlsx.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import pandas as pd

from _paths import analysis_out

OUT = analysis_out("02_05_cursor_stability/confirm_count_gait_support.py")
SRC = OUT / "confirm_support_enrichment_person.csv"
XLSX = OUT / "confirm_DS_followers.xlsx"


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")

    enr = pd.read_csv(SRC)
    enr = enr[(enr["event"] == "confirm") & (enr["grain"] == "support")].copy()

    d = enr[enr["state"] == "double"][
        [
            "participant",
            "layout",
            "interaction",
            "n_trials",
            "n_s",
            "N",
            "event_share",
            "time_share",
            "enrichment",
        ]
    ].rename(
        columns={
            "n_s": "n_confirm_DS",
            "N": "n_confirm_labeled",
            "event_share": "share_confirm_DS",
            "time_share": "share_time_DS",
            "enrichment": "E_double",
        }
    )
    s = enr[enr["state"] == "single"][
        ["participant", "layout", "interaction", "enrichment"]
    ].rename(columns={"enrichment": "E_single"})

    cells = d.merge(s, on=["participant", "layout", "interaction"], how="left")
    cells["bout"] = cells["layout"].astype(str) + "_" + cells["interaction"].astype(str)
    cells["follows_DS_trend"] = cells["E_double"] > 1.0
    cells["note"] = cells["follows_DS_trend"].map(
        {True: "follows (E_double>1)", False: "does NOT follow (E_double<=1)"}
    )
    cells = cells.sort_values(["follows_DS_trend", "participant", "layout", "interaction"])

    does_not = cells[~cells["follows_DS_trend"]].copy()

    person = (
        cells.groupby("participant", as_index=False)
        .agg(
            n_cells=("E_double", "count"),
            n_follow=("follows_DS_trend", "sum"),
            mean_E_double=("E_double", "mean"),
            mean_E_single=("E_single", "mean"),
        )
    )
    person["n_not_follow"] = person["n_cells"] - person["n_follow"]
    person["frac_follow"] = person["n_follow"] / person["n_cells"]
    person = person.sort_values(["frac_follow", "mean_E_double"], ascending=[True, True])

    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        cells.to_excel(writer, sheet_name="all_cells", index=False)
        does_not.to_excel(writer, sheet_name="does_not_follow", index=False)
        person.to_excel(writer, sheet_name="person_summary", index=False)

    print(f"wrote {XLSX}")
    print(f"does_not_follow rows: {len(does_not)} / {len(cells)} cells")


if __name__ == "__main__":
    main()
