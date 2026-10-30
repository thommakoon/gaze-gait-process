#!/usr/bin/env python3
"""Standing vs walk·no-IC vs walk·has-IC — transit MT (leave→first hit).

Person-first matched min-n on the two walking arms; standing is its own
person mean (not forced into the walking subsample).

Classes (leave→hit window):
  standing   — PracticeRing / PracticeRectangle
  no_ic      — walking, n_ic_move == 0
  has_ic     — walking, n_ic_move >= 1

Sources:
  transit_s  — finalize/shared/episodes_with_transit.csv
  n_ic_move  — features/phase_ic_counts/episodes_cohort.csv (merged on appear/confirm)

Usage (from scripts/02_analysis/):
    uv run python features/stand_vs_noIC_vs_IC_transit.py
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import numpy as np
import pandas as pd
from scipy import stats

from _paths import INTERACTIONS, COHORT_OUT_ROOT, analysis_out
from across_people import part_name
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)
TRANSIT = COHORT_OUT_ROOT / "finalize" / "shared" / "episodes_with_transit.csv"
IC_EP = analysis_out("features/phase_ic_counts.py") / "episodes_cohort.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
RNG = np.random.default_rng(0)

ARMS = ("standing", "no_ic", "has_ic")
ARM_LABEL = {
    "standing": "Standing",
    "no_ic": "Walk · no IC",
    "has_ic": "Walk · has IC",
}


def _attach_n_ic_move(tr: pd.DataFrame, ic: pd.DataFrame) -> pd.DataFrame:
    """Merge n_ic_move onto walking transit rows via appear/confirm (ms rounded)."""
    walk = tr[tr["speed_group"].astype(str) == "walking"].copy()
    stand = tr[tr["speed_group"].astype(str) == "standing"].copy()
    stand["n_ic_move"] = np.nan
    stand["arm"] = "standing"

    ic = ic.copy()
    for df in (walk, ic):
        df["appear_r"] = pd.to_numeric(df["appear_unix_ms"], errors="coerce").round(0)
        df["confirm_r"] = pd.to_numeric(df["confirm_unix_ms"], errors="coerce").round(0)

    keys = ["participant", "speed", "interaction", "appear_r", "confirm_r"]
    ic_slim = (
        ic[keys + ["n_ic_move"]]
        .drop_duplicates(subset=keys, keep="first")
    )
    m = walk.merge(ic_slim, on=keys, how="left")
    n_ic = pd.to_numeric(m["n_ic_move"], errors="coerce")
    m = m[n_ic.notna()].copy()
    n_ic = n_ic.loc[m.index]
    m["arm"] = np.where(n_ic <= 0, "no_ic", "has_ic")
    return pd.concat([stand, m], ignore_index=True)


def _match_person_cell(g: pd.DataFrame) -> tuple[pd.DataFrame | None, dict | None]:
    """Min-n match walking arms; standing = all trials. Returns (trials, person_rec)."""
    stand = g[g["arm"] == "standing"]
    no = g[g["arm"] == "no_ic"]
    has = g[g["arm"] == "has_ic"]
    if len(stand) < 1 or len(no) < 1 or len(has) < 1:
        return None, None
    n = int(min(len(no), len(has)))
    if n < 1:
        return None, None
    seed = int(RNG.integers(0, 1_000_000))
    no_i = no.sample(n=n, random_state=seed)
    has_i = has.sample(n=n, random_state=seed + 1)
    matched = pd.concat([stand, no_i, has_i], ignore_index=True)
    matched = matched.assign(n_matched_walk=n)

    def _mean_ms(df: pd.DataFrame) -> float:
        v = pd.to_numeric(df["transit_ms"], errors="coerce")
        v = v[np.isfinite(v) & (v > 0)]
        return float(v.mean()) if len(v) else float("nan")

    rec = {
        "n_standing": int(len(stand)),
        "n_matched_walk": n,
        "n_no_ic_pool": int(len(no)),
        "n_has_ic_pool": int(len(has)),
        "transit_ms_standing": _mean_ms(stand),
        "transit_ms_no_ic": _mean_ms(no_i),
        "transit_ms_has_ic": _mean_ms(has_i),
    }
    return matched, rec


def main() -> None:
    if not TRANSIT.is_file():
        raise SystemExit(f"missing {TRANSIT} — run core/compute_transit.py")
    if not IC_EP.is_file():
        raise SystemExit(f"missing {IC_EP} — run features/phase_ic_counts.py")

    tr = pd.read_csv(TRANSIT)
    tr["participant"] = tr["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    tr = tr[tr["participant"].isin(COHORT)].copy()
    tr["transit_ms"] = pd.to_numeric(tr["transit_s"], errors="coerce") * 1000.0
    tr = tr[np.isfinite(tr["transit_ms"]) & (tr["transit_ms"] > 0)].copy()

    ic = pd.read_csv(IC_EP)
    ic["participant"] = ic["participant"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = _attach_n_ic_move(tr, ic)
    print(
        f"trials after IC attach: standing={int((ep.arm=='standing').sum())}  "
        f"no_ic={int((ep.arm=='no_ic').sum())}  has_ic={int((ep.arm=='has_ic').sum())}"
    )

    person_rows: list[dict] = []
    trial_rows: list[pd.DataFrame] = []
    for (layout, inter), g in ep.groupby(["layout", "interaction"], dropna=False):
        for pid, pg in g.groupby("participant"):
            matched, rec = _match_person_cell(pg)
            if matched is None or rec is None:
                continue
            matched = matched.assign(layout=layout, interaction=inter, participant=pid)
            trial_rows.append(matched)
            person_rows.append(
                {
                    "participant": pid,
                    "layout": layout,
                    "interaction": inter,
                    **rec,
                    "delta_has_minus_no": rec["transit_ms_has_ic"] - rec["transit_ms_no_ic"],
                    "delta_no_minus_stand": rec["transit_ms_no_ic"] - rec["transit_ms_standing"],
                    "delta_has_minus_stand": rec["transit_ms_has_ic"] - rec["transit_ms_standing"],
                }
            )

    if not person_rows:
        raise SystemExit("no complete-case people (need standing + both walking IC classes)")

    person = pd.DataFrame(person_rows)
    trials = pd.concat(trial_rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    person.to_csv(OUT / "matched_person.csv", index=False)
    trials.to_csv(OUT / "matched_trials.csv", index=False)

    sum_rows = []
    for (layout, inter), g in person.groupby(["layout", "interaction"], dropna=False):
        n = int(len(g))
        row: dict = {
            "layout": layout,
            "interaction": inter,
            "n_people": n,
            "mean_n_matched_walk": float(g["n_matched_walk"].mean()),
        }
        for arm in ARMS:
            col = f"transit_ms_{arm}"
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row[f"mean_{arm}"] = float(np.mean(v)) if v.size else float("nan")
            row[f"se_{arm}"] = (
                float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
            )
        for name, col in (
            ("has_minus_no", "delta_has_minus_no"),
            ("no_minus_stand", "delta_no_minus_stand"),
            ("has_minus_stand", "delta_has_minus_stand"),
        ):
            d = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if d.size >= 5:
                try:
                    _s, p = stats.wilcoxon(d, alternative="two-sided")
                except ValueError:
                    p = float("nan")
                row[f"median_delta_{name}"] = float(np.median(d))
                row[f"wilcoxon_p_{name}"] = float(p) if np.isfinite(p) else float("nan")
            else:
                row[f"median_delta_{name}"] = float("nan")
                row[f"wilcoxon_p_{name}"] = float("nan")
        sum_rows.append(row)

    summary = pd.DataFrame(sum_rows)
    summary.to_csv(OUT / "matched_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
