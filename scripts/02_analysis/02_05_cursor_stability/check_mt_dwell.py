#!/usr/bin/env python3
"""Per-trial split of logged MT into aiming vs dwell-before-confirm.

Quest ``movement_time_s`` is appear→confirm (includes hover). This script checks
that we can also get:

  movement_only_s  = appear → first hit   (cursor first on target)
  dwell_s          = first hit → confirm  (hover until pinch)

and that ``movement_only_s + dwell_s ≈ movement_time_s``.

No gait needed — Quest JSON only.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/check_mt_dwell.py --participants 80 81
    uv run python 02_05_cursor_stability/check_mt_dwell.py --participant 80 --bout Ring
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import (
    DATA_ROOT,
    INTERACTIONS,
    STAGE_DIRS,
    add_bout_args,
    bout_dir,
    bout_labels,
    is_practice_bout,
    scan_bout_names,
)
from fitts_gait_onset import load_fitts_selections, pick_quest_json
from interaction_gait_stride import load_first_hits
from mark_bad_ic_periods import discover_bouts

RECON_TOL_S = 0.08  # first-hit is a Quest frame (~10–20 ms); allow a few frames


def discover_quest_bouts(args: argparse.Namespace) -> list[Path]:
    """Like discover_bouts, but keep a bout if the matching Quest JSON exists."""
    try:
        found = discover_bouts(args)
    except FileNotFoundError:
        found = []
    extra: list[Path] = []
    if args.participant:
        speeds = scan_bout_names(args.participant, args.speed)
        interactions = [args.interaction] if args.interaction else list(INTERACTIONS)
        for speed in speeds:
            for interaction in interactions:
                b = bout_dir(args.participant, speed, interaction)
                if b in found:
                    continue
                try:
                    pick_quest_json(b)
                except (FileNotFoundError, FileExistsError):
                    continue
                extra.append(b)
    return found + extra


def _bout_labels_extra(bout: Path) -> dict:
    speed = bout.parent.name
    return {
        "participant": bout_labels(bout)[0],
        "speed": speed,
        "interaction": bout.name,
        "layout": "rect" if "Rectangle" in speed else "ring",
        "speed_group": "standing" if is_practice_bout(speed) else "walking",
    }


def split_episode_mt(bout: Path) -> tuple[pd.DataFrame, dict]:
    subject, run = bout_labels(bout)
    labels = _bout_labels_extra(bout)
    qpath = pick_quest_json(bout)
    sel_all = load_fitts_selections(qpath, success_only=False)
    n_iso = int(len(sel_all))
    n_success = int(sel_all["success"].sum()) if n_iso else 0
    hit_rate = n_success / n_iso if n_iso else float("nan")
    sel = sel_all[sel_all["success"] == True].reset_index(drop=True)  # noqa: E712
    n_sel = int(len(sel))
    if sel.empty:
        return pd.DataFrame(), {
            "subject": subject,
            "run": run,
            **labels,
            "quest_file": qpath.name,
            "n_iso_selections": n_iso,
            "n_success": n_success,
            "hit_rate": hit_rate,
            "n_selections": 0,
            "n_with_first_hit": 0,
            "ok": False,
            "note": "no successful selections after drop training + first-of-ID-lap",
        }

    hits = load_first_hits([qpath], sel)
    hit_key = hits[["start_num", "end_num", "selection_unix_ms", "event_unix_ms"]].rename(
        columns={"event_unix_ms": "first_hit_unix_ms"}
    )
    ep = sel.merge(hit_key, on=["start_num", "end_num", "selection_unix_ms"], how="left")
    ep["appear_unix_ms"] = ep["selection_unix_ms"] - ep["movement_time_s"].astype(float) * 1000.0
    ep["confirm_unix_ms"] = ep["selection_unix_ms"]
    ep["movement_only_s"] = (ep["first_hit_unix_ms"] - ep["appear_unix_ms"]) / 1000.0
    ep["dwell_s"] = (ep["confirm_unix_ms"] - ep["first_hit_unix_ms"]) / 1000.0
    ep["mt_recon_s"] = ep["movement_only_s"] + ep["dwell_s"]
    ep["mt_recon_err_s"] = ep["mt_recon_s"] - ep["movement_time_s"].astype(float)
    ep.insert(0, "subject", subject)
    ep.insert(1, "run", run)
    for k, v in labels.items():
        ep[k] = v

    has_hit = ep["first_hit_unix_ms"].notna()
    err = ep.loc[has_hit, "mt_recon_err_s"].astype(float)
    n_hit = int(has_hit.sum())
    n_ok = int((err.abs() <= RECON_TOL_S).sum()) if n_hit else 0
    meta = {
        "subject": subject,
        "run": run,
        **labels,
        "quest_file": qpath.name,
        "n_iso_selections": n_iso,
        "n_success": n_success,
        "hit_rate": hit_rate,
        "n_selections": n_sel,
        "n_with_first_hit": n_hit,
        "frac_with_first_hit": n_hit / n_sel if n_sel else float("nan"),
        "n_recon_ok": n_ok,
        "frac_recon_ok": n_ok / n_hit if n_hit else float("nan"),
        "median_mt_s": float(ep["movement_time_s"].median()) if n_sel else float("nan"),
        "median_movement_only_s": float(ep.loc[has_hit, "movement_only_s"].median()) if n_hit else float("nan"),
        "median_dwell_s": float(ep.loc[has_hit, "dwell_s"].median()) if n_hit else float("nan"),
        "median_recon_err_s": float(err.median()) if n_hit else float("nan"),
        "n_negative_dwell": int((ep.loc[has_hit, "dwell_s"] < 0).sum()) if n_hit else 0,
        "n_negative_movement": int((ep.loc[has_hit, "movement_only_s"] < 0).sum()) if n_hit else 0,
        "ok": bool(n_hit >= 0.8 * n_sel and (n_ok >= 0.9 * n_hit if n_hit else False)),
    }
    return ep, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_bout_args(p)
    p.add_argument("--participants", nargs="+", default=None, help="Default: 80 81")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/participants/_mt_dwell_check/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parts = args.participants
    if parts is None and not args.participant:
        parts = ["80", "81"]
    if args.participant and not parts:
        parts = [args.participant]

    bouts: list[Path] = []
    for part in parts:
        ns = argparse.Namespace(
            participant=part,
            speed=args.speed or None,
            interaction=args.interaction,
            bout_dir=args.bout_dir,
        )
        if ns.speed is None:
            for speed in scan_bout_names(part, None):
                ns_s = argparse.Namespace(
                    participant=part,
                    speed=speed,
                    interaction=args.interaction,
                    bout_dir=None,
                )
                bouts.extend(discover_quest_bouts(ns_s))
        else:
            bouts.extend(discover_quest_bouts(ns))
    if args.bout_dir:
        bouts = [Path(args.bout_dir)]

    out_dir = args.out_dir or (DATA_ROOT / "participants" / "_mt_dwell_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    all_ep: list[pd.DataFrame] = []
    for bout in bouts:
        ep, meta = split_episode_mt(bout)
        summaries.append(meta)
        if not ep.empty:
            bout_out = bout / STAGE_DIRS["gait"] / "mt_dwell_check"
            bout_out.mkdir(parents=True, exist_ok=True)
            ep.to_csv(bout_out / "episodes_mt_dwell.csv", index=False)
            all_ep.append(ep)
        flag = "OK" if meta["ok"] else "CHECK"
        print(
            f"{flag}  {meta['subject']}/{meta['run']}: "
            f"sel={meta['n_selections']}  first_hit={meta['n_with_first_hit']} "
            f"({100 * meta.get('frac_with_first_hit', float('nan')):.0f}%)  "
            f"recon={meta.get('n_recon_ok', 0)}/{meta['n_with_first_hit']}  "
            f"med MT={meta['median_mt_s']:.2f}s  "
            f"move={meta['median_movement_only_s']:.2f}s  "
            f"dwell={meta['median_dwell_s']:.2f}s"
        )

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    if all_ep:
        pd.concat(all_ep, ignore_index=True).to_csv(out_dir / "episodes_mt_dwell_all.csv", index=False)
    print(f"\nWrote {out_dir}")
    print("movement_only_s = appear->first hit;  dwell_s = first hit->confirm;  MT = appear->confirm")


if __name__ == "__main__":
    main()
