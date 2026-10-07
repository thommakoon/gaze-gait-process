#!/usr/bin/env python3
"""Shift-null for matched no-IC vs 1-IC transit MT (does NOT edit existing data).

Reads episodes_all.csv read-only. Real 0/1 labels = n_ic_in_transit.
Fake labels: circular-shift all bout IC times by a random offset, then
recount how many fall in each leave→hit window (MT unchanged).

Per shuffle: same matched n=min(n0,n1) design as matched_noIC_vs_1IC_aiming.
Compare person-mean Δ transit_ms (1-IC − 0-IC) real vs null distribution.

Writes ONLY under this script's analysis_out folder — never overwrites
matched_noIC_vs_1IC_aiming/ or episodes_all.csv.

Usage (from scripts/02_analysis/):
    uv run python 02_05_cursor_stability/matched_noIC_vs_1IC_shift_null.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import ensure_analysis_path

ensure_analysis_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _paths import INTERACTIONS, STAGE_DIRS, WALKING_BOUTS, analysis_out, bout_dir
from across_people import INTER_STYLE, part_name
from fitts_gait_onset import grid_t0_ns, load_pc_offset_ns
from phase_ic_counts import _ensure_bad_ic_windows, _foot_ics_ms
from support_state_enrichment import _load_usable_unique_ids

OUT = analysis_out(__file__)  # new folder only
EP = analysis_out("02_05_cursor_stability/transit_ic_jitter.py") / "episodes_all.csv"
COHORT = {part_name(x) for x in _load_usable_unique_ids()}
RNG = np.random.default_rng(0)
N_SHUFFLE = 200
METRICS = (("transit_ms", "Transit MT (ms)"), ("mt_residual_ms", "ID-residual MT (ms)"))


def _parse_run(run: str) -> tuple[str, str, str] | None:
    if not isinstance(run, str) or "_" not in run or run.startswith("Practice"):
        return None
    speed, inter = run.split("_", 1)
    if speed not in WALKING_BOUTS or inter not in INTERACTIONS:
        return None
    layout = "rect" if "Rectangle" in speed else "ring"
    return layout, speed, inter


def _fit_id_residual(df: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(df["amplitude_m"], errors="coerce")
    w = pd.to_numeric(df["width_m"], errors="coerce")
    y = pd.to_numeric(df["transit_ms"], errors="coerce")
    id_ = np.log2(a / w + 1.0)
    ok = np.isfinite(id_) & np.isfinite(y) & (w > 0)
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    if int(ok.sum()) < 10:
        return resid
    x = id_[ok].to_numpy(dtype=float)
    yy = y[ok].to_numpy(dtype=float)
    X = np.column_stack([np.ones(x.size), x])
    coef, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    pred = coef[0] + coef[1] * id_
    resid.loc[ok] = y[ok] - pred[ok]
    return resid


def _count_in_window(ics: np.ndarray, leave: float, hit: float) -> int:
    if ics.size == 0 or not (np.isfinite(leave) and np.isfinite(hit) and hit > leave):
        return 0
    return int(np.sum((ics > leave) & (ics < hit)))


def _circular_shift(ics: np.ndarray, shift_ms: float, t_lo: float, t_hi: float) -> np.ndarray:
    if ics.size == 0:
        return ics
    span = float(t_hi - t_lo)
    if not np.isfinite(span) or span <= 0:
        return ics + shift_ms
    return t_lo + np.mod(ics - t_lo + shift_ms, span)


def _match_deltas(
    df: pd.DataFrame,
    *,
    label_col: str,
    layout_keep: dict[str, set[str]],
    cell_keep: dict[tuple[str, str], set[str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Person-level Δ (one_ic - no_ic) for each metric; matched n per person."""
    rows = []
    for (layout, inter), g in df.groupby(["layout", "interaction"], dropna=False):
        keep = layout_keep.get(layout, set())
        if (layout, inter) in cell_keep:
            keep = keep & cell_keep[(layout, inter)]
        if len(keep) < 5:
            continue
        for pid in sorted(keep):
            sub = g[g["participant"] == pid]
            a = sub[sub[label_col] == "no_ic"]
            b = sub[sub[label_col] == "one_ic"]
            n = int(min(len(a), len(b)))
            if n < 1:
                continue
            a_i = a.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
            b_i = b.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
            rec = {
                "participant": pid,
                "layout": layout,
                "interaction": inter,
                "n_matched": n,
            }
            for col, _ in METRICS:
                va = pd.to_numeric(a_i[col], errors="coerce")
                vb = pd.to_numeric(b_i[col], errors="coerce")
                m0 = float(va.mean()) if va.notna().any() else np.nan
                m1 = float(vb.mean()) if vb.notna().any() else np.nan
                rec[f"{col}_no_ic"] = m0
                rec[f"{col}_one_ic"] = m1
                rec[f"{col}_delta"] = m1 - m0
            rows.append(rec)
    return pd.DataFrame(rows)


def _summarize(person: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for (layout, inter), g in person.groupby(["layout", "interaction"], dropna=False):
        for col, lab in METRICS:
            d = pd.to_numeric(g[f"{col}_delta"], errors="coerce").to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            n = int(d.size)
            if n < 5:
                continue
            try:
                p = float(stats.wilcoxon(d, alternative="greater").pvalue)
            except ValueError:
                p = float("nan")
            rows.append(
                {
                    "source": source,
                    "layout": layout,
                    "interaction": inter,
                    "metric": col,
                    "label": lab,
                    "n_people": n,
                    "mean_delta": float(np.mean(d)),
                    "median_delta": float(np.median(d)),
                    "se_delta": float(np.std(d, ddof=1) / np.sqrt(n)),
                    "wilcoxon_p_greater": p,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if not EP.is_file():
        raise SystemExit(f"missing {EP}")
    # read-only load
    ep = pd.read_csv(EP)
    ep["participant"] = ep["subject"].map(
        lambda s: part_name(s) if not str(s).startswith("participant") else s
    )
    ep = ep[ep["participant"].isin(COHORT)].copy()
    parsed = ep["run"].map(_parse_run)
    ep = ep.loc[parsed.notna()].copy()
    parsed = parsed.loc[ep.index]
    ep["layout"] = [t[0] for t in parsed]
    ep["speed"] = [t[1] for t in parsed]
    ep["interaction"] = [t[2] for t in parsed]
    ep["n_ic_real"] = pd.to_numeric(ep["n_ic_in_transit"], errors="coerce")
    ep["transit_ms"] = pd.to_numeric(ep["transit_ms"], errors="coerce")
    ep["mt_residual_ms"] = _fit_id_residual(ep)
    ep["leave"] = pd.to_numeric(ep["leave_unix_ms"], errors="coerce")
    ep["hit"] = pd.to_numeric(ep["first_hit_unix_ms"], errors="coerce")

    # load real IC times per bout (cached); also bout time span from quest grid
    ic_cache: dict[tuple[str, str, str], np.ndarray] = {}
    span_cache: dict[tuple[str, str, str], tuple[float, float]] = {}

    def bout_ics(pid: str, speed: str, inter: str) -> np.ndarray:
        key = (pid, speed, inter)
        if key in ic_cache:
            return ic_cache[key]
        n = int(pid.replace("participant", ""))
        bout = bout_dir(n, speed, inter)
        try:
            offset_ns, _ = load_pc_offset_ns(bout)
            t0 = grid_t0_ns(bout)
        except Exception:
            ic_cache[key] = np.array([], dtype=float)
            return ic_cache[key]
        windows = _ensure_bad_ic_windows(bout)
        run = f"{speed}_{inter}"
        parts = []
        for foot in ("left", "right"):
            arr = _foot_ics_ms(
                bout, pid, run, foot, t0=t0, offset_ns=offset_ns, windows=windows
            )
            if arr is not None and len(arr):
                parts.append(np.asarray(arr, dtype=float))
        ics = np.concatenate(parts) if parts else np.array([], dtype=float)
        ic_cache[key] = ics
        # span from quest grid if present
        grid = bout / STAGE_DIRS["grid"] / "quest_200hz.csv"
        if grid.is_file():
            t_ns = pd.read_csv(grid, usecols=["t_utc_ns"])["t_utc_ns"].to_numpy(dtype=float)
            unix = (t_ns - offset_ns) / 1e6
            span_cache[key] = (float(np.nanmin(unix)), float(np.nanmax(unix)))
        elif ics.size:
            span_cache[key] = (float(ics.min()), float(ics.max()))
        else:
            span_cache[key] = (np.nan, np.nan)
        return ics

    # ensure caches filled
    for (pid, speed, inter), _ in ep.groupby(["participant", "speed", "interaction"]):
        bout_ics(str(pid), str(speed), str(inter))

    # --- REAL: keep real 0/1 only ---
    real = ep[ep["n_ic_real"].isin([0, 1])].copy()
    real["klass"] = np.where(real["n_ic_real"] == 0, "no_ic", "one_ic")

    def build_keeps(df: pd.DataFrame, label_col: str = "klass"):
        cell_keep: dict[tuple[str, str], set[str]] = {}
        for (layout, inter), g in df.groupby(["layout", "interaction"], dropna=False):
            counts = g.groupby(["participant", label_col]).size().unstack(fill_value=0)
            if "no_ic" not in counts.columns or "one_ic" not in counts.columns:
                continue
            keep = counts.index[(counts["no_ic"] >= 1) & (counts["one_ic"] >= 1)]
            if len(keep) >= 5:
                cell_keep[(layout, inter)] = {str(p) for p in keep}
        layout_keep: dict[str, set[str]] = {}
        for layout in ("ring", "rect"):
            sets = [
                cell_keep[(layout, inter)]
                for inter in INTERACTIONS
                if (layout, inter) in cell_keep
            ]
            if len(sets) == 3:
                layout_keep[layout] = set.intersection(*sets)
            elif sets:
                layout_keep[layout] = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return cell_keep, layout_keep

    cell_real, layout_real = build_keeps(real, "klass")
    person_real = _match_deltas(
        real, label_col="klass", layout_keep=layout_real, cell_keep=cell_real, rng=RNG
    )
    sum_real = _summarize(person_real, "real")

    # --- FAKE shuffles ---
    shuffle_rows = []
    for s in range(N_SHUFFLE):
        # one shift per bout
        shifted: dict[tuple[str, str, str], np.ndarray] = {}
        for key, ics in ic_cache.items():
            t_lo, t_hi = span_cache.get(key, (np.nan, np.nan))
            if ics.size == 0 or not np.isfinite(t_lo):
                shifted[key] = ics
                continue
            span = t_hi - t_lo
            shift = float(RNG.uniform(0.0, span)) if span > 0 else 0.0
            shifted[key] = _circular_shift(ics, shift, t_lo, t_hi)

        fake_n = []
        for _, r in ep.iterrows():
            key = (str(r["participant"]), str(r["speed"]), str(r["interaction"]))
            ics = shifted.get(key, np.array([]))
            fake_n.append(_count_in_window(ics, float(r["leave"]), float(r["hit"])))
        fake = ep.copy()
        fake["n_ic_fake"] = fake_n
        fake = fake[fake["n_ic_fake"].isin([0, 1])].copy()
        fake["klass"] = np.where(fake["n_ic_fake"] == 0, "no_ic", "one_ic")
        cell_f, layout_f = build_keeps(fake, "klass")
        # use REAL complete-case people so N matches real figure when possible
        # intersect with who still has both fake classes
        layout_use = {
            lay: layout_real.get(lay, set()) & layout_f.get(lay, set())
            for lay in ("ring", "rect")
        }
        cell_use = {
            k: (cell_real.get(k, set()) & cell_f.get(k, set()))
            for k in set(cell_real) | set(cell_f)
        }
        person_f = _match_deltas(
            fake,
            label_col="klass",
            layout_keep=layout_use,
            cell_keep=cell_use,
            rng=RNG,
        )
        if person_f.empty:
            continue
        for (layout, inter), g in person_f.groupby(["layout", "interaction"]):
            for col, _ in METRICS:
                d = pd.to_numeric(g[f"{col}_delta"], errors="coerce")
                d = d[np.isfinite(d)]
                if len(d) < 5:
                    continue
                shuffle_rows.append(
                    {
                        "shuffle": s,
                        "layout": layout,
                        "interaction": inter,
                        "metric": col,
                        "n_people": int(len(d)),
                        "mean_delta": float(d.mean()),
                        "median_delta": float(d.median()),
                    }
                )
        if (s + 1) % 50 == 0:
            print(f"shuffle {s + 1}/{N_SHUFFLE}")

    shuf = pd.DataFrame(shuffle_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    person_real.to_csv(OUT / "person_real_matched.csv", index=False)
    sum_real.to_csv(OUT / "summary_real.csv", index=False)
    shuf.to_csv(OUT / "shuffle_mean_deltas.csv", index=False)

    # null summary vs real
    cmp_rows = []
    for _, r in sum_real.iterrows():
        layout, inter, metric = r["layout"], r["interaction"], r["metric"]
        null = shuf[
            (shuf["layout"] == layout)
            & (shuf["interaction"] == inter)
            & (shuf["metric"] == metric)
        ]["mean_delta"].to_numpy(dtype=float)
        null = null[np.isfinite(null)]
        real_d = float(r["mean_delta"])
        if null.size == 0:
            continue
        # p = P(fake mean_delta >= real) — is real larger than chance?
        p_ge = float(np.mean(null >= real_d))
        cmp_rows.append(
            {
                "layout": layout,
                "interaction": inter,
                "metric": metric,
                "label": r["label"],
                "n_people_real": int(r["n_people"]),
                "real_mean_delta": real_d,
                "real_median_delta": float(r["median_delta"]),
                "fake_mean_delta": float(np.mean(null)),
                "fake_se": float(np.std(null, ddof=1) / np.sqrt(null.size)),
                "fake_p05": float(np.quantile(null, 0.05)),
                "fake_p95": float(np.quantile(null, 0.95)),
                "n_shuffles": int(null.size),
                "p_fake_ge_real": p_ge,
            }
        )
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT / "real_vs_shift_null.csv", index=False)

    # plot: real vs fake mean Δ transit_ms
    for layout, title in (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)")):
        sub = cmp[(cmp["layout"] == layout) & (cmp["metric"] == "transit_ms")]
        if sub.empty:
            continue
        # enforce same N annotation
        ns = sub["n_people_real"].unique()
        n_star = int(ns[0]) if len(ns) == 1 else int(ns.min())
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        inters = [i for i in INTERACTIONS if i in set(sub["interaction"])]
        x = np.arange(len(inters))
        w = 0.36
        real_y, fake_y, fake_se, labels = [], [], [], []
        for inter in inters:
            row = sub[sub["interaction"] == inter].iloc[0]
            real_y.append(float(row["real_mean_delta"]))
            fake_y.append(float(row["fake_mean_delta"]))
            fake_se.append(float(row["fake_se"]))
            lab = INTER_STYLE.get(inter, {}).get("label", inter)
            p = float(row["p_fake_ge_real"])
            labels.append(f"{lab}\np_null≥real={p:.2g}")
        ax.bar(
            x - w / 2,
            real_y,
            w,
            label="Real 1−0 IC",
            color="#c0392b",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + w / 2,
            fake_y,
            w,
            yerr=fake_se,
            label=f"Shift-null ({N_SHUFFLE}×)",
            color="0.65",
            edgecolor="black",
            linewidth=0.4,
            capsize=3,
        )
        ax.axhline(0, color="0.4", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Δ transit MT (ms)  [1-IC − 0-IC]")
        ax.set_title(
            f"{title}: real vs IC-time shift null\n"
            f"matched · complete-case N={n_star} · MT values unchanged, labels only"
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"real_vs_null_transit_mt_{layout}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote real_vs_null_transit_mt_{layout}.png")

    # residual plot too
    for layout, title in (("ring", "Ring (2D)"), ("rect", "Rectangle (1D)")):
        sub = cmp[(cmp["layout"] == layout) & (cmp["metric"] == "mt_residual_ms")]
        if sub.empty:
            continue
        ns = sub["n_people_real"].unique()
        n_star = int(ns[0]) if len(ns) == 1 else int(ns.min())
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        inters = [i for i in INTERACTIONS if i in set(sub["interaction"])]
        x = np.arange(len(inters))
        w = 0.36
        real_y, fake_y, fake_se, labels = [], [], [], []
        for inter in inters:
            row = sub[sub["interaction"] == inter].iloc[0]
            real_y.append(float(row["real_mean_delta"]))
            fake_y.append(float(row["fake_mean_delta"]))
            fake_se.append(float(row["fake_se"]))
            lab = INTER_STYLE.get(inter, {}).get("label", inter)
            p = float(row["p_fake_ge_real"])
            labels.append(f"{lab}\np_null≥real={p:.2g}")
        ax.bar(
            x - w / 2,
            real_y,
            w,
            label="Real 1−0 IC",
            color="#c0392b",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + w / 2,
            fake_y,
            w,
            yerr=fake_se,
            label=f"Shift-null ({N_SHUFFLE}×)",
            color="0.65",
            edgecolor="black",
            linewidth=0.4,
            capsize=3,
        )
        ax.axhline(0, color="0.4", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Δ ID-residual MT (ms)  [1-IC − 0-IC]")
        ax.set_title(
            f"{title}: real vs IC-time shift null (ID-residual)\n"
            f"matched · complete-case N={n_star}"
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            OUT / f"real_vs_null_residual_mt_{layout}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    print(cmp.to_string(index=False))
    print(f"-> {OUT}")
    print("NOTE: did not modify episodes_all.csv or matched_noIC_vs_1IC_aiming/")


if __name__ == "__main__":
    main()
