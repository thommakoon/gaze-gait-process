"""ISO-style Fitts trial filter: drop training and the first target of each ID lap.

Ring: first dot of each A×W lap (movement starts from the previous layout).
Rectangle: opening L of each ID (Quest ``opening_selection``; not a Fitts movement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_id_openers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the first recorded target of each contiguous A×W lap."""
    if df.empty:
        return df
    out = df.sort_values("selection_unix_ms").reset_index(drop=True)
    n = len(out)
    opener = np.zeros(n, dtype=bool)
    if "opening_selection" in out.columns:
        opener |= out["opening_selection"].fillna(False).to_numpy(dtype=bool)

    if "ring_index" in out.columns and out["ring_index"].notna().any():
        idx = pd.to_numeric(out["ring_index"], errors="coerce").to_numpy()
        if "is_training" in out.columns:
            train = out["is_training"].fillna(False).to_numpy(dtype=bool)
        else:
            train = np.zeros(n, dtype=bool)
        new_lap = np.ones(n, dtype=bool)
        new_lap[1:] = (idx[1:] != idx[:-1]) | (train[1:] != train[:-1])
        opener |= new_lap
    elif n:
        opener[0] = True

    return out.loc[~opener].reset_index(drop=True)


def drop_training_and_id_openers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop training laps, then the first target of each remaining ID lap."""
    if df.empty:
        return df
    out = df.sort_values("selection_unix_ms").reset_index(drop=True)
    if "is_training" in out.columns:
        out = out.loc[out["is_training"] != True].copy()  # noqa: E712
        out = out.reset_index(drop=True)
    if out.empty:
        return out
    return drop_id_openers(out)


def assign_id_repetition(df: pd.DataFrame) -> pd.DataFrame:
    """1-based visit number of each A×W ID block within a bout.

    Standing bouts cycle the 6 IDs twice; walking cycles them three times.
    ``id_rep`` is the k-th visit of that ``ring_name`` (or ``ring_index``).
    """
    if df.empty:
        out = df.copy()
        out["id_rep"] = pd.Series(dtype="Int64")
        return out
    out = df.copy()
    bout_keys = [c for c in ("participant", "speed", "interaction") if c in out.columns]
    if not bout_keys or "selection_unix_ms" not in out.columns:
        out["id_rep"] = pd.NA
        return out
    id_col = "ring_name" if "ring_name" in out.columns else (
        "ring_index" if "ring_index" in out.columns else None
    )
    if id_col is None:
        out["id_rep"] = pd.NA
        return out

    parts: list[pd.DataFrame] = []
    for _, g in out.groupby(bout_keys, dropna=False, sort=False):
        g = g.sort_values("selection_unix_ms")
        ids = g[id_col].astype(str).to_numpy()
        new_lap = np.ones(len(g), dtype=bool)
        if len(g) > 1:
            new_lap[1:] = ids[1:] != ids[:-1]
        lap_id = np.cumsum(new_lap)
        g = g.copy()
        g["_lap_id"] = lap_id
        first = (
            g.groupby([id_col, "_lap_id"], dropna=False)["selection_unix_ms"]
            .min()
            .reset_index()
        )
        first["id_rep"] = (
            first.groupby(id_col, dropna=False)["selection_unix_ms"]
            .rank(method="dense")
            .astype(int)
        )
        g = g.merge(first[[id_col, "_lap_id", "id_rep"]], on=[id_col, "_lap_id"], how="left")
        g = g.drop(columns=["_lap_id"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def keep_id_reps(
    df: pd.DataFrame,
    *,
    min_rep: int = 2,
    max_rep: int = 3,
    walking_only: bool = True,
) -> pd.DataFrame:
    """Keep ID visits in ``[min_rep, max_rep]``.

    Default: walking drops cycle 1 (practice) and keeps 2–3. Standing only
    has two cycles, so it is left as-is when ``walking_only`` is True.
    """
    if df.empty:
        return df
    out = df if "id_rep" in df.columns else assign_id_repetition(df)
    rep = pd.to_numeric(out["id_rep"], errors="coerce")
    in_range = rep.isna() | ((rep >= min_rep) & (rep <= max_rep))
    if walking_only and "speed_group" in out.columns:
        is_walk = out["speed_group"].astype(str) == "walking"
        keep = (~is_walk) | in_range
    else:
        keep = in_range
    return out.loc[keep].copy()


def keep_first_id_reps(df: pd.DataFrame, max_rep: int = 2) -> pd.DataFrame:
    """Deprecated alias: keep visits 1..``max_rep`` (including the practice cycle)."""
    return keep_id_reps(df, min_rep=1, max_rep=max_rep, walking_only=False)
