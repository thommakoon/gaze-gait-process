"""lin2025 processed outputs  ->  per-session gazegait artifacts.

Reads stride and event tables from StrokeGait/processed/<subj>/<visit>/ and
re-joins t_utc_ns onto each event using foot_t_utc_map.parquet emitted by
bridge/to_lin2025.

Produces:
    derived/<session>/gait/lf_strides.parquet
    derived/<session>/gait/rf_strides.parquet
    derived/<session>/gait/lf_events.parquet   (heel_strike, toe_off)
    derived/<session>/gait/rf_events.parquet

Each row carries a UTC-ns timestamp so downstream fusion can event-lock any
head/eye signal to gait events without any further alignment work.
"""

from __future__ import annotations

from pathlib import Path


def import_session(
    session_id: str,
    lin2025_data_root: Path,
    subject: str,
    visit: str,
    derived_dir: Path,
) -> None:
    """Read lin2025's processed outputs for one subject/visit and write the
    UTC-stamped gait artifacts into derived/<session>/gait/.
    """
    raise NotImplementedError
