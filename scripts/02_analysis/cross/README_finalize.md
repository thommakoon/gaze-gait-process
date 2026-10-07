# Finalize paper figures

Pastel restyle of the keep-list plots for the paper.

## Code (here)

`scripts/02_analysis/finalize/`

| Script | Role |
|--------|------|
| `run_all.py` | Orchestrator |
| `compute_transit.py` | Leave→first-hit `transit_s` on N=24 episodes |
| `compute_saccade_stand_walk.py` | Stand vs walk Neon **I-DT** saccade counts (needs Practice 200 Hz grids) |
| `fig_A_factor.py` | Size / amplitude: hit, MT, transit, throughput |
| `fig_B_stand_walk.py` | Overall stand vs walk (+ saccade) |
| `fig_C_gait.py` | Walking gait coupling (10% bins) |
| `style.py` | Pastel colors |

Also restored dependency: `scripts/02_analysis/02_03_saccade/cursor_ivt.py` (pointer I-VT fallback).

Practice 200 Hz grids: `scripts/01_clean/grid_practice_n24.py` (or normal `run_pipeline.py` on Practice*).

```powershell
cd scripts\02_analysis
uv run python finalize/run_all.py
# plots only (shared CSVs already built):
uv run python finalize/run_all.py --plots-only
# optional leave→hit pointing distance (slow):
uv run python finalize/fig_C_gait.py --leave-hit-distance
```

## Outputs

`data/participants/_02_analysis/finalize/`

- `shared/episodes_with_transit.csv`
- `A_factor/` — main effects of **size**, **amplitude**, **locomotion** (hit/MT/transit/throughput; layout × modality panels; stand+walk averaged for size/amplitude)
- `B_stand_walk/` — hit, MT, transit, throughput, saccade_count (Neon I-DT)
- `C_gait/` — pointing speed/distance, confirm count/distance, **saccade count** vs LF phase

**Note:** Default pointing distance in C uses appear→confirm continuous samples. Re-run C with `--leave-hit-distance` for strict leave→first-hit distance.