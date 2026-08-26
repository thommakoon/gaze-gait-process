# 02_08 — standing vs walking plots

Line plots like the paper figure: Standing / Walking on x, Head / Hand / Eye as series, Ring vs Rectangle as panels. Mean±SE of **person medians** (analysis 7 unit).

Does not re-run analyses. Needs `check_mt_dwell` episodes. Time metrics are mean±SE of **person medians**. Fixation count is mean±SE of **person means** (average fixations per selection).

```powershell
cd scripts\02_analysis
uv run python 02_08_stand_walk_plots/plot_stand_walk.py
uv run python 02_08_stand_walk_plots/plot_stand_walk.py --participants 22 32
```

Writes `data/participants/_stand_walk_plots/`:

| File | Metric |
|------|--------|
| `final_angle_deg.png` | Last Quest cursor–target angle at pinch-cut (125 ms before confirm) |
| `latency_s.png` | Appear → leave previous (last frame still on start) |
| `transit_s.png` | Leave previous → first hit |
| `dwell_s.png` | First hit → confirm |
| `movement_time_s.png` | Appear → confirm |
| `movement_only_s.png` | Appear → first hit |
| `throughput_bps.png` | Nominal ID / MT (bits/s) |
| `n_fixation.png` | Mean fixations per selection (person mean, then mean±SE). I-VT stills of the active pointer (Quest wall, 5-frame median, ≥80 ms) |

People without `episodes_mt_dwell` are skipped (run analysis 1 first).
