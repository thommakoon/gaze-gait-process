# 02_analysis — layered sub-pipelines

```text
scripts/02_analysis/
  core/        # 3a trial clocks
  features/    # 3b gait / saccade / confirm features
  summaries/   # 3c performance tables
  cross/       # 3d paper figures (writes data/.../finalize/)
  run_paper.py
  layers.py
```

```powershell
cd scripts\02_analysis
uv run python run_paper.py --list
uv run python run_paper.py --plots-only
uv run python run_paper.py
```

Outputs mirror the same layers under `data/participants/_02_analysis/{core,features,summaries}/`.
Paper PNGs stay under `…/_02_analysis/finalize/` (Overleaf paths unchanged).

New analysis: add a script under `features/` or `cross/`, register in `layers.py` if needed.
Archived explorers: `old_analysis/2026-10-07/`.
