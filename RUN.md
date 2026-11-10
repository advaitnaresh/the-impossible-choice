# Run and reproduce

## Data

Download the public [NYU regional Big Data Bowl dataset](https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data) and point `BDB_DATA_DIR` at its `data/` folder. The repository intentionally includes derived results and screenshots, not the 2021 raw tracking files. The optional nflverse 2021 play-by-play and player crosswalk are downloaded when `--download-pbp` is used.

## Setup

```bash
git clone https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data.git ../nfl-big-data-bowl-regional-event-data
export BDB_DATA_DIR="$(cd ../nfl-big-data-bowl-regional-event-data && pwd)/data"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run those commands from this project checkout; set `BDB_DATA_DIR` to another absolute path if the data is already available elsewhere.

## Analyze

```bash
python3 analysis.py --max-games 4 --download-pbp --force
python3 analysis.py --download-pbp --force
python3 scripts/render_snapshots.py
python3 scripts/render_readme_animation.py
python3 -m pytest -q
```

The short run is a smoke test and overwrites the saved outputs; run the full analysis afterward to restore the complete result set. The full run writes compact per-game caches under `outputs/cache/`, which is excluded from Git. Each tracking game is loaded separately to limit memory use.

## Explore

```bash
streamlit run app.py
```

`outputs/featured_play_replay.html` is a self-contained replay for the selected actual play. The app requires the raw tracking data through `BDB_DATA_DIR` so it can load any browsed play. The precomputed CSV files and PNG snapshots can be inspected without the raw data.

## Files to inspect

| File | Purpose |
| --- | --- |
| `METRIC.md` | Metric eligibility, geometry, thresholds, and limits |
| `outputs/finding.md` | Data-derived findings and held-out result |
| `outputs/ddt_play_results.csv` | One row per eligible play |
| `outputs/ddt_conflict_frames.csv` | Winning pair-defender triple for each flagged frame |
| `outputs/ddt_conflict_segments.csv` | Continuous flagged windows |
| `outputs/baseline_comparison.csv` | Held-out B0/B1/B2 comparison |
| `outputs/coverage_summary.csv` | Descriptive man/zone summary |
| `snapshots/` | README GIF and stills rendered from the saved results and actual tracking |
