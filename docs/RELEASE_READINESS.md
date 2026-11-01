# Feature-branch release readiness

This branch has a tested source package, Docker smoke check, read-only market
observation, causal four-class research labels, and entry safety guards. The
public `v0.2.0` tag remains on its existing commit. This document does not
claim profitable or production-ready trading.

## Verified locally

- Unit tests, dependency consistency, wheel build and allowlisted ZIP build.
- Public Kalshi quote and official settlement reads without credentials or orders.
- The release ZIP excludes raw observations, keys, local databases and logs.
- `quickstart.py --check` deliberately returns status 2 for the bundled legacy
  eight-feature model. The nine-feature trading path does not silently use it.

## Required before promoting this branch

1. Collect genuine, timestamped Coinbase depth snapshots long enough to cover
   the intended chronological training, calibration and out-of-sample periods.
   Start or resume the read-only recorder with
   `python -m kalshi_ml.data.orderbook --output data/raw/coinbase_obi.csv --interval 4`.
   It holds an exclusive file lock, checks existing rows before appending and
   skips repeated/regressed book sequence or time. `wc -l data/raw/coinbase_obi.csv`
   shows the local row count (including the header); the file is Git-ignored.
2. Train the nine-feature artifact from those snapshots, retain its provenance,
   and pass `quickstart.py --check --model <trusted-artifact>`.
   First run `python -m kalshi_ml.data.quality --minutes 3000` and inspect
   `data/processed/quality_report.json`; its collection gate is about data
   availability, not model performance.
3. Join frozen pre-settlement predictions to recorded Kalshi quote observations
   and official outcomes. Review calibration, executable price/depth, fees,
   stale-data exclusions and time-block out-of-sample performance.
4. Run the replacement artifact in paper mode, inspect the SQLite journal and
   failure/restart behavior, then review the draft PR and cut a new release.

The OHLCV labeler still uses an analytical quote proxy. Its four classes cannot
stand in for actual Kalshi order fills or prove the TP/SL strategy's return.
