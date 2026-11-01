# Kalshi market observation journal

Run the public, read-only collector in a separate process:

```bash
python -m kalshi_ml.data.kalshi_observer --environment PROD --output data/raw/kalshi
```

After installation, `kalshi-ml-observe` runs the same command. A non-blocking
directory lock prevents a second collector from writing to the same journal.
Use a different output directory for DEMO and PROD; the reader rejects mixed
environment or series records.

Use `--once` for a single poll or `--poll-sec 4` for the default continuous cadence. No API key or trading flag is needed. The process does not call account or order endpoints. Stop it with Ctrl-C. Run only one collector against a journal directory at a time.

`kalshi_observations.jsonl` is an append-only, fsynced journal with file permissions `0600`. It is excluded by `.gitignore`. Each event records the request start, response receipt, exact public API source URL, environment, ticker and raw JSON response. Market and orderbook responses have **separate** timestamps: they are not an atomic quote. The parsed top-of-book bid is the highest displayed bid. The owned-side ask is the complement of the opposite-side bid under Kalshi's single-book convention. Sizes describe the displayed opposite bids, not a guaranteed fill. Empty books retain null prices. The collector does not record model probabilities or claim executable fills.

The observer remembers every recorded market ticker across restarts and polls it until the public market endpoint reports `settled` or `finalized` with an official result or settlement value. It stores the exchange's `settlement_ts` separately from the time the response arrived. Closed or unresolved markets do not get a guessed outcome. Invalid or contradictory response fields stop collection visibly. Existing incomplete/invalid journal lines also cause a stop so they are not silently skipped.

This journal supplies evidence for later validation. It does **not** make the OHLCV triple-barrier labeler's analytical quote proxy into a historical Kalshi fill or exit. A later join should require ticker, temporal order, documented model training cutoff, quote freshness, displayed size, fees and official payout provenance before any performance claim.
