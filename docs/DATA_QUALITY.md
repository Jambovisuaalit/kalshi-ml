# Coinbase OBI and Kalshi observation quality

Run the read-only audit while both recorders continue writing:

```bash
python -m kalshi_ml.data.quality --minutes 3000
```

An installed package also provides `kalshi-ml-quality`. Outputs default to
`data/processed/quality_report.json` and `data/processed/kalshi_joined_quotes.csv`.
Both remain local and Git-ignored. Use an explicit timezone-aware `--as-of`
to reproduce a past audit. A concurrently incomplete final JSONL line is
reported and ignored; a malformed complete line fails the audit.

For each UTC minute decision boundary, OBI uses only the latest snapshot
received by that instant. Both the exchange timestamp and receipt time must
be no more than 10 seconds old. The report distinguishes missing and stale
minutes, shows coverage, the longest unusable run, the largest snapshot gap
and the 95th percentile age of accepted observations. No synthetic OBI values
are filled in.

Kalshi orderbook records retain their own receipt times and exact source URL.
They are joined by ticker to an official binary settlement only when the
quote was received **before** the exchange's official settlement timestamp
and that settlement had been observed by the audit cutoff. The CSV keeps
the source journal line numbers, displayed YES/NO bids and complementary asks,
settlement payout, and both settlement timestamps. The report counts late,
unsettled and incomplete quotes separately and shows a chronological 70/30
split by distinct settled contract. Multiple snapshots of one contract never
become independent outcome labels.

`collection_ready` is a **data availability** gate: at least 3,000 audited
minute decisions, 95% fresh OBI coverage, and 100 settled contracts with
complete joined quotes **inside the same OBI window**. It does not certify model calibration, out-of-sample
performance, depth, fill probability or profitability. Train/validation must
still use frozen predictions made before their corresponding settlements.
