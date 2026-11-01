# Bundled pretrained model — 0.2.0

**Legacy artifact:** the post-0.2.0 nine-feature OBI/calibration pipeline rejects
this eight-feature model. It has no OBI, probability calibrator or training
volatility reference. Train a new artifact using recorded Coinbase books;
these historical metrics do not describe the new pipeline.

File: `hgb_btc15m_prod.joblib`. The filename is the deployment path; it is not
a production-performance certification. Status: **research example, not live
validated**. This is a real fitted HistGradientBoostingClassifier, not a stub.

- Training source: 3000 real Coinbase BTC/USD one-minute candles; the final
  reproducible window ends at 2026-10-30 19:55 UTC.
- Labels: SPOT_PROXY, 15-minute candle open/close, not official Kalshi payouts.
- Chronological split: 137 training cycles / 59 held-out cycles;
  1908 training rows / 826 OOS rows. Labels crossing the fit cutoff were purged.
- Test AUC-ROC: 0.8731413994; Brier: 0.1451530940; log loss: 0.4537741068.
- Synthetic-quote OOS backtests after the specified fees were negative:
  OLHC: -$5.17 (52 trades); OHLC: -$4.66 (53 trades).
- Those backtests assumed a normal-CDF midpoint, a two-cent spread and assumed
  intraminute high/low ordering. They are not historical Kalshi fills.
- This release repackages the fitted weights without retraining. Original
  training versions and export versions remain separate in the metadata.

Inputs are the eight v0.2.0 features: delta_usd, remaining_sec, z_score,
vol_15m, vol_60m, vol_ratio, mom_1m, mom_3m.
There is no fitted scaler: preprocessing is identity. The exact sklearn
version is required by the loader. The manifest checksum detects corruption
relative to this bundle, but is not a publisher signature or proof of trust.
Only load joblib files from a source you trust: pickle-based loading executes
Python. A source/weights integrity check is not a guarantee of model quality.

Before treating this as a strategy for capital deployment, obtain appropriate
official outcome/quote data, validate on a longer independent period and
evaluate the actual account, liquidity, fees and operational behavior. This
package makes no profitability claim and has not been tested with live orders.
