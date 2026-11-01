# Kalshi KXBTC15M — actual-market signal validation

**Status: instrumentation only. No historical Kalshi quotes or verified official settlements are included. No profitability claim.**

## Required evidence

Collect **immutable timestamped** Kalshi KXBTC15M YES/NO top-of-book bid/ask quotes from the actual exchange, official contract settlement and settlement timestamp, and the ML P(YES) **as recorded before the contract settled**. Preserve raw response payloads, source URL, collection/receipt timestamps, ticker and contract rules in a separate audit store. Ensure price/strike reference uses Kalshi's actual settlement methodology; do not substitute Binance/Coinbase proxy outcomes.

A model trained on future data must not be backfilled into earlier snapshots. For historical evaluation, regenerate predictions with a strictly time-split model whose training ends before the evaluation period; retain training cutoff and model hash. **Do not use the existing 2026-trained production model to score 2025 retrospectively.**

## Snapshot CSV schema

Required columns:

| Column | Meaning |
|---|---|
| ticker | Actual Kalshi market ticker |
| observed_at | UTC quote and prediction observation time (ISO 8601) |
| settled_at | UTC official settlement time (ISO 8601) |
| yes_bid_cents, yes_ask_cents | Actual executable quote top-of-book, cents |
| no_bid_cents, no_ask_cents | Actual executable quote top-of-book, cents |
| model_p_yes | Frozen pre-settlement model P(YES), 0..1 |
| settlement_yes | Official YES result, 0 or 1 |

Use one row per observed snapshot. Do not add rows from synthetic spot quotes or guessed fills. Rows with missing or inconsistent evidence are rejected, not imputed. Market depth and fee tier are **not** inferred.

## Offline evaluation

```bash
python scripts/validate_kalshi_market.py --input /path/to/verified_kalshi_snapshots.csv \
  --output reports/kalshi_validation --fee-cents 2 --slippage-cents 1 --min-edge-cents 3
python -m unittest tests/test_kalshi_market_validation.py
```

Outputs: `report.json`, `calibration.csv`, `simulated_decisions.csv`.

Brier score, log loss, and equal-width reliability bins are calculated on the **earliest snapshot per distinct contract**, avoiding repeated settlement labels. The executable *hypothetical* entry decision chooses the **earliest** quote crossing the prespecified net EV threshold per ticker. P(YES) * 100 - YES ask - assumed fees - assumed slippage, or (1-P(YES)) * 100 - NO ask - assumed fees - assumed slippage. Settlement payout is 100 cents if correct, 0 otherwise. One contract, no exits before settlement. This deliberately tests the signal and entry price independently of TP/SL tuning.

**This is NOT evidence of fills**: quote visibility is not depth, priority, or guaranteed execution. No live trading is performed.

## Acceptance gates

1. Evidence: official Kalshi quotes and settlements matched by ticker and timestamp; full source provenance, no future leakage.
2. Model: frozen before evaluation, model version/hash and cutoff logged; probability calibration measured.
3. Economics: prespecified EV threshold and realistic fees, depth and slippage; net mean realized PnL > 0 on a **later untouched holdout**; sufficient sample and time-block bootstrap confidence interval excluding zero.
4. Operations: no live orders until execution reconciliation, venue restrictions and failure modes are independently verified.

If official quote and outcome history is unavailable, collect prospective paper-trading snapshots; do not label spot proxies as actual Kalshi data.
