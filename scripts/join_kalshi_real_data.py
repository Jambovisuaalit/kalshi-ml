#!/usr/bin/env python3
"""Join Kalshi historical markets, minute candles and trades into an audit CSV.

No synthetic quotes, model probabilities, trade fills, or outcome inference.
Minute candles are interval aggregates, NOT executable point-in-time order books.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
import pandas as pd

def read_jsonl(path):
    if not path.is_file():raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try:yield json.loads(line)
                except json.JSONDecodeError as e:raise ValueError(f"{path}:{n}: {e}") from e

def timestamp(value):
    if value is None or value=="":return pd.NaT
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        unit="ms" if abs(value)>1e11 else "s"
        return pd.to_datetime(value,unit=unit,utc=True,errors="coerce")
    return pd.to_datetime(value,utc=True,errors="coerce")

def cents(value):
    if isinstance(value,dict):
        # Kalshi candle price fields are nested OHLC objects.
        value=value.get("close")
    if value is None or isinstance(value,bool):return None
    try:
        number=float(value)
        return number if math.isfinite(number) and 0<=number<=100 else None
    except (ValueError,TypeError):return None

def candle_ticker(envelope):
    payload=envelope.get("payload",{})
    ticker=payload.get("ticker")
    if ticker:return ticker
    url=envelope.get("request_url","")
    parts=urlparse(url).path.split("/")
    try:return unquote(parts[parts.index("markets")+1])
    except (ValueError,IndexError):return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,default=Path("reports/kalshi_real_data"))
    ap.add_argument("--output",type=Path,default=None)
    args=ap.parse_args()
    root=args.input;output=args.output or root/"kalshi_joined_audit.csv"
    markets={}
    issues=Counter()
    for m in read_jsonl(root/"settled_markets.jsonl"):
        t=m.get("ticker")
        if not t:
            issues["market_without_ticker"]+=1
            continue
        if t in markets:
            if m!=markets[t]:issues["conflicting_market_duplicates"]+=1
            else:issues["duplicate_markets"]+=1
            continue
        markets[t]=m
    trade_by_minute=defaultdict(lambda:{"count":0,"volume":0.0})
    unique_trades=set()
    for envelope in read_jsonl(root/"raw_trades.jsonl"):
        for tr in envelope.get("payload",{}).get("trades",[]):
            t=tr.get("ticker")
            ts=timestamp(tr.get("created_time") or tr.get("created_at") or tr.get("ts"))
            if not t or pd.isna(ts):
                issues["trade_missing_ticker_or_timestamp"]+=1
                continue
            trade_id=tr.get("trade_id") or tr.get("id")
            if trade_id:
                key=(t,str(trade_id))
                if key in unique_trades:issues["duplicate_trades"]+=1;continue
                unique_trades.add(key)
            bucket=trade_by_minute[(t,ts.floor("min"))]
            bucket["count"]+=1
            try:bucket["volume"]+=float(tr.get("count_fp") or tr.get("count") or 0)
            except (TypeError,ValueError):issues["invalid_trade_volume"]+=1
    rows=[];candles_per_market=Counter();seen_candles=set()
    for envelope in read_jsonl(root/"raw_candles.jsonl"):
        t=candle_ticker(envelope)
        if not t or t not in markets:
            issues["unmatched_candle_response"]+=1;continue
        m=markets[t]
        for c in envelope.get("payload",{}).get("candlesticks",[]):
            ts=timestamp(c.get("end_period_ts") or c.get("start_period_ts") or c.get("timestamp"))
            if pd.isna(ts):issues["candle_missing_timestamp"]+=1;continue
            key=(t,ts)
            if key in seen_candles:issues["duplicate_candles"]+=1;continue
            seen_candles.add(key)
            close_time=timestamp(m.get("close_time"))
            open_time=timestamp(m.get("open_time") or m.get("created_time"))
            # end_period_ts is an interval label; never treat it as an
            # independently executable snapshot at this timestamp.
            bid=cents(c.get("yes_bid"));ask=cents(c.get("yes_ask"))
            # Do not derive the NO side from YES quotes: this is a separate
            # historical observation requirement.
            no_bid=cents(c.get("no_bid"));no_ask=cents(c.get("no_ask"))
            valid_quotes=(bid is not None and ask is not None and bid<=ask)
            if bid is not None and ask is not None and bid>ask:
                issues["crossed_yes_candles"]+=1
            if not valid_quotes:issues["missing_or_invalid_yes_quotes"]+=1
            if no_bid is None or no_ask is None:
                issues["missing_no_quotes"]+=1
            if pd.notna(close_time) and ts>close_time:
                issues["candle_after_close"]+=1
            if pd.notna(open_time) and ts<open_time-pd.Timedelta(minutes=1):
                issues["candle_before_open"]+=1
            tr=trade_by_minute.get((t,ts.floor("min")),{"count":0,"volume":0.0})
            rows.append({
                "ticker":t,"candle_time_utc":ts.isoformat(),
                "market_open_utc":open_time.isoformat() if pd.notna(open_time) else "",
                "market_close_utc":close_time.isoformat() if pd.notna(close_time) else "",
                "official_result":m.get("result"),
                "yes_bid_close_cents":bid,"yes_ask_close_cents":ask,
                "no_bid_close_cents":no_bid,"no_ask_close_cents":no_ask,
                "yes_spread_cents":round(ask-bid,4) if valid_quotes else None,
                "trade_count_same_minute":tr["count"],
                "trade_volume_same_minute":tr["volume"],
                "model_p_yes":None,
                "model_status":"MISSING_HISTORICAL_PRE_SETTLEMENT_INFERENCE",
                "quote_type":"HISTORICAL_MINUTE_CANDLE_NOT_EXECUTABLE_SNAPSHOT",
                "quote_valid":valid_quotes,
            })
            candles_per_market[t]+=1
    columns=["ticker","candle_time_utc","market_open_utc","market_close_utc",
        "official_result","yes_bid_close_cents","yes_ask_close_cents",
        "no_bid_close_cents","no_ask_close_cents","yes_spread_cents",
        "trade_count_same_minute","trade_volume_same_minute","model_p_yes",
        "model_status","quote_type","quote_valid"]
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=columns).sort_values(
        ["ticker","candle_time_utc"]).to_csv(output,index=False)
    spreads=[r["yes_spread_cents"] for r in rows if r["yes_spread_cents"] is not None]
    report={
        "status":"DATA_AUDIT_ONLY_NO_MODEL_VALIDATION",
        "markets":len(markets),"markets_with_candles":len(candles_per_market),
        "markets_without_candles":sorted(set(markets)-set(candles_per_market)),
        "candle_rows":len(rows),"distinct_trade_ids":len(unique_trades),
        "trade_observations_joined_by_minute":sum(r["trade_count_same_minute"] for r in rows),
        "result_counts":dict(Counter(str(m.get("result")) for m in markets.values())),
        "valid_yes_quote_candles":sum(r["quote_valid"] for r in rows),
        "yes_spread_p50_cents":float(pd.Series(spreads).median()) if spreads else None,
        "yes_spread_p90_cents":float(pd.Series(spreads).quantile(.9)) if spreads else None,
        "model_predictions_available":0,
        "issues":dict(issues),
        "limitations":[
            "Candle OHLC closes are interval aggregates, not point-in-time executable bids/asks.",
            "Trades are grouped by ticker and minute; candle time label may refer to interval end.",
            "NO quotes are never imputed from YES prices.",
            "Model predictions cannot be reconstructed without time-safe historical model evidence.",
            "No executable EV, Brier score, or realized model PnL can be claimed from this report."]}
    report_path=output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+chr(10))
    print(json.dumps({"csv":str(output),"audit":str(report_path),**report},indent=2,ensure_ascii=False))
if __name__=="__main__":main()
