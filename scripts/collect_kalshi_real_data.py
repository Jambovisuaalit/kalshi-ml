#!/usr/bin/env python3
"""Read-only public Kalshi KXBTC15M data acquisition.

Archive: actual settled markets, one-minute bid/ask candles and trade prints.
Live: current market bid/ask snapshots (no retrospective model probabilities).
Raw payloads retained. No orders, API keys, or fabricated ML predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import requests

BASE="https://external-api.kalshi.com/trade-api/v2"
SERIES="KXBTC15M"

def utc_now():
    return datetime.now(timezone.utc).isoformat()

class Client:
    def __init__(self,delay=0.4):
        self.session=requests.Session()
        self.delay=delay
        self.last=0.0
    def get(self,path,params=None):
        for attempt in range(6):
            elapsed=time.monotonic()-self.last
            if elapsed<self.delay:time.sleep(self.delay-elapsed)
            self.last=time.monotonic()
            try:
                response=self.session.get(BASE+path,params=params,timeout=25,
                    headers={"User-Agent":"kalshi-ml-research/1.0"})
                if response.status_code in (429,500,502,503,504):
                    if attempt==5:response.raise_for_status()
                    time.sleep(min(30,2**attempt));continue
                response.raise_for_status()
                return {"received_at":utc_now(),"request_url":response.url,
                        "payload":response.json()}
            except (requests.Timeout,requests.ConnectionError):
                if attempt==5:raise
                time.sleep(min(30,2**attempt))
        raise RuntimeError("retry budget exhausted")
    def pages(self,path,params,key,max_pages):
        cursor=None;seen=set()
        for _ in range(max_pages):
            p=dict(params)
            if cursor:p["cursor"]=cursor
            envelope=self.get(path,p)
            records=envelope["payload"].get(key)
            if not isinstance(records,list):
                raise RuntimeError("unexpected response shape "+path)
            yield envelope,records
            cursor=envelope["payload"].get("cursor")
            if not cursor:break
            if cursor in seen:raise RuntimeError("repeated cursor")
            seen.add(cursor)
        else:raise RuntimeError("pagination truncated at max-pages")

def append_jsonl(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        for item in items:f.write(json.dumps(item,separators=(",",":"),ensure_ascii=False)+"\n")

def collect(args):
    api=Client(args.delay)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    manifest={"started_at":utc_now(),"series":SERIES,"mode":args.mode,
              "api_base":BASE,"max_markets":args.max_markets}
    cutoff=api.get("/historical/cutoff")
    append_jsonl(out/"cutoff.jsonl",[cutoff])
    manifest["cutoff"]=cutoff["payload"]
    if args.mode=="archive":
        # Historical list is paginated. Never silently assume a partial walk is complete.
        seen=set();count=0
        for envelope,markets in api.pages("/historical/markets",
                {"series_ticker":SERIES,"limit":1000}, "markets",args.max_pages):
            append_jsonl(out/"raw_historical_market_pages.jsonl",[envelope])
            for market in markets:
                ticker=market.get("ticker","")
                if not ticker.startswith(SERIES+"-") or ticker in seen:continue
                seen.add(ticker)
                if market.get("result") not in ("yes","no"):
                    continue
                if args.start_date and str(market.get("close_time",""))[:10]<args.start_date:continue
                if args.end_date and str(market.get("close_time",""))[:10]>args.end_date:continue
                append_jsonl(out/"settled_markets.jsonl",[market])
                # Each market is short-lived. 1-minute candles provide recorded
                # bid/ask snapshots, NOT historical depth or guaranteed fills.
                import datetime as dt
                def unix(value):
                    return int(dt.datetime.fromisoformat(value.replace("Z","+00:00")).timestamp())
                open_time=market.get("open_time") or market.get("created_time")
                close_time=market.get("close_time")
                if open_time and close_time:
                    candles=api.get("/historical/markets/"+quote(ticker,safe="")+"/candlesticks",
                        {"start_ts":unix(open_time)-60,"end_ts":unix(close_time)+60,
                         "period_interval":1})
                    append_jsonl(out/"raw_candles.jsonl",[candles])
                # All public trades, cursor pagination.
                for trades_envelope,_ in api.pages("/historical/trades",
                        {"ticker":ticker,"limit":1000},"trades",args.max_pages):
                    append_jsonl(out/"raw_trades.jsonl",[trades_envelope])
                count+=1
                print("ARCHIVED",count,ticker,flush=True)
                if count>=args.max_markets:break
            if count>=args.max_markets:break
        manifest["markets_saved"]=count
    else:
        count=0
        for envelope,markets in api.pages("/markets",
                {"series_ticker":SERIES,"status":"open","limit":100},
                "markets",args.max_pages):
            append_jsonl(out/"raw_live_market_pages.jsonl",[envelope])
            for market in markets:
                if not market.get("ticker","").startswith(SERIES+"-"):continue
                record={"received_at":envelope["received_at"],
                    "ticker":market["ticker"],"market":market,
                    "model_p_yes":None,
                    "model_status":"NOT_OBSERVED_DO_NOT_BACKFILL"}
                append_jsonl(out/"live_market_snapshots.jsonl",[record])
                count+=1
        manifest["snapshots_saved"]=count
    manifest["finished_at"]=utc_now()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("RESULT",json.dumps(manifest),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("archive","live"),default="archive")
    parser.add_argument("--output",type=Path,default=Path("reports/kalshi_real_data"))
    parser.add_argument("--start-date",help="UTC YYYY-MM-DD")
    parser.add_argument("--end-date",help="UTC YYYY-MM-DD")
    parser.add_argument("--max-markets",type=int,default=100)
    parser.add_argument("--max-pages",type=int,default=1000)
    parser.add_argument("--delay",type=float,default=.4)
    args=parser.parse_args()
    if args.max_markets<1 or args.max_pages<1 or args.delay<0:
        parser.error("max-markets/max-pages must be positive and delay nonnegative")
    collect(args)

if __name__=="__main__":main()
