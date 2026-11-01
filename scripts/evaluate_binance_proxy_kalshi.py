#!/usr/bin/env python3
"""Score pre-trained Binance spot-proxy P(YES) versus official Kalshi outcomes.

Historical Kalshi candle asks are aggregates, NOT executable fills.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def score(frame):
    if frame.empty:return {"contracts":0,"brier":None,"log_loss":None,"ece_5bin":None}
    p=frame.p_yes.to_numpy(float);y=(frame.official_result=="yes").to_numpy(float)
    q=np.clip(p,1e-12,1-1e-12);b=np.minimum(np.floor(p*5).astype(int),4)
    ece=sum((b==j).sum()*abs(p[b==j].mean()-y[b==j].mean())
            for j in range(5) if (b==j).any())/len(frame)
    return {"contracts":len(frame),"brier":float(np.mean((p-y)**2)),
       "log_loss":float(np.mean(-y*np.log(q)-(1-y)*np.log(1-q))),
       "ece_5bin":float(ece),"baseline_brier_0_5":.25}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit",type=Path,default=Path("reports/kalshi_real_data/kalshi_joined_audit.csv"))
    ap.add_argument("--predictions",type=Path,default=Path("reports/kalshi_real_data/binance_proxy_oos.csv"))
    ap.add_argument("--output",type=Path,default=Path("reports/kalshi_real_data/binance_proxy_validation"))
    ap.add_argument("--fee-cents",type=float,default=2.)
    ap.add_argument("--slippage-cents",type=float,default=1.)
    ap.add_argument("--min-edge-cents",type=float,default=3.)
    args=ap.parse_args()
    if min(args.fee_cents,args.slippage_cents,args.min_edge_cents)<0:
        ap.error("Costs and edge must be nonnegative")
    audit=pd.read_csv(args.audit);pred=pd.read_csv(args.predictions)
    for field in ("prediction_at","trained_through","p_yes","model_sha256","source"):
        if field not in pred:raise ValueError("Prediction field missing: "+field)
    if audit.empty or pred.empty:raise ValueError("Empty input")
    if not pred.source.eq("BINANCE_SPOT_PROXY_NOT_KALSHI").all():
        raise ValueError("Not a verified Binance proxy prediction export")
    if pred.model_sha256.nunique()!=1 or not pred.model_sha256.astype(str).str.fullmatch("[0-9a-fA-F]{64}").all():
        raise ValueError("Invalid or inconsistent model identity")
    for field in ("prediction_at","trained_through"):
        pred[field]=pd.to_datetime(pred[field],utc=True,errors="raise")
    for field in ("candle_time_utc","market_close_utc"):
        audit[field]=pd.to_datetime(audit[field],utc=True,errors="raise")
    if pred.prediction_at.duplicated().any() or audit.duplicated(["ticker","candle_time_utc"]).any():
        raise ValueError("Duplicate event times")
    if (pred.trained_through>=pred.prediction_at).any():
        raise ValueError("LOOKAHEAD: training cutoff after prediction")
    if pred.p_yes.isna().any() or not np.isfinite(pred.p_yes.to_numpy(float)).all() or not pred.p_yes.between(0,1).all():
        raise ValueError("Invalid model probabilities")
    if not audit.official_result.isin(["yes","no"]).all():
        raise ValueError("Invalid official outcomes")
    joined=pd.merge_asof(audit.sort_values("candle_time_utc"),pred.sort_values("prediction_at"),
       left_on="candle_time_utc",right_on="prediction_at",
       direction="backward",allow_exact_matches=False)
    # Never treat candles labelled at or after market close as entries.
    valid=(joined.candle_time_utc<joined.market_close_utc) & joined.p_yes.notna() & joined.yes_ask_close_cents.notna()
    first=joined.loc[valid].sort_values(["ticker","candle_time_utc"]).drop_duplicates("ticker",keep="first").copy()
    if first.empty:raise ValueError("No overlapping, pre-close predictions")
    first["settlement_yes"]=(first.official_result=="yes").astype(int)
    first["yes_ev_proxy_cents"]=100*first.p_yes-first.yes_ask_close_cents-args.fee_cents-args.slippage_cents
    first["yes_signal"]=first.yes_ev_proxy_cents>=args.min_edge_cents
    first["yes_pnl_proxy_cents"]=np.where(first.yes_signal,
       first.settlement_yes*100-first.yes_ask_close_cents-args.fee_cents-args.slippage_cents,np.nan)
    first=first.sort_values(["market_close_utc","ticker"]).reset_index(drop=True)
    first["slice"]="development"
    first.loc[int(np.floor(len(first)*.7)):,"slice"]="later_diagnostic"
    report={"status":"RESEARCH_ONLY_NOT_EXECUTABLE",
      "markets_in_audit":int(audit.ticker.nunique()),"matched_contracts":len(first),
      "model_sha256":str(pred.model_sha256.iloc[0]),
      "source":"Binance BTCUSDT spot 1m model features and proxy labels; Kalshi actual contract outcomes",
      "quote_warning":"YES ask historical 1m candle close is NOT an executable offer",
      "availability_warning":"Binance publication lag +1 second is an unverified assumption",
      "holdout_warning":"Later diagnostic slice is NOT an untouched external future holdout",
      "fee_cents_assumed":args.fee_cents,"slippage_cents_assumed":args.slippage_cents,
      "min_yes_edge_cents":args.min_edge_cents,"slices":{}}
    for name,group in first.groupby("slice"):
        bets=group[group.yes_signal]
        report["slices"][name]={**score(group),"yes_signals":len(bets),
           "total_yes_pnl_proxy_cents":float(bets.yes_pnl_proxy_cents.sum())}
    args.output.mkdir(parents=True,exist_ok=True)
    first.to_csv(args.output/"first_per_contract.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+chr(10))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
