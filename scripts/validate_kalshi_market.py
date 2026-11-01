#!/usr/bin/env python3
"""Offline, read-only Kalshi quote/settlement and ML probability validation.

Input is an independently sourced, timestamped snapshot CSV. Never reconstruct
Kalshi prices from BTC spot or substitute proxy settlement for official result.
No trading, credentials, network calls, or model fitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ("ticker","observed_at","settled_at","yes_bid_cents",
            "yes_ask_cents","no_bid_cents","no_ask_cents",
            "model_p_yes","settlement_yes")
COSTS = ("fee_cents_per_contract","slippage_cents_per_contract")

def load(path: Path) -> pd.DataFrame:
    frame=pd.read_csv(path)
    missing=set(REQUIRED)-set(frame.columns)
    if missing:raise ValueError("missing required columns: "+", ".join(sorted(missing)))
    if frame.empty:raise ValueError("no observations")
    if frame[list(REQUIRED)].isna().any().any():raise ValueError("missing required values")
    if frame.ticker.astype(str).str.strip().eq("").any():raise ValueError("blank ticker")
    for field in ("observed_at","settled_at"):
        frame[field]=pd.to_datetime(frame[field],utc=True,errors="raise")
    if (frame.observed_at>=frame.settled_at).any():
        raise ValueError("look-ahead: observed_at must precede settled_at")
    numeric=["yes_bid_cents","yes_ask_cents","no_bid_cents","no_ask_cents","model_p_yes","settlement_yes"]
    for col in numeric:
        frame[col]=pd.to_numeric(frame[col],errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("non-finite numeric values")
    for col in ("yes_bid_cents","yes_ask_cents","no_bid_cents","no_ask_cents"):
        if not frame[col].between(0,100).all():raise ValueError("quote out of 0..100: "+col)
    if ((frame.yes_bid_cents>frame.yes_ask_cents)|
        (frame.no_bid_cents>frame.no_ask_cents)).any():
        raise ValueError("crossed bid/ask")
    if not frame.model_p_yes.between(0,1).all():raise ValueError("invalid model probability")
    if not frame.settlement_yes.isin([0,1]).all():raise ValueError("settlement_yes must be 0 or 1")
    if frame.groupby("ticker").settlement_yes.nunique().gt(1).any():
        raise ValueError("contradictory official outcomes for same ticker")
    if frame.groupby("ticker").settled_at.nunique().gt(1).any():
        raise ValueError("contradictory settlement times for same ticker")
    if frame.duplicated(["ticker","observed_at"]).any():
        raise ValueError("duplicate market snapshots")
    # A genuine executable bid/ask must not imply a crossed YES/NO pair.
    if ((frame.yes_bid_cents+frame.no_bid_cents>100)|
        (frame.yes_ask_cents+frame.no_ask_cents<100)).any():
        raise ValueError("inconsistent complementary YES/NO quotes")
    return frame.sort_values(["ticker","observed_at"]).reset_index(drop=True)

def evaluate(frame:pd.DataFrame,fee:float,slippage:float,min_edge:float,bin_count:int)->tuple[dict,pd.DataFrame,pd.DataFrame]:
    if fee<0 or slippage<0:raise ValueError("fees and slippage must be nonnegative")
    if min_edge<0:raise ValueError("min_edge must be nonnegative")
    if bin_count<2:raise ValueError("bins must be >= 2")
    # First quote per contract is the independent calibration unit. Multiple
    # snapshots of one contract are not independent labels.
    first=frame.drop_duplicates("ticker",keep="first").copy()
    p=first.model_p_yes.to_numpy(float);y=first.settlement_yes.to_numpy(int)
    eps=1e-12
    brier=float(np.mean((p-y)**2))
    ll=float(np.mean(-y*np.log(np.clip(p,eps,1-eps))-(1-y)*np.log(np.clip(1-p,eps,1-eps))))
    first["calibration_bin"]=np.minimum((first.model_p_yes*bin_count).astype(int),bin_count-1)
    bins=first.groupby("calibration_bin",observed=True).agg(
        contracts=("ticker","size"),mean_prediction=("model_p_yes","mean"),
        actual_yes_rate=("settlement_yes","mean")).reset_index()
    bins["absolute_error"]=(bins.mean_prediction-bins.actual_yes_rate).abs()
    ece=float((bins.contracts*bins.absolute_error).sum()/len(first))
    # At most one simulated decision per ticker; avoid cherry-picking the
    # most favorable snapshot after seeing the settlement.
    # Choose earliest snapshot meeting the fixed edge threshold.
    q=frame.copy()
    q["yes_entry_cost_cents"]=q.yes_ask_cents+fee+slippage
    q["no_entry_cost_cents"]=q.no_ask_cents+fee+slippage
    q["yes_ev_cents"]=100*q.model_p_yes-q.yes_entry_cost_cents
    q["no_ev_cents"]=100*(1-q.model_p_yes)-q.no_entry_cost_cents
    q["best_side"]=np.where(q.yes_ev_cents>=q.no_ev_cents,"YES","NO")
    q["best_ev_cents"]=np.maximum(q.yes_ev_cents,q.no_ev_cents)
    eligible=q[(q.best_ev_cents>=min_edge)&
               (((q.best_side=="YES")&(q.yes_entry_cost_cents<=100))|
                ((q.best_side=="NO")&(q.no_entry_cost_cents<=100)))].copy()
    selected=eligible.drop_duplicates("ticker",keep="first").copy()
    selected["realized_pnl_cents"]=np.where(
        selected.best_side=="YES",
        100*selected.settlement_yes-selected.yes_entry_cost_cents,
        100*(1-selected.settlement_yes)-selected.no_entry_cost_cents)
    selected["profitable"]=selected.realized_pnl_cents>0
    gross_wins=float(selected.loc[selected.realized_pnl_cents>0,"realized_pnl_cents"].sum())
    gross_losses=float(-selected.loc[selected.realized_pnl_cents<0,"realized_pnl_cents"].sum())
    report={
        "status":"RESEARCH_ONLY_NOT_EXECUTED",
        "source":"user-provided official Kalshi snapshots and settlements (not independently verified by this script)",
        "distinct_contracts":int(len(first)),
        "quote_snapshots":int(len(frame)),
        "brier_model":round(brier,6),
        "log_loss_model":round(ll,6),
        "expected_calibration_error":round(ece,6),
        "calibration_bins":bin_count,
        "fee_cents_per_contract":fee,
        "slippage_cents_per_contract":slippage,
        "minimum_predicted_edge_cents":min_edge,
        "eligible_contracts":int(len(selected)),
        "coverage":round(len(selected)/len(first),6),
        "mean_predicted_ev_cents":round(float(selected.best_ev_cents.mean()),4) if len(selected) else None,
        "mean_realized_pnl_cents":round(float(selected.realized_pnl_cents.mean()),4) if len(selected) else None,
        "total_realized_pnl_usd":round(float(selected.realized_pnl_cents.sum())/100,2),
        "win_rate":round(float(selected.profitable.mean()),6) if len(selected) else None,
        "profit_factor":round(gross_wins/gross_losses,6) if gross_losses else None,
        "notes":["One earliest eligible snapshot per contract; no intra-contract trade path.",
                 "Quote availability is not evidence of executable fill or depth.",
                 "Official market outcome and market prices must be verified at source.",
                 "The CSV must contain probabilities recorded before settlement, never re-fitted on these labels.",
                 "Fees are supplied assumptions, not exchange-reconciled actual fees.",
                 "All contracts in this dataset may share market conditions; confidence intervals require time-block evaluation."]}
    return report,bins,selected

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("reports/kalshi_validation"))
    ap.add_argument("--fee-cents",type=float,default=2.)
    ap.add_argument("--slippage-cents",type=float,default=1.)
    ap.add_argument("--min-edge-cents",type=float,default=3.)
    ap.add_argument("--bins",type=int,default=10)
    args=ap.parse_args()
    frame=load(args.input)
    report,bins,selected=evaluate(frame,args.fee_cents,args.slippage_cents,args.min_edge_cents,args.bins)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    bins.to_csv(args.output/"calibration.csv",index=False)
    selected.to_csv(args.output/"simulated_decisions.csv",index=False)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
