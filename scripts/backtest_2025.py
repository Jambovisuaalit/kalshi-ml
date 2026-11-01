"""2025 out-of-sample BTCUSDT spot-proxy study for Kalshi-ML.
2024-12 train, all 2025 test. NEVER historical Kalshi quotes or returns.
"""
import io, json, math, hashlib, zipfile, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import requests
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

OUT=Path("reports/backtest-2025")
OUT.mkdir(parents=True,exist_ok=True)
BASE="https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m"
START=pd.Timestamp("2025-01-01T00:00:00Z")
END=pd.Timestamp("2026-01-01T00:00:00Z")
NAMES=["delta_usd","remaining_sec","z_score","vol_15m","vol_60m","vol_ratio","mom_1m","mom_3m"]

def historical():
    sess=requests.Session()
    frames=[]; manifest=[]
    for year,months in ((2024,[12]),(2025,list(range(1,13)))):
        for month in months:
            file=f"BTCUSDT-1m-{year}-{month:02d}.zip"
            url=f"{BASE}/{file}"
            for attempt in range(4):
                try:
                    blob=sess.get(url,timeout=120)
                    blob.raise_for_status()
                    check=sess.get(url+".CHECKSUM",timeout=30)
                    check.raise_for_status()
                    expected=check.text.split()[0].lower()
                    actual=hashlib.sha256(blob.content).hexdigest()
                    if actual!=expected:raise RuntimeError("checksum mismatch "+file)
                    with zipfile.ZipFile(io.BytesIO(blob.content)) as archive:
                        files=[n for n in archive.namelist() if n.endswith(".csv")]
                        if len(files)!=1: raise RuntimeError("unexpected ZIP contents")
                        data=pd.read_csv(archive.open(files[0]),header=None,usecols=range(6),names=["ts","open","high","low","close","volume"])
                    data["time"]=pd.to_datetime(data.pop("ts"),unit="us" if year>=2025 else "ms",utc=True)
                    frames.append(data)
                    manifest.append({"file":file,"rows":len(data),"sha256":actual})
                    print("DOWNLOADED",file,len(data),flush=True)
                    break
                except (requests.RequestException,zipfile.BadZipFile):
                    if attempt==3: raise
                    time.sleep(2*(attempt+1))
    frame=pd.concat(frames,ignore_index=True).set_index("time").sort_index()
    if frame.index.has_duplicates: raise RuntimeError("duplicate timestamps")
    expected=pd.date_range("2024-12-01T00:00Z","2025-12-31T23:59Z",freq="min")
    if not frame.index.equals(expected):raise RuntimeError("Missing bars / unexpected timestamp units or gaps")
    p=frame[["open","high","low","close"]]
    if not ((p>0).all().all() and (frame.high>=p[["open","low","close"]].max(axis=1)).all() and (frame.low<=p[["open","high","close"]].min(axis=1)).all()):
        raise RuntimeError("bad OHLC data")
    return frame,manifest

frame,manifest=historical()
cl=frame.close.to_numpy(float)
op=frame.open.to_numpy(float)
hi=frame.high.to_numpy(float)
lo=frame.low.to_numpy(float)
times=frame.index
minute=((times.asi8//(60*10**9))%15).astype(int)
ix=np.arange(len(frame))
cycle=ix-minute
strike=op[cycle]
ret=pd.Series(np.log(cl)).diff()
sigma15=ret.rolling(15).std(ddof=1).shift(1).to_numpy()
sigma60=ret.rolling(60).std(ddof=1).shift(1).to_numpy()
prev=np.roll(cl,1);lag1=np.roll(cl,2);lag3=np.roll(cl,4)
prev[:65]=np.nan;lag1[:65]=np.nan;lag3[:65]=np.nan
def inputs(indices,offsets,spot):
    remain=900.-minute[indices]*60.-offsets
    d=spot-strike[indices]
    s15=sigma15[indices];s60=sigma60[indices]
    v15=s15*math.sqrt(15);v60=s60*math.sqrt(60)
    return np.column_stack((d,remain,d/np.maximum(.01,spot*s15*np.sqrt(remain/60.)),
        v15,v60,np.divide(v15,v60,out=np.ones_like(v15),where=v60>0),
        spot/lag1[indices]-1,spot/lag3[indices]-1))

# The 2024-12 labels settle entirely before 2025-01-01. 2026-trained model is not used.
train=ix[(times>=pd.Timestamp("2024-12-01T01:10Z"))&(times<START)&(minute>=1)]
train=train[cycle[train]+14 < np.searchsorted(times.asi8,START.value)]
y=(cl[cycle[train]+14]>strike[train]).astype(int)
notie=cl[cycle[train]+14]!=strike[train]
train=train[notie];y=y[notie]
x=inputs(train,np.zeros(len(train)),prev[train])
valid=np.isfinite(x).all(axis=1)
x=x[valid];y=y[valid]
if len(set(y))!=2: raise RuntimeError("single training class")
model=HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,
    min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=42)
model.fit(pd.DataFrame(x,columns=NAMES),y)
print("TRAIN 2024-12",len(y),"yes rate",y.mean(),flush=True)
test=ix[(times>=START)&(times<END)]
if len(test)!=525600:raise RuntimeError("2025 is not a full 525600-minute year")


# Every decision is at the minute boundary, using ONLY the prior CLOSED bar.
# No OHLC high/low of the active minute enters a feature, quote or exit decision.
# This is still a synthetic spot-proxy market, NOT executable Kalshi history.
CHUNK=45000
predictions=np.empty(len(test),dtype=float)
yes_mid=np.empty(len(test),dtype=np.int16)
for offset in range(0,len(test),CHUNK):
    ids=test[offset:offset+CHUNK]
    features=inputs(ids,np.zeros(len(ids)),prev[ids])
    if not np.isfinite(features).all():raise RuntimeError("nonfinite historical features")
    predictions[offset:offset+len(ids)]=model.predict_proba(pd.DataFrame(features,columns=NAMES))[:,1]
    yes_mid[offset:offset+len(ids)]=np.clip(np.floor(ndtr(features[:,2])*100+.5),0,100).astype(np.int16)
print("PRECOMPUTED CAUSAL QUOTES",len(test),flush=True)

def simulate(tp,sl,delta_stop,spread=2,slippage=0,fee=2,capital_cents=100000):
    """Two ledgers: unconstrained one-contract strategy and funded execution.

    The unconstrained ledger NEVER stops trading due to negative cash; it is
    not a feasible trading account. Funded ledger skips unaffordable entries.
    Settlement happens only at the next cycle boundary; one entry per cycle.
    """
    half=spread//2
    free=capital_cents; funded=capital_cents
    free_peak=capital_cents; funded_peak=capital_cents
    free_dd=0; funded_dd=0
    positions=[None,None]; entered=[-1,-1]
    month_data={}
    trades=[]; signals=0; funded_skipped=0
    def record(ledger,month,pnl,reason,side,entry,exitprice):
        nonlocal free,funded
        if ledger==0:free+=exitprice-(fee if reason!="Settlement" else 0)
        else:funded+=exitprice-(fee if reason!="Settlement" else 0)
        bucket=month_data.setdefault(month,{"unconstrained_pnl_cents":0,"funded_pnl_cents":0,
            "unconstrained_trades":0,"funded_trades":0,"unconstrained_win_cents":0,
            "unconstrained_loss_cents":0,"funded_win_cents":0,"funded_loss_cents":0,
            "unconstrained_max_drawdown_cents":0,"funded_max_drawdown_cents":0})
        prefix="unconstrained" if ledger==0 else "funded"
        bucket[prefix+"_pnl_cents"]+=pnl
        bucket[prefix+"_trades"]+=1
        bucket[prefix+("_win_cents" if pnl>0 else "_loss_cents")]+=abs(pnl)
        if ledger==0:
            trades.append({"timestamp":str(times[k]),"month":month,"side":side,
                "entry_cents":entry,"exit_cents":exitprice,"pnl_cents":pnl,"reason":reason})
    for j,k in enumerate(test):
        month=times[k].strftime("%Y-%m")
        c=int(cycle[k])
        mid=int(yes_mid[j]); yb=max(0,mid-half);ya=min(100,mid+half)
        nb=100-ya;na=100-yb
        delta=prev[k]-strike[k]
        prob=float(predictions[j])
        # Settlement of prior cycle uses its last close; only now observable.
        for ledger in range(2):
            pos=positions[ledger]
            if pos is None:continue
            reason=None
            if pos["cycle"]!=c:
                yes=cl[pos["cycle"]+14]>strike[pos["cycle"]]
                exitprice=100 if yes==(pos["side"]=="YES") else 0
                reason="Settlement"
                pnl=exitprice-pos["entry"]-fee
            else:
                bid=yb if pos["side"]=="YES" else nb
                if delta_stop and ((pos["side"]=="YES" and delta<=0) or
                                   (pos["side"]=="NO" and delta>=0)):
                    reason="Delta-SL"
                elif pos["entry"]-bid>=sl:reason="Hard-SL"
                elif bid>=tp:reason="TP"
                if reason:
                    exitprice=max(0,bid-slippage)
                    pnl=exitprice-pos["entry"]-2*fee
            if reason:
                record(ledger,month,pnl,reason,pos["side"],pos["entry"],exitprice)
                positions[ledger]=None
        remain=900-int(minute[k])*60
        side=None;ask=0
        if 180<=remain<=800 and minute[k]>=1:
            if prob>=.55 and delta>=5:side="YES";ask=ya
            elif prob<=.45 and delta<=-5:side="NO";ask=na
        if side and 45<=ask<=66 and ask+slippage<=66:
            signals+=1
            for ledger in range(2):
                if positions[ledger] is not None or entered[ledger]==c:continue
                fill=ask+slippage
                if ledger==1 and funded<fill+fee:
                    funded_skipped+=1
                    continue
                if ledger==0:free-=fill+fee
                else:funded-=fill+fee
                positions[ledger]={"cycle":c,"side":side,"entry":fill}
                entered[ledger]=c
        for ledger in range(2):
            pos=positions[ledger]
            value=max(0,(yb if pos["side"]=="YES" else nb)-slippage-fee) if pos else 0
            if ledger==0:
                equity=free+value
                free_peak=max(free_peak,equity)
                free_dd=max(free_dd,free_peak-equity)
                current_dd=free_peak-equity
                prefix="unconstrained"
            else:
                equity=funded+value
                funded_peak=max(funded_peak,equity)
                funded_dd=max(funded_dd,funded_peak-equity)
                current_dd=funded_peak-equity
                prefix="funded"
            bucket=month_data.setdefault(month,{"unconstrained_pnl_cents":0,"funded_pnl_cents":0,
                "unconstrained_trades":0,"funded_trades":0,"unconstrained_win_cents":0,
                "unconstrained_loss_cents":0,"funded_win_cents":0,"funded_loss_cents":0,
                "unconstrained_max_drawdown_cents":0,"funded_max_drawdown_cents":0})
            bucket[prefix+"_max_drawdown_cents"]=max(bucket[prefix+"_max_drawdown_cents"],current_dd)
    # Final position settles at end of 2025; no future 2026 bar is used.
    k=int(test[-1])
    for ledger,pos in enumerate(positions):
        if pos:
            yes=cl[pos["cycle"]+14]>strike[pos["cycle"]]
            exitprice=100 if yes==(pos["side"]=="YES") else 0
            pnl=exitprice-pos["entry"]-fee
            record(ledger,"2025-12",pnl,"Settlement",pos["side"],pos["entry"],exitprice)
    if free-capital_cents!=sum(t["pnl_cents"] for t in trades):
        raise RuntimeError("unconstrained PnL ledger mismatch")
    months=[]
    for month,b in sorted(month_data.items()):
        row={"month":month}
        for prefix in ("unconstrained","funded"):
            win=b[prefix+"_win_cents"];loss=b[prefix+"_loss_cents"]
            row[prefix+"_pnl_usd"]=round(b[prefix+"_pnl_cents"]/100,2)
            row[prefix+"_trades"]=b[prefix+"_trades"]
            row[prefix+"_profit_factor"]=round(win/loss,4) if loss else None
            row[prefix+"_max_drawdown_usd"]=round(b[prefix+"_max_drawdown_cents"]/100,2)
        months.append(row)
    win=sum(max(0,t["pnl_cents"]) for t in trades)
    loss=-sum(min(0,t["pnl_cents"]) for t in trades)
    label=f"tp{tp}_sl{sl}_delta{int(delta_stop)}_spread{spread}_slip{slippage}"
    summary={"scenario":label,"tp_cents":tp,"sl_cents":sl,"delta_stop":delta_stop,
        "spread_cents":spread,"slippage_cents":slippage,"fee_per_side_cents":fee,
        "unconstrained_trades":len(trades),
        "unconstrained_pnl_usd":round((free-capital_cents)/100,2),
        "unconstrained_win_rate":round(sum(t["pnl_cents"]>0 for t in trades)/len(trades),5) if trades else None,
        "unconstrained_profit_factor":round(win/loss,4) if loss else None,
        "unconstrained_max_drawdown_usd":round(free_dd/100,2),
        "funded_pnl_usd":round((funded-capital_cents)/100,2),
        "funded_max_drawdown_usd":round(funded_dd/100,2),
        "funded_trades":sum(x["funded_trades"] for x in months),
        "funded_skipped_insufficient_cash":funded_skipped,
        "signals":signals,
        "exits":dict(Counter(t["reason"] for t in trades))}
    return summary,months,trades

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--full",action="store_true",help="48 scenarios; otherwise 2 quick checks")
args=parser.parse_args()
scenarios=[]
monthly=[]
all_trades=[]
grid=[(tp,sl,d,spread,slip) for spread,slip in ((2,0),(4,1))
      for d in (True,False) for tp in (75,80,85,88) for sl in (10,14,20)]
if not args.full:grid=[(88,14,True,2,0),(88,14,False,2,0)]
for idx,(tp,sl,d,spread,slip) in enumerate(grid):
    result,months,trades=simulate(tp,sl,d,spread,slip)
    scenarios.append(result)
    monthly.extend([{"scenario":result["scenario"],**x} for x in months])
    all_trades.extend([{"scenario":result["scenario"],**x} for x in trades])
    print("SCENARIO_RESULT",idx+1,len(grid),json.dumps(result),flush=True)
pd.DataFrame(scenarios).to_csv(OUT/"scenario_summary.csv",index=False)
pd.DataFrame(monthly).to_csv(OUT/"monthly_metrics.csv",index=False)
pd.DataFrame(all_trades).to_csv(OUT/"unconstrained_trades.csv",index=False)
meta={"source":"Binance BTCUSDT spot 1m 2024-12 and all 2025",
    "train":"2024-12","test":"2025-01-01 through 2025-12-31",
    "type":"SPOT_PROXY, synthetic normal-CDF quotes, NOT actual Kalshi book/fills/settlements",
    "causality":"At each minute boundary use only previous CLOSED minute; no intraminute high/low",
    "model":"2024-12 trained HistGradientBoostingClassifier frozen for 2025",
    "capital":"Unconstrained strategy ledger may go negative; funded ledger skips entries without cash",
    "drawdown":"Mark-to-synthetic-bid global peak-to-trough; monthly values are global peak-to-trough observed in each month",
    "data_manifest":manifest,"train_rows":len(y),"scenarios":scenarios}
(OUT/"summary.json").write_text(json.dumps(meta,indent=2))
print("FINISHED_2025_BACKTEST",len(scenarios),flush=True)
