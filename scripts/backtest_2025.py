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

def simulate(path,spread,slippage,fee=2):
    cash=100000;peak=cash;drawdown=0;pos=None;trades=[];entered=set()
    signals=0;observations=0
    for offset in range(0,len(test),45000):
        ids=test[offset:offset+45000]
        size=len(ids)
        spot=np.stack((prev[ids],lo[ids] if path=="OLHC" else hi[ids],
            hi[ids] if path=="OLHC" else lo[ids]),axis=1).reshape(-1)
        which=np.repeat(ids,3)
        seconds=np.tile(np.array([0.,20.,40.]),size)
        feats=inputs(which,seconds,spot)
        if not np.isfinite(feats).all():raise RuntimeError("nonfinite prediction inputs")
        prob=model.predict_proba(pd.DataFrame(feats,columns=NAMES))[:,1]
        mid=np.clip(np.floor(ndtr(feats[:,2])*100+.5).astype(int),spread//2,100-spread//2)
        yb=mid-spread//2;ya=mid+spread//2
        nb=100-ya;na=100-yb
        for j,k in enumerate(which):
            c=int(cycle[k]);delta=spot[j]-strike[k]
            ts=str(times[k]+pd.Timedelta(seconds=float(seconds[j])))
            if pos is not None and c!=pos["cycle"]:
                # Proxy outcome is known only as previous cycle ends.
                yes=cl[pos["cycle"]+14]>strike[pos["cycle"]]
                payout=100 if yes==(pos["side"]=="YES") else 0
                pnl=payout-pos["entry"]-fee
                cash+=payout
                trades.append(dict(time=ts,side=pos["side"],pnl_cents=pnl,
                    entry_cents=pos["entry"],exit_cents=payout,reason="Settlement"))
                pos=None
            if pos:
                bid=int(yb[j] if pos["side"]=="YES" else nb[j])
                reason=None
                if (pos["side"]=="YES" and delta<=0) or (pos["side"]=="NO" and delta>=0):
                    reason="Delta-SL"
                elif pos["entry"]-bid>=14:reason="Hard-SL"
                elif bid>=88:reason="TP"
                if reason:
                    exitprice=max(0,bid-slippage)
                    pnl=exitprice-pos["entry"]-2*fee
                    cash+=exitprice-fee
                    trades.append(dict(time=ts,side=pos["side"],pnl_cents=pnl,
                        entry_cents=pos["entry"],exit_cents=exitprice,reason=reason))
                    pos=None
            remain=900-minute[k]*60-seconds[j]
            if pos is None and c not in entered and minute[k]>=1 and 180<=remain<=800:
                buy_side=None;ask=0
                if prob[j]>=.55 and delta>=5:buy_side="YES";ask=int(ya[j])
                elif prob[j]<=.45 and delta<=-5:buy_side="NO";ask=int(na[j])
                if buy_side and 45<=ask<=66:
                    signals+=1
                    fill=ask+slippage
                    if fill<=66 and cash>=fill+fee*2:
                        cash-=fill+fee
                        entered.add(c)
                        pos=dict(cycle=c,side=buy_side,entry=fill)
            equity=cash+((max(0,int(yb[j] if pos["side"]=="YES" else nb[j])-slippage)-fee) if pos else 0)
            peak=max(peak,equity)
            drawdown=max(drawdown,peak-equity)
            observations+=1
        print("PROGRESS",path,spread,slippage,round((offset+size)/len(test)*100,1),flush=True)
    if pos:
        yes=cl[pos["cycle"]+14]>strike[pos["cycle"]]
        payout=100 if yes==(pos["side"]=="YES") else 0
        pnl=payout-pos["entry"]-fee
        cash+=payout
        trades.append(dict(time=str(END),side=pos["side"],pnl_cents=pnl,
            entry_cents=pos["entry"],exit_cents=payout,reason="Settlement"))
    if cash-100000 != sum(t["pnl_cents"] for t in trades):
        raise RuntimeError("PnL ledger unreconciled")
    win=sum(max(0,t["pnl_cents"]) for t in trades)
    loss=-sum(min(0,t["pnl_cents"]) for t in trades)
    rec=dict(scenario=f"{path}_spread{spread}_slippage{slippage}_fee{fee}",
        trades=len(trades),wins=sum(t["pnl_cents"]>0 for t in trades),
        win_rate=round(sum(t["pnl_cents"]>0 for t in trades)/len(trades),6) if trades else None,
        pnl_usd=(cash-100000)/100,
        profit_factor=round(win/loss,4) if loss else None,
        max_drawdown_usd=drawdown/100,
        fee_usd=(sum(1 if t["reason"]=="Settlement" else 2 for t in trades)*fee)/100,
        exits=dict(Counter(t["reason"] for t in trades)),signals=signals,observations=observations)
    pd.DataFrame(trades).to_csv(OUT/(rec["scenario"]+".csv"),index=False)
    print("SCENARIO_RESULT",json.dumps(rec),flush=True)
    return rec
scenarios=[simulate("OLHC",2,0),simulate("OHLC",2,0),
    simulate("OLHC",4,1),simulate("OHLC",4,1)]
meta={"source":"Binance BTCUSDT spot 1m, December 2024 and all 2025",
    "train":"2024-12", "test":"2025-01-01 through 2025-12-31",
    "type":"SPOT_PROXY synthetic normal-CDF quotes; not observed Kalshi order book, fees, fills or settlement",
    "model":"new 2024-12 HistGradientBoostingClassifier, fixed through 2025",
    "assumptions":"1 contract, 1 entry per 15m cycle, simulated quote spread, adverse slippage; intraminute path OHLC ordering unknown",
    "data_manifest":manifest,"train_rows":len(y),"scenarios":scenarios}
(OUT/"summary.json").write_text(json.dumps(meta,indent=2))
print("FINISHED_2025_BACKTEST",json.dumps(scenarios),flush=True)
