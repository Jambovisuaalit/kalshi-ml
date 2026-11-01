#!/usr/bin/env python3
"""Leakage-safe BTCUSDT spot-proxy training: December 2024 -> December 2025."""
import argparse
import hashlib
import json
import math
import pickle
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m"
FEATURES = ["delta_usd","remaining_sec","z_score","vol_15m","vol_60m","vol_ratio","mom_1m","mom_3m"]

def load(year,month,cache):
    name=f"BTCUSDT-1m-{year}-{month:02d}.zip"
    path=cache/name
    sha=cache/(name+".CHECKSUM")
    cache.mkdir(parents=True,exist_ok=True)
    for local,url in ((path,f"{BASE}/{name}"),(sha,f"{BASE}/{name}.CHECKSUM")):
        if not local.exists():
            response=requests.get(url,timeout=120)
            response.raise_for_status()
            local.write_bytes(response.content)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha.read_text().split()[0].lower()!=digest:
        raise ValueError("Binance archive SHA256 mismatch: "+name)
    with zipfile.ZipFile(path) as archive:
        filenames=[x for x in archive.namelist() if x.endswith(".csv")]
        if len(filenames)!=1:raise ValueError("Ambiguous Binance archive")
        frame=pd.read_csv(archive.open(filenames[0]),header=None,usecols=range(6),
                          names=["ts","open","high","low","close","volume"])
    frame["timestamp"]=pd.to_datetime(frame.pop("ts"),unit="us" if year>=2025 else "ms",utc=True)
    frame=frame.set_index("timestamp").sort_index()
    start=pd.Timestamp(f"{year}-{month:02d}-01T00:00:00Z")
    end=start+pd.offsets.MonthBegin(1)
    if not frame.index.equals(pd.date_range(start,end,freq="min",inclusive="left")):
        raise ValueError("Missing or duplicate BTCUSDT minutes: "+name)
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("Invalid Binance prices")
    return frame,{"archive":name,"sha256":digest,"rows":len(frame)}

def samples(frame):
    close=frame.close.to_numpy(float)
    minute=frame.index.minute.to_numpy()%15
    indices=np.arange(len(close))
    cycle=indices-minute
    strike=frame.open.to_numpy(float)[cycle]
    prior=np.roll(close,1);lag1=np.roll(close,2);lag3=np.roll(close,4)
    prior[:65]=np.nan;lag1[:65]=np.nan;lag3[:65]=np.nan
    returns=pd.Series(np.log(close)).diff()
    sigma15=returns.rolling(15).std(ddof=1).shift(1).to_numpy()
    sigma60=returns.rolling(60).std(ddof=1).shift(1).to_numpy()
    remain=900-minute*60
    d=prior-strike
    v15=sigma15*math.sqrt(15);v60=sigma60*math.sqrt(60)
    x=pd.DataFrame(np.column_stack([d,remain,
       d/np.maximum(.01,prior*sigma15*np.sqrt(remain/60)),v15,v60,
       np.divide(v15,v60,out=np.ones(len(close)),where=v60>0),
       prior/lag1-1,prior/lag3-1]),index=frame.index,columns=FEATURES)
    target=(close[cycle+14]>strike).astype(int)
    eligible=(minute>=1)&np.isfinite(x.to_numpy(float)).all(axis=1)
    no_tie=(close[cycle+14]!=strike)
    return x,target,eligible,no_tie,strike

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("data/raw/binance_monthly"))
    p.add_argument("--output",type=Path,default=Path("reports/kalshi_real_data/binance_proxy_oos.csv"))
    args=p.parse_args()
    train,train_meta=load(2024,12,args.cache)
    test,test_meta=load(2025,12,args.cache)
    assert train.index.max()<pd.Timestamp("2025-01-01T00:00:00Z")<test.index.min()
    xt,yt,eligible_train,no_tie,_=samples(train)
    xo,_,oos,_,strike=samples(test)
    ok=eligible_train & no_tie
    model=HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,
      max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,
      early_stopping=False,random_state=42)
    model.fit(xt.loc[ok,FEATURES],yt[ok])
    positions=np.flatnonzero(oos)
    probability=model.predict_proba(xo.iloc[positions][FEATURES])[:,1]
    digest=hashlib.sha256(pickle.dumps(model,protocol=4)).hexdigest()
    result=pd.DataFrame({"timestamp":test.index[positions].astype(str),
      "prediction_at":(test.index[positions]+pd.Timedelta(seconds=1)).astype(str),
      "trained_through":"2025-01-01T00:00:00+00:00",
      "p_yes":probability,"strike_proxy_usdt":strike[positions],
      "model_sha256":digest,"source":"BINANCE_SPOT_PROXY_NOT_KALSHI",
      "availability":"CLOSED_MINUTE_PLUS_1_SECOND_ASSUMED_NOT_VERIFIED"})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    meta={"status":"OOS_SPOT_PROXY_NOT_KALSHI","training":train_meta,
      "evaluation_input":test_meta,"train_samples":int(ok.sum()),
      "predictions":len(result),"model_sha256":digest,
      "labels":"Binance spot-proxy 15m close > first-minute open; NEVER official Kalshi settlement",
      "lookahead_rule":"Fit December 2024 only; infer December 2025 only"}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta,indent=2)+chr(10))
    print(json.dumps({"path":str(args.output),**meta},indent=2))
if __name__=="__main__":main()
