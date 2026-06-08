#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Predicted-best LLM vs an index of the best 2024 models that PERSIST into the modern era.
No chaining: the benchmark = per-round average of those 2024 models, compared to predicted-best
only on the modern rounds where they are actually present (paired two-level bootstrap)."""
import os, json, glob
from datetime import datetime, date
import numpy as np, pandas as pd
PFS="fb_work/forecastbench-processed-forecast-sets"; RD24=os.path.join(PFS,"2024-07-21")
MARKET={"manifold","metaculus","polymarket","infer"}; LO,HI=datetime(2025,3,1),datetime(2026,6,1); B=40000; RNG=np.random.default_rng(0); NDMIN,NMMIN=8,3
def load(f): return json.load(open(f,encoding="utf-8"))
def is_zs(m): return "(zero shot)" in m
def fam(m):
    s=m.lower()
    for k in ["gpt","o3","o4","claude","gemini","grok","qwen","deepseek","llama","mixtral","mistral","kimi","glm"]:
        if k in s: return "openai" if k in ("gpt","o3","o4") else k
    return "other"
def idx_from(ad,am):
    if not ad or not am: return np.nan
    return 100*(1-np.sqrt(max(0.5*np.mean(ad)+0.5*np.mean(am),0)))
# candidate best-2024 models (by 2024 score) to test for persistence
CAND={"GPT-4-Turbo":["gpt-4-turbo"],"Claude-3.5-Sonnet":["claude-3-5-sonnet"],"GPT-4o":["gpt-4o-2024","gpt-4o ("],
      "Claude-3-Opus":["claude-3-opus"],"GPT-4-0613":["gpt-4-0613"],"Gemini-1.5-Pro":["gemini-1.5-pro"],"Mistral-Large":["mistral-large"]}
def whichcand(m):
    s=m.lower()
    for k,ps in CAND.items():
        if any(p in s for p in ps): return k
    return None
# ---- 2024 human-set score of each candidate (to define "best") ----
hs=load(os.path.join(RD24,"2024-07-21.ForecastBench.human_super.json")); hk=set()
for x in hs["forecasts"]:
    if isinstance(x["id"],list) or not x.get("resolved"): continue
    hk.add((x["source"],str(x["id"]),x.get("resolution_date")))
def hidx(fc):
    ad,am=[],[]
    for x in fc:
        if isinstance(x["id"],list) or x.get("forecast") is None: continue
        k=(x["source"],str(x["id"]),x.get("resolution_date"))
        if k not in hk: continue
        b=(x["forecast"]-x["resolved_to"])**2; (am if x["source"] in MARKET else ad).append(b)
    return idx_from(ad,am)
sc2024={}
for fp in glob.glob(os.path.join(RD24,"*.json")):
    d=load(fp); m=d["model"]
    if ".ForecastBench." in os.path.basename(fp) or not is_zs(m): continue
    k=whichcand(m)
    if k: sc2024[k]=hidx(d["forecasts"])
# ---- dense window: per (model,round) raw index, contribs for bootstrap ----
def contribs(fc):
    D,M={},{}
    for x in fc:
        if isinstance(x["id"],list) or not x.get("resolved") or x.get("forecast") is None: continue
        k=(x["source"],str(x["id"]),x.get("resolution_date")); (M if x["source"] in MARKET else D)[k]=(x["forecast"]-x["resolved_to"])**2
    return D,M
CB={}; rows=[]
for rn in sorted(os.listdir(PFS)):
    rd=os.path.join(PFS,rn)
    if not os.path.isdir(rd): continue
    try: dt=datetime.strptime(rn,"%Y-%m-%d")
    except: continue
    if not (LO<=dt<=HI): continue
    CB[rn]={}
    for fp in glob.glob(os.path.join(rd,"*.json")):
        d=load(fp); m=d["model"]
        if d.get("model_organization")=="ForecastBench" or "freeze" in m.lower() or not is_zs(m): continue
        D,M=contribs(d["forecasts"]); v=idx_from(list(D.values()),list(M.values()))
        if np.isfinite(v): CB[rn][m]=(D,M); rows.append((rn,m,v))
pan=pd.DataFrame(rows,columns=["round","model","idx"]); pan["rel"]=pan["idx"]-pan.groupby("round")["idx"].transform("mean")
rounds=sorted(pan["round"].unique()); K=2.0
def shrink(v,pr): return pr if len(v)==0 else (sum(v)+K*pr)/(len(v)+K)
pbm={}
for t in rounds:
    hist=pan[pan["round"]<t]; cur=pan[pan["round"]==t]
    if cur.empty: continue
    bh=hist.groupby(hist["model"].map(fam))["rel"].mean().to_dict(); best=None
    for _,r in cur.iterrows():
        own=hist[hist["model"]==r["model"]]["rel"].values; pr=bh.get(fam(r["model"]),0.0); pr=0.0 if not np.isfinite(pr) else pr
        sc=shrink(list(own),pr)
        if best is None or sc>best[0]: best=(sc,r["model"])
    pbm[t]=best[1] if best else None
# ---- persistence: which candidates appear in modern rounds ----
pres={k:[] for k in CAND}
for t in rounds:
    for m in CB[t]:
        k=whichcand(m)
        if k: pres[k].append(t)
print("candidate | 2024 score | #modern rounds present | span")
persist=[]
for k in sorted(CAND,key=lambda k:-(sc2024.get(k) or 0)):
    P=sorted(pres[k]); sc=sc2024.get(k)
    print(f"  {k:18s} {('%.1f'%sc) if sc else '  - '} | {len(P):2d} | {(P[0]+'..'+P[-1]) if P else '-'}")
    if P and sc and sc>=56.0: persist.append(k)   # 'best' 2024 models that persist
print("\nBEST-2024 PERSISTERS used in index:",persist)

# ---- build per-round matrices; benchmark = avg of persisters present; predicted-best ----
def build(t):
    nm=list(CB[t]); DK=sorted(set().union(*[CB[t][m][0] for m in nm])); MK=sorted(set().union(*[CB[t][m][1] for m in nm]))
    D=np.array([[CB[t][m][0].get(k,np.nan) for k in DK] for m in nm]); M=np.array([[CB[t][m][1].get(k,np.nan) for k in MK] for m in nm])
    valid=np.array([np.sum(~np.isnan(D[i]))>=NDMIN and np.sum(~np.isnan(M[i]))>=NMMIN for i in range(len(nm))])
    bidx=[i for i,m in enumerate(nm) if whichcand(m) in persist]
    pbi=nm.index(pbm[t]) if pbm[t] in nm else None
    return dict(nm=nm,D=D,M=M,valid=valid,nd=len(DK),nm_d=len(MK),bidx=[i for i in bidx if valid[i]],pbi=pbi)
RB={t:build(t) for t in rounds}
def rv(t,di,mi):
    R=RB[t]
    with np.errstate(invalid="ignore"):
        mix=100*(1-np.sqrt(np.clip(0.5*np.nanmean(R["D"][:,di],axis=1)+0.5*np.nanmean(R["M"][:,mi],axis=1),0,None)))
    bench=np.nanmean([mix[i] for i in R["bidx"]]) if R["bidx"] else np.nan
    pb=mix[R["pbi"]] if (R["pbi"] is not None and R["valid"][R["pbi"]]) else np.nan
    return pb,bench
# comparison rounds: modern rounds with >=1 persister present (valid) AND predicted-best present
W=[t for t in rounds if RB[t]["bidx"] and RB[t]["pbi"] is not None and RB[t]["nd"]>=15 and RB[t]["nm_d"]>=3 and np.isfinite(rv(t,np.arange(RB[t]["nd"]),np.arange(RB[t]["nm_d"]))[1])]
print(f"\ncomparison rounds (no chaining): {len(W)}  {W[0]}..{W[-1]}")
pbv=[rv(t,np.arange(RB[t]['nd']),np.arange(RB[t]['nm_d']))[0] for t in W]; bnv=[rv(t,np.arange(RB[t]['nd']),np.arange(RB[t]['nm_d']))[1] for t in W]
print(f"predicted-best mean={np.nanmean(pbv):.1f} | best-2024-persisters index mean={np.nanmean(bnv):.1f} | gap={np.nanmean(pbv)-np.nanmean(bnv):+.2f}")
# paired two-level bootstrap (rounds then questions)
diffs=[]; nW=len(W)
for _ in range(B):
    rs=[W[i] for i in RNG.integers(0,nW,nW)]; dd=[]
    for t in rs:
        di=RNG.integers(0,RB[t]["nd"],RB[t]["nd"]); mi=RNG.integers(0,RB[t]["nm_d"],RB[t]["nm_d"]); pb,bn=rv(t,di,mi)
        dd.append(pb-bn)
    diffs.append(np.nanmean(dd))
diffs=np.array(diffs); p1=float(np.mean(diffs<=0))
print(f"predicted-best - persisters gap: mean {np.nanmean(diffs):+.2f}  95% CI [{np.percentile(diffs,2.5):+.2f},{np.percentile(diffs,97.5):+.2f}]")
print(f"p-value: 1-sided={p1:.3f}  2-sided={2*min(p1,1-p1):.3f}")
