#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replicate & extend ForecastBench Fig 1(b): forecast accuracy (Brier) vs log10 training compute (FLOP).
REPLICATE: 2024-07 models (Epoch AI FLOP) scored on the human set (raw Brier). Fit log-linear, find
           the superforecaster-crossing FLOP + bootstrap CI (as in the paper).
EXTEND:    2025-26 models with FLOP, CHAINED onto the 2024 accuracy scale via the shared benchmark
           (Claude-3.5-Sonnet, GPT-4o) offset between the dense window and the 2024-07 set.
Closed frontier models (GPT-4.1, o-series, Claude-4.x, Gemini-2.5/3) have NO FLOP -> annotated as missing.
"""
import os, json, glob
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PFS="fb_work/forecastbench-processed-forecast-sets"; QFE="fb_work/qfe_tournament.json"; OUT="."; RD24=os.path.join(PFS,"2024-07-21")
MARKET={"manifold","metaculus","polymarket","infer"}; DATASET={"acled","dbnomics","fred","wikipedia","yfinance"}; HMAX=62; LO,HI=datetime(2025,3,1),datetime(2026,6,1); B=3000; RNG=np.random.default_rng(0)
def load(f): return json.load(open(f,encoding="utf-8"))
def is_zs(m): return "(zero shot)" in m

# ---- training compute (FLOP). tier: solid / rough(imputed) / spec(ulative) / sub(paper substitution) ----
FLOP24={"GPT-4o":(3.8e25,"rough"),"GPT-4-Turbo":(2.2e25,"rough"),"GPT-4-0613":(2.1e25,"solid"),"Claude-3-Opus":(1.64e25,"rough"),
        "Claude-3-5-Sonnet":(2.7e25,"rough"),"Gemini-1.5-Pro":(1.58e25,"rough"),"Llama-2-70B":(8.1e23,"solid"),"Llama-3-8B":(7.2e23,"solid"),
        "Llama-3-70B":(7.861e24,"solid"),"Mixtral-8x7B":(7.74e23,"rough"),"Mixtral-8x22B":(2.34e24,"rough"),"Mistral-Large":(1.12e25,"rough"),
        "GPT-3.5-Turbo":(2.6e24,"sub"),"Claude-2.1":(3.9e24,"sub"),"Qwen1.5-110B":(1.3e24,"sub")}
FLOPX={"Claude-3-7-Sonnet":(3.35e25,"rough"),"DeepSeek-V3.1":(3.594e24,"solid"),"Qwen3-235B":(4.752e24,"rough"),
       "Llama-3.3-70B":(6.865e24,"solid"),"Llama-4-Scout":(4.08e24,"rough"),"Kimi-K2":(2.976e24,"solid"),
       "GPT-5":(6.6e25,"spec"),"Grok-4":(5.0e26,"spec")}
PAT={"GPT-4o":["gpt-4o-2024","gpt-4o ("],"GPT-4-Turbo":["gpt-4-turbo"],"GPT-4-0613":["gpt-4-0613","gpt-4 ("],"Claude-3-Opus":["claude-3-opus"],
     "Claude-3-5-Sonnet":["claude-3-5-sonnet"],"Gemini-1.5-Pro":["gemini-1.5-pro"],"Llama-2-70B":["llama-2-70b"],"Llama-3-8B":["llama-3-8b"],
     "Llama-3-70B":["llama-3-70b"],"Mixtral-8x7B":["mixtral-8x7b"],"Mixtral-8x22B":["mixtral-8x22b"],"Mistral-Large":["mistral-large"],
     "GPT-3.5-Turbo":["gpt-3.5-turbo"],"Claude-2.1":["claude-2.1"],"Qwen1.5-110B":["qwen1.5-110b"],
     "Claude-3-7-Sonnet":["claude-3-7-sonnet"],"DeepSeek-V3.1":["deepseek-v3.1"],"Qwen3-235B":["qwen3-235b"],"Llama-3.3-70B":["llama-3.3-70b"],
     "Llama-4-Scout":["llama-4-scout"],"Kimi-K2":["kimi-k2"],"GPT-5":["gpt-5-2025-08-07"],"Grok-4":["grok-4-0709"]}
def keyof(m,keys):
    s=m.lower()
    for k in keys:
        if any(p in s for p in PAT[k]): return k
    return None

# ---- scorers ----
hs=load(os.path.join(RD24,"2024-07-21.ForecastBench.human_super.json")); hkeys=set()
for x in hs["forecasts"]:
    if isinstance(x["id"],list) or not x.get("resolved"): continue
    hkeys.add((x["source"],str(x["id"]),x.get("resolution_date")))
def raw_human(fc):
    ad,am=[],[]
    for x in fc:
        if isinstance(x["id"],list) or x.get("forecast") is None: continue
        k=(x["source"],str(x["id"]),x.get("resolution_date"))
        if k not in hkeys: continue
        b=(x["forecast"]-x["resolved_to"])**2; (am if x["source"] in MARKET else ad).append(b)
    if not ad or not am: return np.nan
    return 100*(1-np.sqrt(max(0.5*np.mean(ad)+0.5*np.mean(am),0)))
_q=[r for r in load(QFE) if r["question_fixed_effect"] is not None]; QD={}
for r in _q:
    if r["source"] in MARKET: QD[(r["source"],r["id"],r["forecast_due_date"])]=r["question_fixed_effect"]
    else: QD[(r["source"],r["id"],r["forecast_due_date"],int(r["horizon"]))]=r["question_fixed_effect"]
GM=np.mean([r["question_fixed_effect"] for r in _q if r["source"] in MARKET]); GD=np.mean([r["question_fixed_effect"] for r in _q if r["source"] in DATASET])
def gam(s,i,f,du,re): return QD.get((s,i,f)) if s in MARKET else QD.get((s,i,f,(re-du).days))
def adj_index(fc):
    ad,am=[],[]
    for x in fc:
        if isinstance(x["id"],list) or not x["resolved"]: continue
        s=x["source"]; f=x["forecast_due_date"][:10]; du=datetime.fromisoformat(x["forecast_due_date"]); re=datetime.fromisoformat(x["resolution_date"])
        if (re-du).days>HMAX: continue
        g=gam(s,x["id"],f,du,re)
        if g is None: continue
        b=(x["forecast"]-x["resolved_to"])**2; (am if s in MARKET else ad).append(b-(g-(GM if s in MARKET else GD)))
    if not ad or not am: return np.nan
    return 100*(1-np.sqrt(max(0.5*np.mean(ad)+0.5*np.mean(am),0)))
def brier(idx): return (1-idx/100)**2

# ---- 2024-07 indices (human set) ----
idx24={}
for fp in glob.glob(os.path.join(RD24,"*.json")):
    bn=os.path.basename(fp)
    if ".ForecastBench." in bn or "freeze" in bn: continue
    d=load(fp); m=d["model"]
    if not is_zs(m): continue
    k=keyof(m,FLOP24.keys())
    if k:
        v=raw_human(d["forecasts"])
        if np.isfinite(v): idx24[k]=v
super24=raw_human(hs["forecasts"]); superB=brier(super24)
print(f"super 2024-07 index={super24:.1f} Brier={superB:.3f} | matched 2024 models: {sorted(idx24)}")

# ---- dense-window adjusted indices (for extension + benchmark offset) ----
denseacc={}; densebench={"Claude-3-5-Sonnet":[],"GPT-4o":[]}
for rn in sorted(os.listdir(PFS)):
    rd=os.path.join(PFS,rn)
    if not os.path.isdir(rd): continue
    try: dt=datetime.strptime(rn,"%Y-%m-%d")
    except: continue
    if not (LO<=dt<=HI): continue
    for fp in glob.glob(os.path.join(rd,"*.json")):
        d=load(fp); m=d["model"]
        if d.get("model_organization")=="ForecastBench" or "freeze" in m.lower() or not is_zs(m): continue
        kx=keyof(m,FLOPX.keys()); kb=keyof(m,["Claude-3-5-Sonnet","GPT-4o"])
        if kx is None and kb is None: continue
        v=adj_index(d["forecasts"])
        if not np.isfinite(v): continue
        if kx: denseacc.setdefault(kx,[]).append(v)
        if kb: densebench[kb].append(v)
denseacc={k:np.mean(v) for k,v in denseacc.items() if len(v)>=2}
# chain offset Delta = (dense adj) - (2024-07 raw) on the shared benchmark models
deltas=[]
for b_ in ["Claude-3-5-Sonnet","GPT-4o"]:
    if densebench[b_] and b_ in idx24: deltas.append(np.mean(densebench[b_])-idx24[b_])
Delta=float(np.mean(deltas)); print(f"chain offset Delta (dense-adj minus 2024-raw) = {Delta:+.2f} from {len(deltas)} benchmark models")

# ---- assemble points: (key, log10 flop, brier, group, tier) ----
rows=[]
for k,(F,tier) in FLOP24.items():
    if k in idx24: rows.append((k,np.log10(F),brier(idx24[k]),"2024",tier,idx24[k]))
for k,(F,tier) in FLOPX.items():
    if k in denseacc:
        cidx=denseacc[k]-Delta; rows.append((k,np.log10(F),brier(cidx),"2025-26",tier,cidx))
df=pd.DataFrame(rows,columns=["model","logflop","brier","group","tier","cidx"]); df.to_csv(os.path.join(OUT,"forecastbench_flop_scaling.csv"),index=False)
print(f"\npoints: {len(df)} ({(df.group=='2024').sum()} replication + {(df.group=='2025-26').sum()} extension)")
print(df.sort_values("logflop").to_string(index=False,float_format=lambda v:f"{v:.3f}"))

# ---- fits + crossing (Brier = a + b*logflop ; solve a+b*x=superB) ----
def fit_cross(sub):
    x=sub["logflop"].values; y=sub["brier"].values; b1,b0=np.polyfit(x,y,1); r=np.corrcoef(x,y)[0,1]
    xc=(superB-b0)/b1 if b1!=0 else np.nan
    return b0,b1,r,xc
def boot_cross(sub,n=B):
    xs=[]; idxarr=np.arange(len(sub))
    for _ in range(n):
        s=sub.iloc[RNG.integers(0,len(sub),len(sub))]
        if s["logflop"].nunique()<2: continue
        b1,b0=np.polyfit(s["logflop"],s["brier"],1)
        if b1<0: xs.append((superB-b0)/b1)
    xs=np.array(xs); return (np.percentile(xs,[2.5,50,97.5]) if len(xs) else (np.nan,)*3)
rep=df[df.group=="2024"]; comb=df.copy(); combhi=df[df.tier!="spec"]   # combined excl. speculative GPT-5/Grok-4
for name,sub in [("REPLICATION (2024 only)",rep),("EXTENDED (2024+2025-26)",comb),("EXTENDED minus speculative",combhi)]:
    b0,b1,r,xc=fit_cross(sub); lo,md,hi=boot_cross(sub)
    print(f"\n{name}: n={len(sub)} slope={b1:+.4f}/dex r={r:+.2f} | crossing FLOP=10^{xc:.2f}={10**xc:.2e} 95%CI[10^{lo:.2f},10^{hi:.2f}]")

# ---- figure ----
fig,ax=plt.subplots(figsize=(11,7.2))
col={"2024":"#c0392b","2025-26":"#1d4ed8"}; mk={"solid":"o","rough":"o","spec":"^","sub":"s"}
for _,r in df.iterrows():
    ax.scatter(r["logflop"],r["brier"],s=72,c=col[r["group"]],marker=mk.get(r["tier"],"o"),
               edgecolor="white",linewidth=.6,alpha=.5 if r["tier"] in("spec","rough","sub") else .95,zorder=4)
    ax.annotate(r["model"],(r["logflop"],r["brier"]),fontsize=6.0,alpha=.85,xytext=(3,3),textcoords="offset points")
xx=np.linspace(23.5,27.0,100)
for sub,c,lab,ls in [(rep,"#c0392b","2024 fit (replication)","--"),(combhi,"#1d4ed8","2024+2025-26 fit (ext., excl. spec.)","-")]:
    b0,b1,rr,xc=fit_cross(sub); ax.plot(xx,b0+b1*xx,c=c,ls=ls,lw=2,alpha=.85,label=f"{lab}: slope {b1:+.3f}/dex,  r={rr:+.2f},  R²={rr**2:.2f}")
ax.axhline(superB,color="#0e7c4a",ls=":",lw=1.8); ax.text(23.65,superB+0.004,f"Superforecaster Brier = {superB:.3f}",color="#0e7c4a",fontsize=9)
ax.set_xlabel("Log training compute  (log10 FLOP)",fontsize=10.5); ax.set_ylabel("Forecast Brier score  (chained to 2024 scale; lower = better)",fontsize=10.5)
ax.set_title("ForecastBench Fig 1(b) replicated & extended: forecast accuracy vs training compute",fontsize=12.5)
ax.set_xlim(23.5,27.0); ax.set_ylim(0.08,0.36); ax.grid(alpha=.25); ax.legend(loc="lower left",fontsize=9)
fig.text(0.5,-0.005,"Red ● = 2024 models (Epoch AI FLOP, human-set Brier) — replicates Fig 1(b).  Blue ● = 2025-26 models (FLOP available), accuracy CHAINED to the 2024 scale via the "
 f"Claude-3.5-Sonnet/GPT-4o benchmark offset ({Delta:+.1f} idx).  ▲ = speculative FLOP (GPT-5, Grok-4);  ■ = paper's hand-substituted FLOP.\n"
 "MISSING — no public FLOP, so EXCLUDED: GPT-4.1, o3, o4-mini, Claude-Sonnet-4/4.5, Claude-Opus-4.x, Gemini-2.5/3, Grok-4-Fast — i.e. the strongest 2025-26 forecasters.  "
 "r = signed Pearson correlation; R² = r² = variance explained. The low-compute, efficient 2025-26 open models sit above the 2024 line, weakening the compute–accuracy relationship.",
 ha="center",fontsize=6.6,va="top")
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(os.path.join(OUT,"forecastbench_flop_scaling.png"),dpi=150,bbox_inches="tight")
print("\nwrote forecastbench_flop_scaling.png/.csv")
