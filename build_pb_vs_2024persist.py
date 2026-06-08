#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predicted-best LLM vs. the strong 2024 models that persist into the modern era (paper's p = 0.013).

For each modern round we compare the prequential predicted-best LLM to a benchmark equal to the
per-round average of the strong 2024 models still present that round (e.g. Claude-3.5-Sonnet,
GPT-4o, GPT-4-Turbo, Claude-3-Opus), scored on raw Brier. The comparison runs only on rounds where
at least one such persister is present, via a paired two-level bootstrap (rounds, then questions
within rounds). Prints the two-sided p-value reported in the paper.
"""
import os, json, glob
from datetime import datetime
import numpy as np, pandas as pd
PFS = "fb_work/forecastbench-processed-forecast-sets"; RD24 = os.path.join(PFS, "2024-07-21")
MARKET = {"manifold", "metaculus", "polymarket", "infer"}; LO, HI = datetime(2025, 3, 1), datetime(2026, 6, 1)
B = 40000; RNG = np.random.default_rng(0); NDMIN, NMMIN = 8, 3
def load(f): return json.load(open(f, encoding="utf-8"))
def is_zs(m): return "(zero shot)" in m
def fam(m):
    s = m.lower()
    for k in ["gpt", "o3", "o4", "claude", "gemini", "grok", "qwen", "deepseek", "llama", "mixtral", "mistral", "kimi", "glm"]:
        if k in s: return "openai" if k in ("gpt", "o3", "o4") else k
    return "other"
def index_of(ad, am):  # within-round difficulty-neutral index used for ranking (higher = better)
    if not ad or not am: return np.nan
    return 100 * (1 - np.sqrt(max(0.5 * np.mean(ad) + 0.5 * np.mean(am), 0)))

# candidate strong-2024 models to test for persistence into the modern era
CAND = {"GPT-4-Turbo": ["gpt-4-turbo"], "Claude-3.5-Sonnet": ["claude-3-5-sonnet"], "GPT-4o": ["gpt-4o-2024", "gpt-4o ("],
        "Claude-3-Opus": ["claude-3-opus"], "GPT-4-0613": ["gpt-4-0613"], "Gemini-1.5-Pro": ["gemini-1.5-pro"], "Mistral-Large": ["mistral-large"]}
def whichcand(m):
    s = m.lower()
    for k, ps in CAND.items():
        if any(p in s for p in ps): return k
    return None

# ---- 2024 human-set score of each candidate (defines which 2024 models count as "strong") ----
hs = load(os.path.join(RD24, "2024-07-21.ForecastBench.human_super.json")); hk = set()
for x in hs["forecasts"]:
    if isinstance(x["id"], list) or not x.get("resolved"): continue
    hk.add((x["source"], str(x["id"]), x.get("resolution_date")))
def h_index(fc):
    ad, am = [], []
    for x in fc:
        if isinstance(x["id"], list) or x.get("forecast") is None: continue
        k = (x["source"], str(x["id"]), x.get("resolution_date"))
        if k not in hk: continue
        b = (x["forecast"] - x["resolved_to"]) ** 2; (am if x["source"] in MARKET else ad).append(b)
    return index_of(ad, am)
sc2024 = {}
for fp in glob.glob(os.path.join(RD24, "*.json")):
    d = load(fp); m = d["model"]
    if ".ForecastBench." in os.path.basename(fp) or not is_zs(m): continue
    k = whichcand(m)
    if k: sc2024[k] = h_index(d["forecasts"])

# ---- modern window: per (model, round) Brier contributions + within-round ranking index ----
def contribs(fc):
    D, M = {}, {}
    for x in fc:
        if isinstance(x["id"], list) or not x.get("resolved") or x.get("forecast") is None: continue
        k = (x["source"], str(x["id"]), x.get("resolution_date")); (M if x["source"] in MARKET else D)[k] = (x["forecast"] - x["resolved_to"]) ** 2
    return D, M
CB = {}; rows = []
for rn in sorted(os.listdir(PFS)):
    rd = os.path.join(PFS, rn)
    if not os.path.isdir(rd): continue
    try: dt = datetime.strptime(rn, "%Y-%m-%d")
    except: continue
    if not (LO <= dt <= HI): continue
    CB[rn] = {}
    for fp in glob.glob(os.path.join(rd, "*.json")):
        d = load(fp); m = d["model"]
        if d.get("model_organization") == "ForecastBench" or "freeze" in m.lower() or not is_zs(m): continue
        D, M = contribs(d["forecasts"]); v = index_of(list(D.values()), list(M.values()))
        if np.isfinite(v): CB[rn][m] = (D, M); rows.append((rn, m, v))

# Predicted-best LLM (prequential): rank each round's models by prior-round relative performance
# (shrunk toward the family prior) and pick the predicted best; no look-ahead.
pan = pd.DataFrame(rows, columns=["round", "model", "index"]); pan["rel"] = pan["index"] - pan.groupby("round")["index"].transform("mean")
rounds = sorted(pan["round"].unique()); K = 2.0
def shrink(v, pr): return pr if len(v) == 0 else (sum(v) + K * pr) / (len(v) + K)
pbm = {}
for t in rounds:
    hist = pan[pan["round"] < t]; cur = pan[pan["round"] == t]
    if cur.empty: continue
    bh = hist.groupby(hist["model"].map(fam))["rel"].mean().to_dict(); best = None
    for _, r in cur.iterrows():
        own = hist[hist["model"] == r["model"]]["rel"].values; pr = bh.get(fam(r["model"]), 0.0); pr = 0.0 if not np.isfinite(pr) else pr
        sc = shrink(list(own), pr)
        if best is None or sc > best[0]: best = (sc, r["model"])
    pbm[t] = best[1] if best else None

# ---- which strong-2024 candidates persist into the modern rounds ----
pres = {k: [] for k in CAND}
for t in rounds:
    for m in CB[t]:
        k = whichcand(m)
        if k: pres[k].append(t)
print("candidate           2024 idx   #modern   span")
persist = []
for k in sorted(CAND, key=lambda k: -(sc2024.get(k) or 0)):
    P = sorted(pres[k]); sc = sc2024.get(k)
    print(f"  {k:18s} {('%.1f' % sc) if sc else '  -  '}       {len(P):2d}      {(P[0] + '..' + P[-1]) if P else '-'}")
    if P and sc and sc >= 56.0: persist.append(k)   # strong 2024 models (index >= 56) that persist
print("persisters used as the benchmark:", persist)

# ---- per-round matrices; benchmark = avg of persisters present; predicted-best ----
def build(t):
    nm = list(CB[t]); DK = sorted(set().union(*[CB[t][m][0] for m in nm])); MK = sorted(set().union(*[CB[t][m][1] for m in nm]))
    D = np.array([[CB[t][m][0].get(k, np.nan) for k in DK] for m in nm]); M = np.array([[CB[t][m][1].get(k, np.nan) for k in MK] for m in nm])
    valid = np.array([np.sum(~np.isnan(D[i])) >= NDMIN and np.sum(~np.isnan(M[i])) >= NMMIN for i in range(len(nm))])
    bidx = [i for i, m in enumerate(nm) if whichcand(m) in persist]
    pbi = nm.index(pbm[t]) if pbm[t] in nm else None
    return dict(D=D, M=M, valid=valid, nd=len(DK), nm_d=len(MK), bidx=[i for i in bidx if valid[i]], pbi=pbi)
RB = {t: build(t) for t in rounds}
def rv(t, di, mi):  # raw Brier of (predicted-best, persister benchmark) for one bootstrap draw
    R = RB[t]
    with np.errstate(invalid="ignore"):
        bo = np.clip(0.5 * np.nanmean(R["D"][:, di], axis=1) + 0.5 * np.nanmean(R["M"][:, mi], axis=1), 0, None)
    bidx = R["bidx"]; pbi = R["pbi"]; ok = (pbi is not None and R["valid"][pbi])
    bench = np.nanmean([bo[i] for i in bidx]) if bidx else np.nan
    pb = bo[pbi] if ok else np.nan
    return pb, bench
# comparison rounds: modern rounds with >=1 persister present and predicted-best present
W = [t for t in rounds if RB[t]["bidx"] and RB[t]["pbi"] is not None and RB[t]["nd"] >= 15 and RB[t]["nm_d"] >= 3
     and np.isfinite(rv(t, np.arange(RB[t]["nd"]), np.arange(RB[t]["nm_d"]))[1])]
fv = {t: rv(t, np.arange(RB[t]['nd']), np.arange(RB[t]['nm_d'])) for t in W}
print(f"\ncomparison rounds: {len(W)}  {W[0]}..{W[-1]}")
print(f"predicted-best = {np.nanmean([fv[t][0] for t in W]):.4f} Brier | persisters = {np.nanmean([fv[t][1] for t in W]):.4f} Brier")
# paired two-level bootstrap (rounds, then questions within rounds)
db = []; nW = len(W)
for _ in range(B):
    rs = [W[i] for i in RNG.integers(0, nW, nW)]; g = []
    for t in rs:
        di = RNG.integers(0, RB[t]["nd"], RB[t]["nd"]); mi = RNG.integers(0, RB[t]["nm_d"], RB[t]["nm_d"]); pb, bench = rv(t, di, mi)
        g.append(bench - pb)   # positive = predicted-best better (lower Brier)
    db.append(np.nanmean(g))
db = np.array(db); p1 = float(np.mean(db <= 0))
print(f"\npredicted-best beats the persisting 2024 models by {np.nanmean(db):+.4f} Brier; two-sided p = {2 * min(p1, 1 - p1):.4f}")
