#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2 - Forecasting accuracy of humans vs. the best LLMs (raw Brier, lower = better).

  Top panel (July 2024 human question set): superforecaster median, public median,
    single-best ("oracle") LLM, median LLM, and a fixed-quality benchmark (Claude-3.5-Sonnet).
  Bottom panel (2025-26 modern era): oracle LLM, prequential predicted-best LLM, median LLM,
    and a fixed-quality benchmark (GPT-4.1).

Every forecaster in a panel is scored on the identical resolved questions
(0.5*mean(dataset Brier) + 0.5*mean(market Brier)), so question difficulty cancels within a panel.
The two benchmark models are each present throughout their own era and overlap April-August 2025,
where GPT-4.1 is a stable ~0.015 Brier better than Claude-3.5-Sonnet; this near-constant offset
puts the two panels on a common scale and anchors the cross-era superforecaster-vs-oracle comparison.
95% CIs are two-level bootstraps (resample rounds, then questions within rounds).

Output: forecastbench_two_panel_chain.png / .csv
"""
import os, json, glob
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PFS = "fb_work/forecastbench-processed-forecast-sets"; OUT = "."; RD24 = os.path.join(PFS, "2024-07-21")
MARKET = {"manifold", "metaculus", "polymarket", "infer"}
LO, HI = datetime(2025, 3, 1), datetime(2026, 6, 1); B = 2000; RNG = np.random.default_rng(0); NDMIN, NMMIN = 8, 3

def load(f): return json.load(open(f, encoding="utf-8"))
def is_zs(m): return "(zero shot)" in m
def isb(m, w):  # benchmark-model identifiers
    s = m.lower()
    return {"c35": "claude-3-5-sonnet" in s, "gpt41": "gpt-4.1-2025-04" in s}[w]
def fam(m):     # model family, for the predicted-best prior
    s = m.lower()
    for k in ["gpt", "o3", "o4", "claude", "gemini", "grok", "qwen", "deepseek", "llama", "mixtral", "mistral", "kimi", "glm"]:
        if k in s: return "openai" if k in ("gpt", "o3", "o4") else k
    return "other"
def brier(dv, mv):  # 0.5*dataset + 0.5*market mean Brier over present questions
    d = dv[~np.isnan(dv)]; m = mv[~np.isnan(mv)]
    if len(d) == 0 or len(m) == 0: return np.nan
    return 0.5 * d.mean() + 0.5 * m.mean()
def pct(a): a = np.asarray(a, float); a = a[np.isfinite(a)]; return np.percentile(a, [2.5, 50, 97.5]) if len(a) else (np.nan,) * 3
def nmean(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return x.mean() if len(x) else np.nan

hs = load(os.path.join(RD24, "2024-07-21.ForecastBench.human_super.json")); hkeys = set()
for x in hs["forecasts"]:
    if isinstance(x["id"], list) or not x.get("resolved"): continue
    hkeys.add((x["source"], str(x["id"]), x.get("resolution_date")))
def c_human(fc):  # Brier contributions restricted to the human-forecasted question set
    D, M = {}, {}
    for x in fc:
        if isinstance(x["id"], list) or x.get("forecast") is None: continue
        k = (x["source"], str(x["id"]), x.get("resolution_date"))
        if k in hkeys: (M if x["source"] in MARKET else D)[k] = (x["forecast"] - x["resolved_to"]) ** 2
    return D, M
def c_round(fc):  # Brier contributions over a round's resolved questions
    D, M = {}, {}
    for x in fc:
        if isinstance(x["id"], list) or not x.get("resolved") or x.get("forecast") is None: continue
        k = (x["source"], str(x["id"]), x.get("resolution_date")); (M if x["source"] in MARKET else D)[k] = (x["forecast"] - x["resolved_to"]) ** 2
    return D, M
def vecs(D, M, dk, mk): return (np.array([D.get(k, np.nan) for k in dk]), np.array([M.get(k, np.nan) for k in mk]))
def boot(dv, mv, di, mi):
    d = dv[di]; d = d[~np.isnan(d)]; m = mv[mi]; m = m[~np.isnan(m)]
    if len(d) == 0 or len(m) == 0: return np.nan
    return 0.5 * d.mean() + 0.5 * m.mean()

OLDLAB = "Old benchmark: Claude-3.5-Sonnet"
NEWLAB = "New benchmark: GPT-4.1"

# ===== Panel 1: July 2024 human question set =====
sup = c_human(hs["forecasts"]); pub = c_human(load(os.path.join(RD24, "2024-07-21.ForecastBench.human_public.json"))["forecasts"])
m24 = {}
for fp in glob.glob(os.path.join(RD24, "*.json")):
    bn = os.path.basename(fp)
    if ".ForecastBench." in bn or "freeze" in bn: continue
    d = load(fp)
    if is_zs(d["model"]): m24[d["model"]] = c_human(d["forecasts"])
DK = sorted(set().union(sup[0], pub[0], *[c[0] for c in m24.values()])); MK = sorted(set().union(sup[1], pub[1], *[c[1] for c in m24.values()]))
sd, sm = vecs(*sup, DK, MK); pd_, pm = vecs(*pub, DK, MK); names24 = list(m24)
Dm = np.array([vecs(*m24[m], DK, MK)[0] for m in names24]); Mm = np.array([vecs(*m24[m], DK, MK)[1] for m in names24])
el24 = np.array([np.sum(~np.isnan(Dm[i])) >= 10 and np.sum(~np.isnan(Mm[i])) >= 5 for i in range(len(names24))])
c35i = [i for i, m in enumerate(names24) if isb(m, "c35")][0]
def mi24(di, mi):
    with np.errstate(invalid="ignore"):
        return np.clip(0.5 * np.nanmean(Dm[:, di], axis=1) + 0.5 * np.nanmean(Mm[:, mi], axis=1), 0, None)
nd, nm = len(DK), len(MK); m0 = mi24(np.arange(nd), np.arange(nm)); old24 = m0[c35i]
P1 = {"Superforecaster median (human)": brier(sd, sm), "Public median (human)": brier(pd_, pm),
      "Oracle LLM 2024": np.nanmin(m0[el24]), "Median LLM": np.nanmedian(m0[el24]),
      OLDLAB: old24}
P1b = {k: [] for k in P1}; g_super_old = []
for _ in range(B):
    di = RNG.integers(0, nd, nd); mi = RNG.integers(0, nm, nm); mm = mi24(di, mi)
    s = boot(sd, sm, di, mi); p = boot(pd_, pm, di, mi); oa = mm[c35i]
    P1b["Superforecaster median (human)"].append(s); P1b["Public median (human)"].append(p)
    P1b["Oracle LLM 2024"].append(np.nanmin(mm[el24])); P1b["Median LLM"].append(np.nanmedian(mm[el24]))
    P1b[OLDLAB].append(oa); g_super_old.append(s - oa)
P1ci = {k: pct(v) for k, v in P1b.items()}
print(f"Panel 1 (2024 human set): {nd} dataset + {nm} market Qs | old benchmark (Claude-3.5-Sonnet) = {old24:.3f}")

# ===== modern window 2025-26 =====
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
        D, M = c_round(d["forecasts"]); v = brier(np.array(list(D.values())), np.array(list(M.values())))
        if np.isfinite(v): CB[rn][m] = (D, M); rows.append((rn, m, v))

# Predicted-best LLM (prequential): rank each round's models by their prior-round relative
# performance, shrunk toward their family's prior, and pick the predicted best (no look-ahead).
# Relative performance is measured on a within-round difficulty-neutral index so that rounds with
# different question difficulty are comparable; the bars/CIs below report raw Brier.
pan = pd.DataFrame(rows, columns=["round", "model", "brier"])
pan["index"] = 100 * (1 - np.sqrt(pan["brier"].clip(lower=0)))
pan["rel"] = pan["index"] - pan.groupby("round")["index"].transform("mean")
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
        if best is None or sc > best[0]: best = (sc, r["model"])  # higher relative index = better
    pbm[t] = best[1] if best else None

def build_round(t):
    nm_ = list(CB[t]); DKr = sorted(set().union(*[CB[t][m][0] for m in nm_])); MKr = sorted(set().union(*[CB[t][m][1] for m in nm_]))
    D = np.array([[CB[t][m][0].get(k, np.nan) for k in DKr] for m in nm_]); M = np.array([[CB[t][m][1].get(k, np.nan) for k in MKr] for m in nm_])
    valid = np.array([np.sum(~np.isnan(D[i])) >= NDMIN and np.sum(~np.isnan(M[i])) >= NMMIN for i in range(len(nm_))])
    def ri(w): c = [j for j, m in enumerate(nm_) if isb(m, w)]; return c[0] if c else None
    return dict(D=D, M=M, valid=valid, nd=len(DKr), nm_d=len(MKr), c35=ri("c35"), gpt41=ri("gpt41"),
                pb=(nm_.index(pbm[t]) if pbm[t] in nm_ else None))
RB = {t: build_round(t) for t in rounds}
def round_vals(t, di, mi):
    R = RB[t]
    with np.errstate(invalid="ignore"):
        mix = np.clip(0.5 * np.nanmean(R["D"][:, di], axis=1) + 0.5 * np.nanmean(R["M"][:, mi], axis=1), 0, None)
    gv = lambda key: (mix[R[key]] if (R[key] is not None and R["valid"][R[key]]) else np.nan)
    return {"oracle": np.nanmin(mix[R["valid"]]) if R["valid"].any() else np.nan,
            "median": np.nanmedian(mix[R["valid"]]) if R["valid"].any() else np.nan,
            "pb": (mix[R["pb"]] if (R["pb"] is not None and R["valid"][R["pb"]]) else np.nan),
            "old": gv("c35"), "new": gv("gpt41")}
def covered(t): return RB[t]["nd"] >= 15 and RB[t]["nm_d"] >= 3
fullv = {t: round_vals(t, np.arange(RB[t]["nd"]), np.arange(RB[t]["nm_d"])) for t in rounds}
W = [t for t in rounds if covered(t) and np.isfinite(fullv[t]["new"])]   # window: new benchmark (GPT-4.1) present
O = [t for t in W if np.isfinite(fullv[t]["old"])]                       # overlap: old benchmark also present
print(f"Modern window {W[0]}..{W[-1]} ({len(W)} rounds); benchmarks overlap in {len(O)} rounds")

# Benchmark-equivalence offset and the chained old benchmark (used only for the cross-era
# super-vs-oracle comparison; the bottom panel itself plots GPT-4.1, which is present throughout).
def ochain(rv, e): return rv["old"] if np.isfinite(rv["old"]) else (rv["new"] + e if np.isfinite(rv["new"]) else np.nan)
eps = nmean([fullv[t]["old"] - fullv[t]["new"] for t in O])
keys = ["oracle", "pb", "new", "median"]
P2 = {k: nmean([fullv[t][k] for t in W]) for k in keys}; P2["old_chain"] = nmean([ochain(fullv[t], eps) for t in W])
draws = {k: [] for k in keys + ["old_chain"]}; gap = []; nW, nO = len(W), len(O)
for _ in range(B):
    rs = [W[i] for i in RNG.integers(0, nW, nW)]; os_ = [O[i] for i in RNG.integers(0, nO, nO)]
    ev = []
    for t in os_:
        di = RNG.integers(0, RB[t]["nd"], RB[t]["nd"]); mi = RNG.integers(0, RB[t]["nm_d"], RB[t]["nm_d"]); rv = round_vals(t, di, mi); ev.append(rv["old"] - rv["new"])
    epsb = nmean(ev); acc = {k: [] for k in keys}; oc = []
    for t in rs:
        di = RNG.integers(0, RB[t]["nd"], RB[t]["nd"]); mi = RNG.integers(0, RB[t]["nm_d"], RB[t]["nm_d"]); rv = round_vals(t, di, mi)
        for k in keys: acc[k].append(rv[k])
        oc.append(ochain(rv, epsb))
    for k in keys: draws[k].append(nmean(acc[k]))
    draws["old_chain"].append(nmean(oc)); gap.append(nmean(acc["oracle"]) - nmean(oc))
P2ci = {k: pct(v) for k, v in draws.items()}; gap = np.array(gap)
g1 = np.array(g_super_old); n = min(len(g1), len(gap)); chain = g1[:n] - gap[:n]
chain_pt = (P1["Superforecaster median (human)"] - old24) - (P2["oracle"] - P2["old_chain"]); cl, cm, ch = pct(chain)
print(f"Panel 2: oracle={P2['oracle']:.3f} predicted-best={P2['pb']:.3f} new benchmark (GPT-4.1)={P2['new']:.3f} median={P2['median']:.3f}")
print(f"Overlap offset (old - new) = {eps:+.3f} Brier over {O[0]}..{O[-1]}")
print(f"Cross-era: superforecasters lower (better) than the modern oracle by {-chain_pt:.3f} Brier "
      f"95% CI [{-ch:.3f}, {-cl:.3f}], P(super ahead) = {np.mean(chain < 0):.2f}")

# ===== figure =====
HUM = "#c0392b"; ORA = "#0e7c4a"; MED = "#1d4ed8"; PB = "#8e44ad"; OLD = "#e67e22"; NEW = "#b8860b"
def draw(ax, pt, ci, order, colors, title, anchors):
    y = np.arange(len(order))[::-1]; vals = [pt[k] for k in order]
    err = np.array([[max(pt[k] - ci[k][0], 0), max(ci[k][2] - pt[k], 0)] for k in order]).T
    ax.barh(y, vals, color=colors, height=0.6, xerr=err, error_kw=dict(ecolor="#222", elinewidth=1.1, capsize=3))
    for yy, k in zip(y, order): ax.text(ci[k][2] + 0.003, yy, f"{pt[k]:.3f}", va="center", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=11); ax.set_xlim(0.08, 0.24); ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_title(title, fontsize=13, loc="left"); ax.grid(True, axis="x", alpha=.25)
    for av, c in anchors:
        if np.isfinite(av): ax.axvline(av, color=c, ls="--", lw=1, alpha=.65)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9.5), gridspec_kw={"height_ratios": [5, 4]})
O1 = ["Superforecaster median (human)", "Public median (human)", "Oracle LLM 2024", "Median LLM", OLDLAB]
draw(a1, P1, P1ci, O1, [HUM, HUM, ORA, MED, OLD], "(1) July 2024 - humans vs LLMs (human set, all resolved)", [(old24, OLD)])
P2L = {"Oracle LLM 2025-2026": P2["oracle"], "Predicted-best LLM": P2["pb"], "Median LLM": P2["median"], NEWLAB: P2["new"]}
P2Lci = {"Oracle LLM 2025-2026": P2ci["oracle"], "Predicted-best LLM": P2ci["pb"], "Median LLM": P2ci["median"], NEWLAB: P2ci["new"]}
draw(a2, P2L, P2Lci, list(P2L), [ORA, PB, MED, NEW], f"(2) Modern era 2025-26 ({W[0]}...{W[-1]}, {len(W)} rounds)", [(P2["new"], NEW)])
for a in (a1, a2): a.tick_params(axis="x", labelsize=10); a.set_xlabel("Brier score (raw, on each panel's resolved set; lower = better)", fontsize=12)
fig.suptitle("Humans (2024) vs the modern LLM frontier (2025-26), 95% CIs", fontsize=14, y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "forecastbench_two_panel_chain.png"), dpi=150, bbox_inches="tight")
out = []
for pnl, pt, ci in [("P1", P1, P1ci), ("P2", P2L, P2Lci)]:
    for k in pt: out.append({"panel": pnl, "bar": k, "point": pt[k], "lo": ci[k][0], "hi": ci[k][2]})
pd.DataFrame(out).to_csv(os.path.join(OUT, "forecastbench_two_panel_chain.csv"), index=False)
print("wrote forecastbench_two_panel_chain.png/.csv")
