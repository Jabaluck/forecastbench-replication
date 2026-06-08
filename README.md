# Replication package — *Conditional Regulation of Frontier AI with Automated Insider Forecasts*

Code to reproduce the forecasting-evidence results in the paper, built on the public **ForecastBench**
data (Karger et al. 2024): the human-vs-LLM comparison (Figure 2), the training-compute scaling
figure (Figure 1), and the predicted-best-vs-persisters statistic.

## Contents
| File | Produces |
|---|---|
| `build_two_panel_chain.py` | **Figure 2** — humans (July 2024) vs the modern LLM frontier (2025–26); each panel scored on raw Brier, with a fixed-quality benchmark model. → `forecastbench_two_panel_chain.png/.csv` |
| `build_flop_scaling.py` | **Figure 1** — forecast accuracy vs log₁₀ training compute; replicates & extends Karger et al. (2024) Fig 1(b). → `forecastbench_flop_scaling.png/.csv` |
| `build_pb_vs_2024persist.py` | The headline statistic — the prequential predicted-best LLM vs. the strong 2024 models that persist into 2025–26; prints the two-sided bootstrap **p = 0.013**. |

## Data (not included; public, ~115 MB)
All inputs are public ForecastBench data. Place them under `./fb_work/`:

1. **Processed forecast sets** — download from forecastbench.org (or the ForecastBench datasets repo)
   and extract so rounds live at `fb_work/forecastbench-processed-forecast-sets/<YYYY-MM-DD>/*.json`.
   Each round directory holds the per-model zero-shot forecast files; the `2024-07-21` round also holds
   the `...ForecastBench.human_super.json` / `...human_public.json` human-survey files.
2. **Question fixed effects** — save the ForecastBench question-fixed-effects file as
   `fb_work/qfe_tournament.json` (needed only by `build_flop_scaling.py`).

```
.
├── build_two_panel_chain.py
├── build_flop_scaling.py
├── build_pb_vs_2024persist.py
└── fb_work/
    ├── qfe_tournament.json
    └── forecastbench-processed-forecast-sets/<rounds>/*.json
```

## Running
Python 3.10+ with `numpy`, `pandas`, `matplotlib`:
```
pip install numpy pandas matplotlib
python build_two_panel_chain.py
python build_flop_scaling.py
python build_pb_vs_2024persist.py
```
Outputs (PNGs/CSVs) are written to the working directory. The bootstrap reps (`B`) are set near the
top of each script; lower them for a quick run.

## Method notes
- **Scoring.** A forecaster's score in a round is raw Brier, `0.5·mean(dataset Brier) + 0.5·mean(market Brier)`,
  over the questions it answered (lower = better). Within each panel every forecaster is scored on the
  *identical* resolved questions, so question difficulty cancels and no adjustment is needed.
- **Benchmark models (Figure 2).** Each panel shows the benchmark with real data in that era: the
  **old benchmark, Claude-3.5-Sonnet** (present in July 2024 and the early modern rounds) and the
  **new benchmark, GPT-4.1** (present throughout 2025–26). They overlap April–August 2025, where
  GPT-4.1 is a stable ~0.015 Brier better; this near-constant offset puts the two panels on a common
  scale and anchors the cross-era superforecaster-vs-oracle comparison.
- **Predicted-best (deployable).** Selected prequentially: in each round, models are ranked by their
  prior-round performance only (relative to that round's mean, shrunk toward a model-family prior),
  and the predicted best is picked — removing the winner's-curse bias of an ex-post "oracle" pick.
  Ranking uses a within-round difficulty-neutral index; all *reported* scores are raw Brier.
- **Difficulty adjustment (Figure 1 only).** The compute figure must compare scores across different
  question sets, so each question's raw Brier is adjusted by ForecastBench's published question fixed
  effect. Using those published fixed effects, our per-model scores reproduce ForecastBench's published
  per-model leaderboard closely — within ~2 points on its 0–100 index (≈0.015 Brier on average) across
  the 68 matchable models.
- **Uncertainty.** Two-level bootstrap: resample rounds, then questions within rounds.
- **Training compute (Figure 1).** FLOP estimates are from Epoch AI; closed frontier models without
  public estimates are excluded and noted on the figure.

## Citation
Abaluck, J. (2026). *Conditional Regulation of Frontier AI with Automated Insider Forecasts.*
Built on ForecastBench: Karger, E., et al. (2024), *ForecastBench: A Dynamic Benchmark of AI
Forecasting Capabilities*, arXiv:2409.19839.
