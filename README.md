# Replication package — *Conditional Regulation of Frontier AI with Automated Insider Forecasts*

Code to reproduce the forecasting-evidence figures and statistics in the paper (the human-vs-LLM
comparison, the training-compute scaling figure, and the associated p-values), built on the public
**ForecastBench** data (Karger et al. 2024).

## Contents
| File | Produces |
|---|---|
| `build_two_panel_chain.py` | Humans (July 2024) vs the modern LLM frontier (2025–26), chained through a fixed benchmark basket; prints the predicted-best-vs-new-benchmark p-value. → `forecastbench_two_panel_chain.png/.csv` |
| `build_flop_scaling.py` | Forecast accuracy vs log₁₀ training compute — replicates & extends Karger et al. (2024) Fig 1(b). → `forecastbench_flop_scaling.png/.csv` |
| `build_pb_vs_2024persist.py` | Predicted-best LLM vs an index of the best 2024 models that persist into 2025–26 (no chaining); prints the two-sided bootstrap p-value. |

## Data (not included; ~115 MB, public)
All ForecastBench inputs are public at **https://forecastbench.org**. Place them under `./fb_work/`:

1. **Processed forecast sets** — download `processed_forecast_sets.tar.gz` from the ForecastBench
   data page and extract so that rounds live at
   `fb_work/forecastbench-processed-forecast-sets/<YYYY-MM-DD>/*.json`.
   (Each round directory includes the per-model forecast files and, for the `2024-07-21` round, the
   `...ForecastBench.human_super.json` / `...human_public.json` human-survey files.)
2. **Question fixed effects** — download the ForecastBench question-fixed-effects dataset and save it
   as `fb_work/qfe_tournament.json`. (Needed only by `build_flop_scaling.py`, which uses the
   difficulty-adjusted index; the other two scripts use raw Brier on common question sets.)

Final layout:
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
Requires Python 3.10+ with `numpy`, `pandas`, `matplotlib`:
```
pip install numpy pandas matplotlib
python build_two_panel_chain.py
python build_flop_scaling.py
python build_pb_vs_2024persist.py
```
Outputs (PNGs/CSVs) are written to the working directory. The bootstrap reps (`B`) are set near the
top of each script; lower them for a quick run.

## Method notes
- **Scoring.** Per model/round we use a Brier index `100·(1 − √(0.5·mean(dataset Brier) + 0.5·mean(market Brier)))`.
  Cross-period panels use *raw* Brier on a common resolved question set (difficulty cancels because the
  same questions are scored for every forecaster); the compute figure uses the *difficulty-adjusted*
  index (each question's Brier minus its published question fixed effect), which reproduces the
  ForecastBench leaderboard to within ±0.05.
- **Human benchmark.** Superforecaster and public medians exist only for the one set humans
  forecasted (`2024-07-21`, all resolved questions); LLMs are scored on the identical questions.
- **Chaining.** Because no model spans both periods, the modern-era figure links to the 2024 scale
  through fixed-quality benchmark baskets — an "old" basket (Claude-3.5-Sonnet, GPT-4o) and a "new"
  basket (GPT-4.1, Gemini-2.5-Pro) that score near-identically over their March–August 2025 overlap.
- **Predicted-best (deployable).** Selected prequentially from prior-round performance only,
  removing the winner's-curse/optimizer's-curse bias of an ex-post-best ("oracle") pick.
- **Uncertainty.** Two-level bootstrap (resample rounds, then questions within rounds).
- **External capability data (hard-coded in the scripts, with sources).** LMArena Elo and the
  objective-benchmark values are from public leaderboards; training-compute (FLOP) estimates are from
  **Epoch AI**. Closed frontier models lack public compute estimates and are noted as excluded.

## Citation
Abaluck, J. (2026). *Conditional Regulation of Frontier AI with Automated Insider Forecasts.*
Built on ForecastBench: Karger, E., et al. (2024), *ForecastBench: A Dynamic Benchmark of AI
Forecasting Capabilities*, arXiv:2409.19839.
