# Implementing AI Quant Trading with Python — Dummy Data Examples

This repository contains **33 runnable scripts** covering the quick start in Chapter 00 through Chapter 27, plus a learning guide in Chapter 28. They follow the concepts shared by the Korean, English, and Japanese editions of the book. Runnable examples and repository documentation use English. All Python listings share canonical English source; surrounding explanations and non-Python illustrations may be translated.

In the chapter scripts, all prices, volumes, returns, and order events are **dummy data**, generated or hardcoded in the examples. These exercises are for learning and testing. No account, API key, broker connection, or market data download is required. Installing dependencies requires internet access; the examples run locally afterward. Reported returns describe synthetic experiments.

## 1. Quick start

Clone the repository and enter its root directory, where this README is located. The examples are verified with **Python 3.12**.

```sh
git clone https://github.com/tedpark/stock-trading-ai-examples.git
cd stock-trading-ai-examples

# Run the first example without installing any libraries.
python3 run.py 00
# Expected output: 1000.0 0.2 (final cash / maximum drawdown)

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py core
```

On Windows PowerShell, use `py -3.12` instead of `python3` and activate the environment with `.venv\Scripts\Activate.ps1`. The remaining commands are the same after activation. If activation is restricted, invoke the interpreter directly, for example `.venv\Scripts\python.exe run.py 00`.

## 2. Run a chapter or the full collection

```sh
python run.py --list         # List chapter files.
python run.py 03             # Run Chapter 03 accounting, risk metrics, and DSR.
python ch21_regime_zscore.py # Run one file directly.

# Include SAC (Chapters 14/23) and MLflow (Chapter 27).
python -m pip install -r requirements-full.txt
python run.py all            # Expected summary: PASS 33/33.
python -m unittest discover -s tests -v
python verify.py            # Block outbound connections and run all examples/tests.
```

`core` excludes Chapters 14, 23, and 27, which require Torch/SAC or MLflow. Use the same virtual environment for every command. Missing dependencies cause `all` to report the failed files and return exit code 1. Each script has a 240-second limit; the runner continues checking the remaining scripts after a failure.

SAC trains a small neural network on the CPU for just 128 steps. Torch is a large initial download, so install only `requirements.txt` to start with the core exercises. Random seeds are fixed, but library and operating-system differences can affect model results and the last decimal places.

## Integrated paper-trading service

The chapter scripts remain offline examples. A separate [integrated paper service](paper_service/README.md) now connects data, HMM training, causal signals, risk limits, simulated fills, persistent accounting, an HTTP API and optional MLflow recording. Start it with `python -m uvicorn paper_service.app:app --host 127.0.0.1 --port 8766` after installing `requirements-service.txt`. `verify_service.py` tests actual local HTTP servers. Optional Yahoo downloads require `requirements-market.txt`; broker orders are never submitted.

## Repository-first book workflow

The repository is authoritative. `book_contract.json` maps 107 Python slots per edition to source: 34 runnable mappings read the actual chapter/helper `.py` files, and 73 explanatory/counterexample slots read `book_snippets/`. Change and test those files first. Never fix a manuscript or derived archive instead of the source.

With a private book checkout available, run `python sync_book.py --book-root /path/to/stock-trading-ai --write --refresh-index`, then `python sync_book.py --book-root /path/to/stock-trading-ai --check`. The tool updates all three editions and refuses missing, added, or reordered source anchors. Default mode only checks. Index refresh refuses manuscript drift and never rewrites canonical source. The book build also checks its repository commit/hash lock.

`tests/test_repository_first.py` proves source changes reach all editions, manuscript edits cannot rewrite source, and unbound/missing/wrongly anchored code is rejected. The source archive below is a derived view, not the editing source.

## Complete printed-source inventory

[book_sources/README.md](book_sources/README.md) and `book_sources/index.json` map every fenced code block from all three editions, including all 107 Python blocks per edition. Only 34 per edition map to runnable examples; the remaining 73 are explicitly classified fragments or counterexamples. Exact source inclusion is not a claim that every design is implemented. Run `python verify_book_sources.py` to check integrity and mappings.

## 3. How the examples map to the book

The scripts combine the book's standalone runnable blocks with any definitions they need. Chapters 04, 21, and 22 adapt explanatory or architectural excerpts into runnable dummy-data exercises. Chapter 28 prints a learning path and file list. Chapter 12 removes a duplicate function definition and rejects invalid half-lives, entry thresholds, correlations, and volume multipliers.

The examples demonstrate each chapter's calculations and workflow in a compact form; they do not reconstruct an entire production trading service.

`manifest.json` maps scripts to manuscript filenames and identifies adaptations. `tiny_env.py` contains Chapter 13's environment used by Chapters 14 and 23. `monitor_example.py` provides the monitor function used by Chapter 18's API. `hmm_demo.py` shares dummy data and forward filtering between Chapters 04 and 22.

| Chapter | Example files | What to check |
|---|---|---|
| 00 | [ch00_practice.py](ch00_practice.py) | Positions, cash, and maximum drawdown: 1000.0 / 0.2 |
| 01 | [ch01_pipeline.py](ch01_pipeline.py) | A small price pipeline with trading costs |
| 02 | [ch02_features.py](ch02_features.py) | Four features; train/OOS split: (28, 4) / (12, 4) |
| 03 | [ch03_practice.py](ch03_practice.py)<br>[ch03_metrics.py](ch03_metrics.py)<br>[ch03_dsr.py](ch03_dsr.py) | Accounting, MDD, and DSR; drawdown 0.2 and consistent P&L |
| 04 | [ch04_hmm.py](ch04_hmm.py) | HMM training, variance-based state ordering, normalized probabilities |
| 05 | [ch05_forward.py](ch05_forward.py) | Forward state probabilities without future observations |
| 06 | [ch06_kalman.py](ch06_kalman.py) | Smooth noisy prices with a Kalman filter |
| 07 | [ch07_practice.py](ch07_practice.py) | Joint HMM on 20-dimensional synthetic observations |
| 08 | [ch08_ensemble.py](ch08_ensemble.py) | Combine two probability layers and scale risk exposure |
| 09 | [ch09_kelly.py](ch09_kelly.py) | Kelly sizing with a cap: 0.05 |
| 10 | [ch10_optuna.py](ch10_optuna.py) | Separate Optuna training, validation, and holdout data |
| 11 | [ch11_practice.py](ch11_practice.py) | Synthetic pair cointegration, correlation, and half-life |
| 12 | [ch12_practice.py](ch12_practice.py) | Entry checks with reasons for acceptance or rejection |
| 13 | [ch13_env.py](ch13_env.py) | 11-dimensional observations, round-trip costs, episode termination |
| 14 | [ch14_sac.py](ch14_sac.py) | Small SAC training run and 70-step evaluation |
| 15 | [ch15_optuna.py](ch15_optuna.py) | Optuna search followed by untouched holdout evaluation |
| 16 | [ch16_metrics.py](ch16_metrics.py) | MDD, Sharpe, VaR, CVaR, and contributions |
| 17 | [ch17_order_events.py](ch17_order_events.py) | Partial fills, duplicate-fill handling, and cancellation |
| 18 | [ch18_monitor.py](ch18_monitor.py)<br>[ch18_lab.py](ch18_lab.py) | Monitor reports and HTTP /health and /report endpoints |
| 19 | [ch19_cointegration.py](ch19_cointegration.py) | OLS hedge ratio and cointegration testing |
| 20 | [ch20_screening.py](ch20_screening.py)<br>[ch20_regime_events.py](ch20_regime_events.py) | Pair screening and regime event counts |
| 21 | [ch21_regime_zscore.py](ch21_regime_zscore.py) | Regime-dependent entry, exit, and stop decisions for the same Z-score |
| 22 | [ch22_pipeline.py](ch22_pipeline.py) | Dummy prices through OOS evaluation after costs |
| 23 | [ch23_sac.py](ch23_sac.py) | SAC training and evaluation workflow |
| 24 | [ch24_validation.py](ch24_validation.py) | Reject reversed timestamps, duplicates, and invalid probabilities |
| 25 | [ch25_per.py](ch25_per.py) | PER sampling probabilities and importance weights |
| 26 | [ch26_quantile_cvar.py](ch26_quantile_cvar.py) | CVaR of the lower return tail |
| 27 | [ch27_mlflow.py](ch27_mlflow.py) | Verify local MLflow parameters, metrics, and artifacts |
| 28 | [ch28_guide.py](ch28_guide.py) | Suggested learning path and file listing |

## 4. Modeling assumptions

- In Chapters 04 and 22, the HMM and scaler are fitted on the first 60% of the feature rows. Forward filtering uses only observations available up to each point. State ordering uses total variance across standardized features, not return volatility alone. This ranking does not establish that a state is mean-reverting.
- Chapter 22 decides the position using the close at time t and applies it to the return from t to t+1. This simplified close-fill model charges 0.1% on position changes and includes final liquidation costs. It does not model actual execution, order books, or slippage.
- Chapter 21 uses **dummy probabilities** without an external HMM to isolate the decision rules. It blocks new entries when transition or high volatility dominates checks stops first for existing positions, and rejects new entries at or beyond the stop threshold.
- Chapter 17 changes only in-memory simulated order state. Chapter 18 uses FastAPI TestClient without opening a server port.
- Chapter 27 writes to temporary local SQLite and artifact storage, verifies the logged data, then cleans up. No separate MLflow server is needed.
- Chapter 25 demonstrates PER calculations; Chapter 26 demonstrates CVaR calculations with fractional boundary-bin weights so the requested tail probability is preserved. Neither implements an entire custom RL algorithm.

## 5. Verification and practice

Assertions check expected results in the calculation examples. Full execution and additional tests cover output-oriented examples. Tests check chapter coverage, future-data leakage in features and HMM filtering, RL observations and liquidation costs, episode termination, regime gates and stops, HTTP responses, duplicate fills, and overfill rejection.

Start by changing one price or fee and observing the result. If an assertion with a fixed expected value fails, calculate the new result by hand before updating the expectation. Keep experiments small, and avoid choosing synthetic data just to improve reported performance.

See [REVIEW.md](REVIEW.md) for numerical fixes, input checks, and comment clarifications from the latest review. See `VALIDATION.json` for the verification environment and results. `requirements-lock.txt` records the complete verified environment; use the core/full requirement files above for normal installation. Third-party deprecation warnings may appear depending on your environment.

## Public repository

[GitHub: tedpark/stock-trading-ai-examples](https://github.com/tedpark/stock-trading-ai-examples)

The repository contains learning examples, installation instructions, and tests. It excludes book manuscripts/PDFs, sales and deployment configuration, account information, and real trading data. See [BOOK_COVERAGE.md](BOOK_COVERAGE.md) for the mapping and review notes.
