# Mapping the Examples to the Book

The initial source review compared 24 marked `book-example` blocks in each of the Korean, English, and Japanese editions: **72 blocks in total**, using Python abstract syntax trees (ASTs). All statements from 69 blocks were represented in the source. The other three were the same Chapter 27 example in each edition, adapted to use temporary local SQLite/artifact storage instead of an external MLflow server. Logged parameters, metrics, and files are preserved and checked by reading them back.

The initial English rewrite translates comments, docstrings, display labels, and documentation. A separate AST comparison against the initial release checked that executable structure and non-language constants were unchanged at v1.0.1. Subsequent correctness improvements intentionally change some executable statements; see REVIEW.md. The initial book comparison is provenance evidence; it is not a claim that translated string literals are byte-for-byte identical to the manuscripts.

These compact dummy-data exercises reproduce the chapters' key calculations. They do not turn every short architectural excerpt or exercise into a full production service. `manifest.json` maps scripts to manuscript filenames. The manuscripts are not included in this public repository.

| Scope | Implementation and review notes |
|---|---|
| 00, 03 | Cash, positions, and equity accounting from the book. Chapter 03 executes the preceding bar's signal at the next open and includes final liquidation costs. An additional test compares an opening gap and costs with hand-calculated values. |
| 02 | Includes the `compute_daily_features` definition preceding the runnable block. Scaling uses the training portion. |
| 04 | Demonstrates StandardScaler, a three-state full-covariance HMM, and covariance-trace ordering. Uses 60% for training and the remainder for OOS evaluation. |
| 05, 07 | Forward filtering without future data and the final posterior of each observation prefix. Additional tests check agreement under the same fixed model. |
| 06, 08–10, 15–17, 19–20, 24–26 | Runnable book blocks with dummy inputs, including Optuna holdout evaluation, duplicate fills, FDR, PER, and CVaR. |
| 11 | The book's `screen_pair` with synthetic log prices. Demonstrates pair screening for learning, not live selection. |
| 12 | Uses the book's five entry checks. Removes the duplicate `compute_entry_score` definition and rejects invalid half-lives, thresholds, correlations, and volume multipliers. |
| 13, 14, 23 | The book's 11-dimensional `TinyTradingEnv` and public Stable-Baselines3 SAC. Trains on in-memory dummy returns for 128 steps, then evaluates on 70 separate steps. Uses standard SAC rather than a complete custom algorithm. |
| 18 | Provides the monitor and FastAPI example together. TestClient checks `/health` and `/report`, so no server process is required. |
| 21 | Uses probability-weighted entry, exit, and stop thresholds; alpha-0 position sizing; and a 0.7 confidence gate. The default high-volatility policy blocks new entries. Unspecified transition-state exit/stop values are set to 0.3/9.9 for this exercise. Comparisons include their boundary values. New entries at or beyond their stop threshold are rejected. Demonstrates decisions without a pair-price or P&L model. |
| 22 | Implements four features, 60% training, a 0.65 probability gate, low-volatility mean reversion, and high-volatility momentum. Replaces market downloads with dummy closes and volume. States the close-fill assumption and trading costs. |
| 27 | Replaces the MLflow server address with temporary local storage. Reads back parameters, metrics, and artifact JSON content before cleanup. |
| 28 | Prints the appendix learning path and available scripts. |

Chapters 04 and 22 calculate probabilities with a fixed model after training. Training-period probabilities are not presented as OOS results available at that historical time. Leakage tests keep the model fixed and alter future observations, then verify that past features and filtering results remain unchanged.

GitHub Actions installs dependencies on Python 3.12 and runs `python verify.py`. This command blocks Python socket connections while running the examples and tests. See `VALIDATION.json` for the recorded checks.

## Current manuscript verification

The revised Korean, English, and Japanese manuscripts each contain 34 marked blocks, including two copies of the shared HMM helper. All 34 blocks match the reviewed public source by AST. The 32 distinct chapter exercise scripts (excluding the appendix file-list guide) ran successfully for every edition: 96 script runs and 150 regression test executions in total (50 per edition). Ten isolated architecture checks per edition cover Kelly sizing, zero gates, invalid sizing inputs, Sortino boundaries, drawdown, CVaR tail mass, metadata, risk tails, full-covariance filtering, and invalid order requests. Architecture excerpts are not standalone production applications. See `BOOK_VALIDATION.json`.

The book now includes the optional repository URL, complete helper definitions, local MLflow verification, and corrected source-review behavior. Reading and running the printed exercises does not require downloading this repository.

## Complete source inventory audit (2026-10-13)

The earlier reviews above executed the marked exercises and selected calculations; they did not cover every block as an executable program. The current [exact source inventory](book_sources/README.md) includes every fenced block from all three editions. Each edition has 107 Python blocks: 34 runnable mappings and 73 other fragments, of which five require an enclosing function or loop. Shell commands, configurations and diagrams are archived as text too. No claim is made that the architecture paths (`app/trading/...`) are complete production modules.

This audit corrected a Korean Chapter 15 objective that still referenced an undefined transaction-cost placeholder. All editions now calculate entry, reversal and final liquidation costs explicitly. A regression test checks known totals and exclusion of held-out returns directly from the printed blocks. Additional repository tests execute printed Kalman, half-life, ATR, Kelly and Sortino functions with controlled inputs. Source integrity and runtime validation remain distinct.

## Repository-first synchronization

The current workflow reverses the old manuscript-export direction. `book_contract.json` owns the mapping; runnable blocks read actual `.py` files, while explanatory fragments read canonical `book_snippets/` source. All three editions use identical Python code. `book_sources` is a derived snapshot. Source edits must be tested before `sync_book.py` updates the manuscript. Mismatched source anchors, unbound blocks and direct manuscript changes fail checks. The private book build checks a committed-source hash lock and canonical listing consistency before invoking the PDF compiler. This changes authorship and consistency controls; it does not turn architectural fragments into complete deployed features.

## Content and code boundary audit (2026-10-13)

The Chapter 11 runnable half-life calculation now uses the same exact discrete AR(1) equation as the printed explanation and Chapter 20. Invalid or degenerate screening inputs are rejected. The shared HMM filter preserves structural zero probabilities instead of making impossible states reachable. A new long-stop helper prevents a widening tier from lowering the previous stop. The IBKR connection design fragment returns failure if account initialization fails. The paper service identifies its forecast as log return per bar.

Six new regression cases exercise these boundaries; the repository suite now has 50 tests, and all 33 chapter examples pass with outbound connections blocked. Seven real local HTTP integration groups pass, including SQLite restart recovery and MLflow outage behavior. Live market retrieval and broker connectivity were not rerun in this audit. The three editions clarify exit triggers versus guaranteed fills, stop-state persistence, MongoDB write exceptions, and forecast units. This remains a learning implementation, with architectural excerpts explicitly distinguished from the runnable paper service.
