# Exact printed source inventory

This directory archives code and configuration blocks, not book prose. The index covers all 711 fenced blocks across three editions (237 per edition), including all 321 Python blocks. Identical blocks share an artifact. Files end in `.txt` intentionally: an archived block is not necessarily an executable program. Korean comment/docstring variants are preserved exactly.

For each edition, 34 of 107 Python blocks map to runnable files (33 distinct files, including the shared HMM helper). The other 73 are reference fragments, including five function/loop bodies that cannot compile independently. Some are intentional unsafe counterexamples. Do not execute every artifact or interpret source inclusion as completion of an external integration.

Run `python verify_book_sources.py` to check all hashes, Python parsing, context classifications and runnable AST mappings. Run `python verify.py` for chapter execution and regression tests. `tests/test_book_sources.py` executes selected printed functions with controlled collaborators; untested fragments are not labeled runtime-verified.

`index.json` contains edition, manuscript chapter, fence number, Python block number, source line, exact artifact path/hash, runnable file, classification and standalone compilation result. The full manuscripts are not needed to run the repository checks.

| English chapter (same Python counts in all editions) | Python blocks | Runnable mappings | Other fragments |
|---|---:|---:|---:|
| 00-preface.md | 1 | 0 | 1 |
| 00a-vibe-coding-quickstart.md | 1 | 1 | 0 |
| 01-architecture.md | 4 | 1 | 3 |
| 02-data-pipeline.md | 6 | 1 | 5 |
| 03-backtest-engine.md | 3 | 3 | 0 |
| 04-hmm-basics.md | 7 | 2 | 5 |
| 05-hmm-forward-only.md | 6 | 1 | 5 |
| 06-kalman-filter.md | 5 | 1 | 4 |
| 07-cross-asset-joint-hmm.md | 2 | 1 | 1 |
| 08-two-layer-hmm.md | 2 | 1 | 1 |
| 09-kelly-sizing.md | 4 | 1 | 3 |
| 10-hmm-ensemble.md | 2 | 1 | 1 |
| 11-stat-arb-pairs.md | 6 | 1 | 5 |
| 12-stat-arb-signals.md | 2 | 1 | 1 |
| 13-rl-environment.md | 4 | 1 | 3 |
| 14-sac-agent.md | 2 | 1 | 1 |
| 15-optuna-tuning.md | 4 | 1 | 3 |
| 16-risk-management.md | 8 | 1 | 7 |
| 17-ibkr.md | 7 | 1 | 6 |
| 18-monitoring.md | 2 | 2 | 0 |
| 19-bamberger-pairs-theory.md | 1 | 1 | 0 |
| 20-pair-selection-regime.md | 2 | 2 | 0 |
| 21-regime-zscore-core.md | 1 | 1 | 0 |
| 22-hmm-pipeline-complete.md | 6 | 2 | 4 |
| 23-sac-rl-preview.md | 2 | 1 | 1 |
| 24-production-validation.md | 2 | 1 | 1 |
| 25-rl-algorithm-comparison.md | 1 | 1 | 0 |
| 26-distributional-rl-cvar.md | 7 | 1 | 6 |
| 27-mlflow-serving.md | 7 | 1 | 6 |
| 28-appendix-beginner-map.md | 0 | 0 | 0 |
