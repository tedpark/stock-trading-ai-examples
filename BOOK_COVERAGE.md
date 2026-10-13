# Implementation-backed manuscript coverage

- All 107 Python blocks per edition originate in executable `.py` modules: 34 complete-file listings and 73 named function/class/region excerpts.
- Three editions share the same Python source. Shell and JSON listings also map to repository files through `command_contract.json`.
- `book_snippets` has been removed. Text archives in `book_sources` are generated evidence, never implementation inputs.
- `verify_book_sources.py` checks complete coverage, hashes, syntax, classifications and source equality. Missing, duplicated or substituted listings fail.
- `verify_runtime_sources.py` runs the regression suite and records observed callable execution for every excerpt in `RUNTIME_VALIDATION.json`. This does not measure every branch or prove financial performance.
- `quantlab` implements the additional data, HMM, filters, pairs, risk, training, paper broker and serving components. Tests import these modules rather than execute copied snippet text.
- Default exercises use synthetic data. The broker protocol provides local previews only. No live IBKR order adapter is supplied. SAC serving takes a compatible deterministic actor loader; it is not an HTTP deployment platform.
- Current book code uses the `runtime-book-2026-10-13` release tag. Update implementation and tests first, regenerate all editions, then rebuild PDFs.
