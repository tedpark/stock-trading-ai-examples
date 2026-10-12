# Source and Comment Review — 2026-10-12

The 33 scripts still demonstrate the book's chapter concepts with dummy data. The review covered the Python sources, helper modules, tests, comments, and chapter mapping. The initial manuscript AST comparison remains historical provenance, not a claim that this revision is textually identical to the book.

## Correctness improvements

- **Chapters 03/16/26:** CVaR now integrates the requested tail probability exactly within the equal-mass empirical distribution. A boundary observation receives fractional weight. For atoms `[-4, -2, 0, 2]` and alpha `0.375`, CVaR is `(-4 - 2*0.5)/1.5 = -3.3333`, rather than averaging the worst 50%. Chapter 03/16 also avoid including every tied boundary observation. The book's default Chapter 26 result remains `-0.35 0.5`.
- **Chapter 17:** Reject nonfinite, fractional, zero, and negative quantities before changing state. Repeated cancellation cannot reopen a cancelled order. Delayed execution events remain accounted for, including fills racing with cancellation; duplicate execution IDs are still ignored.
- **Chapter 21:** Do not enter at or beyond the stop threshold. Existing positions still check stops before exits, and both long and short boundary cases are tested.
- **Chapter 24:** Reject nonfinite numeric timestamps without advancing the last accepted timestamp.
- **Chapters 03/06/08/18/25:** Reject invalid numerical parameters in DSR, Kalman filtering, ensemble weights/Kelly multipliers, monitoring limits, and PER. PER computes its normalization totals once rather than once per element.
- **Chapter 27:** Verify the saved artifact's JSON contents as well as its filename, parameters, and metrics.

## Comment and teaching improvements

All new documentation, docstrings, and comments are in English. The review removed a stale instruction to append code already present in Chapter 02 and unused monitor imports. Comments explain units, assumptions, and boundary behavior instead of restating obvious assignments.

- HMM state ranks follow the book's covariance-trace rule over **standardized features**. These are not return-volatility ranks or evidence of mean reversion/trends. Chapter 22's trading rules are illustrative assumptions.
- DSR takes per-observation Sharpe ratios and excess kurtosis; mixing annualized and per-observation statistics is invalid.
- Risk metrics assume equally spaced equity observations and a zero risk-free rate. Signed loss VaR/CVaR may be negative when the lower return tail is positive.
- Chapter 26 treats equally spaced quantile predictions as equal-mass atoms. Fractional weighting fixes tail mass, but does not remove coarse-distribution approximation error or establish model calibration.
- The RL environment observes ten past returns and current exposure. Reward is a simple return minus turnover and final liquidation costs; summing rewards is not compounded portfolio performance. Training for 128 steps is an execution smoke test, not evidence of a profitable agent.
- PER importance weights are normalized over the full dummy buffer; minibatch normalization can differ.

The initial source review changed this repository. A subsequent manuscript review synchronized the runnable examples into all three editions and added the optional repository URL; see BOOK_VALIDATION.json for the checks. The documented refinements preserve the chapter learning objectives while improving behavior when readers vary the inputs.
