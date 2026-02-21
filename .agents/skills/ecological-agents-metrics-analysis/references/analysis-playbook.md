# Analysis Playbook

## Input Contracts

- Preferred source: per-run `outputs/<run_id>/metrics.json`.
- Secondary source: `outputs/metrics.csv`.
- Run ID format: `<timestamp>_<mode>_seed<seed>`.

## Minimum Validation

1. Ensure every mode has at least one run.
2. Ensure mode comparisons use equivalent seed sets when possible.
3. Ensure key metrics exist before ranking modes.

## Suggested Comparison Slice

1. Rank modes by `intelligence_proxy_score` (descending).
2. Check whether top modes also improve `epistemic_survival_rate` and `verification_intelligence`.
3. Examine costs and degradation metrics (`false_belief_cost`, `heuristic_fallback_rate`).
4. Note tradeoffs explicitly rather than assuming universal dominance.

## Common Failure Patterns

- Missing run directories from interrupted execution.
- Mixed runs from incompatible parameter sets in the same `outputs/` root.
- Overinterpreting tiny deltas without confidence estimates.
