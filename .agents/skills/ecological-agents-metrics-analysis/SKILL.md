---
name: ecological-agents-metrics-analysis
description: Analyze EcologicalAgents run artifacts and aggregate metrics across modes and seeds from outputs/. Use when Codex needs to compare ablations, summarize intelligence/epistemic metrics, detect regressions, or produce report-ready tables from metrics.json and metrics.csv.
---

# Ecological Agents Metrics Analysis

## Overview

Aggregate experiment metrics by mode and compare outcomes across ablations with reproducible, script-based summaries.

## Workflow

1. Locate candidate run directories under `outputs/`.
2. Parse run IDs with pattern `<timestamp>_<mode>_seed<seed>`.
3. Load per-run `metrics.json` or fallback to top-level `metrics.csv`.
4. Aggregate by mode with mean, min, max, and run count.
5. Highlight key comparative metrics and suspicious outliers.

## Primary Metrics

Prioritize:

- `intelligence_proxy_score`
- `epistemic_survival_rate`
- `trust_calibration`
- `verification_intelligence`
- `false_belief_cost`
- `rumor_resistance`
- `compute_efficiency`
- `heuristic_fallback_rate`

Use directional interpretation carefully:

- Higher is usually better for scores and rates like `intelligence_proxy_score`.
- Lower is better for penalties/cost-like metrics such as `false_belief_cost` and `heuristic_fallback_rate`.

## Use Script

Run:

```bash
python .agents/skills/ecological-agents-metrics-analysis/scripts/summarize_metrics.py --outputs-dir outputs
```

Optional metric subset:

```bash
python .agents/skills/ecological-agents-metrics-analysis/scripts/summarize_metrics.py --outputs-dir outputs --metrics intelligence_proxy_score,epistemic_survival_rate,false_belief_cost
```

## Reporting Rules

- Report run counts per mode before comparing values.
- Flag missing modes or uneven seed coverage.
- Separate descriptive summary from causal claims.
- Treat mode differences as evidence, not proof, unless controls are clearly matched.

## References

Read when deeper analysis is needed:

- `references/analysis-playbook.md` for interpretation and guardrails.
