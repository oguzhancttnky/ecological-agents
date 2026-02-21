#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_run_id(run_id: str) -> tuple[str, int] | None:
    marker = "_seed"
    if marker not in run_id:
        return None
    prefix, seed_raw = run_id.rsplit(marker, 1)
    try:
        seed = int(seed_raw)
    except ValueError:
        return None
    parts = prefix.split("_", 1)
    if len(parts) != 2:
        return None
    _timestamp, mode = parts
    if not mode:
        return None
    return mode, seed


def load_rows(outputs_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(outputs_dir.glob("*/metrics.json")):
        run_id = path.parent.name
        parsed = parse_run_id(run_id)
        if not parsed:
            continue
        mode, seed = parsed
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        rows.append({"run_id": run_id, "mode": mode, "seed": seed, **payload})
    return rows


def to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize(rows: list[dict], metrics: list[str]) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    lines: list[str] = []
    lines.append("mode,runs," + ",".join(metrics))

    for mode in sorted(grouped):
        mode_rows = grouped[mode]
        values: list[str] = []
        for metric in metrics:
            vals = [to_float(r.get(metric)) for r in mode_rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                values.append("")
            else:
                values.append(f"{mean(vals):.6f}")
        lines.append(f"{mode},{len(mode_rows)}," + ",".join(values))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EcologicalAgents metrics by mode.")
    parser.add_argument("--outputs-dir", default="outputs", help="Root outputs directory")
    parser.add_argument(
        "--metrics",
        default="intelligence_proxy_score,epistemic_survival_rate,trust_calibration,verification_intelligence,false_belief_cost,rumor_resistance,compute_efficiency,heuristic_fallback_rate",
        help="Comma-separated metric names",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    rows = load_rows(outputs_dir)
    if not rows:
        raise SystemExit("No run metrics found under outputs directory.")

    print(summarize(rows, metrics))


if __name__ == "__main__":
    main()
