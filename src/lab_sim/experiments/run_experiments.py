from __future__ import annotations

import logging
import os

from lab_sim.config.settings import AppSettings
from lab_sim.experiments.runner import ExperimentRunner, build_specs, parse_seed_list


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("lab_sim.entrypoint")

    settings = AppSettings.from_env()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    modes_raw = os.getenv(
        "EXPERIMENT_MODES",
        "full_system,no_memory,no_trust,no_verification,no_uncertainty,no_social_pressure,no_identity_pressure,no_consequence,heuristic_only",
    )
    seeds_raw = os.getenv("EXPERIMENT_SEEDS", "11,42,97")
    modes = [m.strip() for m in modes_raw.split(",") if m.strip()]
    seeds = parse_seed_list(seeds_raw)
    logger.info(
        "Loaded experiment plan: modes=%s seeds=%s total_runs=%d",
        modes,
        seeds,
        len(modes) * len(seeds),
    )

    runner = ExperimentRunner(settings)
    try:
        specs = build_specs(modes, seeds)
        runner.run_many(specs)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
