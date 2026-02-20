from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from lab_sim.db.repository import LedgerRepository


class MetricsEngine:
    def __init__(self, repo: LedgerRepository) -> None:
        self.repo = repo

    def compute(self, run_id: str) -> dict[str, float]:
        events = self.repo.get_events_for_run(run_id)
        trust_edges = self.repo.get_trust_edges_for_run(run_id)
        rumors = self.repo.get_rumors_for_run(run_id)
        alliances = self.repo.get_alliances_for_run(run_id)
        # Ecological state snapshots (may be empty if V4 migration not yet run)
        try:
            snapshots = self.repo.get_snapshots_for_run(run_id)
        except Exception:
            snapshots = []

        if not events:
            return {}

        total_actions = len(events)
        verify_actions = sum(
            1 for e in events
            if e["action"] in {"verify_claim", "verify_claim_resolved"}
        )
        verification_rate = verify_actions / total_actions

        # ---- False belief cost tracking ----
        false_belief_cost = 0.0
        false_belief_cost_series: list[float] = []
        for e in events:
            if e["outcome"] in {
                "claim_distorted",
                "claim_shared_false",
                "defense_backfired",
                "accusation_false",
            }:
                false_belief_cost += e["cost"] + 0.25
            false_belief_cost_series.append(false_belief_cost)

        half = max(1, len(false_belief_cost_series) // 2)
        early_avg = mean(false_belief_cost_series[:half]) if false_belief_cost_series else 0.0
        late_avg = mean(false_belief_cost_series[half:]) if false_belief_cost_series else 0.0
        false_belief_cost_reduction = max(0.0, early_avg - late_avg)

        # ---- Verification intelligence ----
        verification_success = sum(
            1 for e in events
            if e["outcome"] in {"verification_confirmed", "verification_refuted"}
        )
        verification_total = sum(
            1 for e in events
            if e["action"] in {"verify_claim", "verify_claim_resolved"}
        )
        verification_intelligence = verification_success / max(1, verification_total)

        # ---- Feedback discrimination ----
        feedback_discrimination = self._feedback_discrimination(events)

        # ---- Trust calibration ----
        trust_calibration = self._trust_calibration(events, trust_edges)

        # ---- Behavioral clustering ----
        behavioral_clustering = self._behavior_differentiation(events)

        # ---- Classic metrics ----
        identity_persistence = self._identity_persistence(events)
        emergent_social_structure = self._social_structure_strength(trust_edges)
        epistemic_stability = self._epistemic_stability(events)
        metacognitive_humility = self._metacognitive_humility(events)
        belief_revision_quality = self._belief_revision_quality(events)
        anti_rigidity = self._anti_rigidity(events)
        social_resilience = self._social_resilience(events)
        cognitive_collapse_rate = self._cognitive_collapse_rate(events)

        # ---- Epistemic survival metrics ----
        epistemic_survival_rate = self._epistemic_survival_rate(events)
        trust_contagion_damage = self._trust_contagion_damage(events)
        rumor_resistance = self._rumor_resistance(rumors, events)
        memory_advantage_score = self._memory_advantage_score(events)
        alliance_stability_index = self._alliance_stability_index(alliances, events)
        isolation_stress_impact = self._isolation_stress_impact(events)
        resource_efficiency = self._resource_efficiency(events)
        verification_roi = self._verification_roi(events)

        # ---- NEW: Ecological computational metrics ----
        compute_efficiency = self._compute_efficiency(events, snapshots)
        heuristic_fallback_rate = self._heuristic_fallback_rate(events, snapshots)
        avg_decision_latency_ms = self._avg_decision_latency_ms(snapshots)
        entropy_trajectory = self._entropy_trajectory(snapshots)
        tool_revocation_rate = self._tool_revocation_rate(snapshots)
        irreversible_damage_mean = self._irreversible_damage_mean(snapshots)
        compute_exhaustion_events = self._compute_exhaustion_events(snapshots)

        # ---- Intelligence proxy — now includes ecological dimensions ----
        intelligence_proxy_score = (
            epistemic_survival_rate * 0.15
            + trust_calibration * 0.12
            + verification_intelligence * 0.12
            + memory_advantage_score * 0.08
            + rumor_resistance * 0.07
            + belief_revision_quality * 0.08
            + alliance_stability_index * 0.06
            + resource_efficiency * 0.05
            + anti_rigidity * 0.04
            + metacognitive_humility * 0.04
            + behavioral_clustering * 0.04
            # ecological dimensions
            + compute_efficiency * 0.06
            + max(0.0, 1.0 - heuristic_fallback_rate) * 0.05
            + max(0.0, 1.0 - entropy_trajectory) * 0.04
        )
        intelligence_proxy_score = max(0.0, min(1.0, intelligence_proxy_score))

        return {
            # Core epistemic metrics
            "trust_calibration": round(trust_calibration, 6),
            "false_belief_cost": round(false_belief_cost, 6),
            "false_belief_cost_reduction": round(false_belief_cost_reduction, 6),
            "verification_rate": round(verification_rate, 6),
            "verification_intelligence": round(verification_intelligence, 6),
            "feedback_discrimination": round(feedback_discrimination, 6),
            # Cognitive metrics
            "metacognitive_humility": round(metacognitive_humility, 6),
            "belief_revision_quality": round(belief_revision_quality, 6),
            "anti_rigidity": round(anti_rigidity, 6),
            "social_resilience": round(social_resilience, 6),
            "cognitive_collapse_rate": round(cognitive_collapse_rate, 6),
            # Structure metrics
            "behavioral_clustering": round(behavioral_clustering, 6),
            "identity_persistence": round(identity_persistence, 6),
            "emergent_social_structure": round(emergent_social_structure, 6),
            "epistemic_stability": round(epistemic_stability, 6),
            # Epistemic survival metrics
            "epistemic_survival_rate": round(epistemic_survival_rate, 6),
            "trust_contagion_damage": round(trust_contagion_damage, 6),
            "rumor_resistance": round(rumor_resistance, 6),
            "memory_advantage_score": round(memory_advantage_score, 6),
            "alliance_stability_index": round(alliance_stability_index, 6),
            "isolation_stress_impact": round(isolation_stress_impact, 6),
            "resource_efficiency": round(resource_efficiency, 6),
            "verification_roi": round(verification_roi, 6),
            # NEW: Ecological computational metrics
            "compute_efficiency": round(compute_efficiency, 6),
            "heuristic_fallback_rate": round(heuristic_fallback_rate, 6),
            "avg_decision_latency_ms": round(avg_decision_latency_ms, 3),
            "entropy_trajectory": round(entropy_trajectory, 6),
            "tool_revocation_rate": round(tool_revocation_rate, 6),
            "irreversible_damage_mean": round(irreversible_damage_mean, 6),
            "compute_exhaustion_events": round(compute_exhaustion_events, 6),
            # Composite
            "intelligence_proxy_score": round(intelligence_proxy_score, 6),
        }

    def write_metrics_csv(
        self,
        output_dir: Path,
        rows: list[dict[str, Any]],
        filename: str = "metrics.csv",
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        if not rows:
            return path
        keys = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return path

    # ==================================================================
    #  ECOLOGICAL COMPUTATIONAL METRICS (new)
    # ==================================================================

    def _compute_efficiency(
        self, events: list[dict[str, Any]], snapshots: list[dict[str, Any]]
    ) -> float:
        """Average reward earned per unit of compute consumed.

        Higher = agent does more with each LLM call.
        """
        if not snapshots:
            # Fallback: use plain resource_efficiency if no snapshots yet
            return self._resource_efficiency(events)
        total_compute = sum(s["compute_budget"] for s in snapshots)
        if total_compute == 0:
            return 0.0
        total_reward = sum(e["reward"] for e in events)
        # Normalise: scale so 1.0 = perfectly efficient
        return min(1.0, total_reward / max(1.0, total_compute))

    def _heuristic_fallback_rate(
        self, events: list[dict[str, Any]], snapshots: list[dict[str, Any]]
    ) -> float:
        """Fraction of decisions resolved by heuristic (not LLM).

        Lower = agents used LLM reasoning more often.
        """
        if snapshots:
            total = len(snapshots)
            heuristic = sum(1 for s in snapshots if s["used_heuristic"])
            return heuristic / max(1, total)
        # Fallback: approximate from event payload
        total = len(events)
        if total == 0:
            return 0.0
        heuristic = sum(
            1 for e in events
            if (e.get("payload") or {}).get("used_heuristic", False)
        )
        return heuristic / total

    def _avg_decision_latency_ms(self, snapshots: list[dict[str, Any]]) -> float:
        """Mean wall-clock LLM call latency across the run, in milliseconds.

        Only non-zero entries (actual LLM calls) are averaged.
        """
        if not snapshots:
            return 0.0
        latencies = [s["decision_latency_ms"] for s in snapshots if s["decision_latency_ms"] > 0]
        return mean(latencies) if latencies else 0.0

    def _entropy_trajectory(self, snapshots: list[dict[str, Any]]) -> float:
        """Slope of memory_entropy averaged across agents over time.

        Positive = entropy rising (cognitive aging outpacing management).
        Negative = agents successfully managing entropy (isolation, good outcomes).
        Returned as a clamped [0, 1] signal where 0.5 = flat.
        """
        if len(snapshots) < 2:
            return 0.5
        by_agent: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for s in snapshots:
            by_agent[s["agent"]].append((s["tick"], s["memory_entropy"]))
        slopes: list[float] = []
        for series in by_agent.values():
            if len(series) < 2:
                continue
            series.sort()
            ticks = [t for t, _ in series]
            vals = [v for _, v in series]
            n = len(ticks)
            t_mean = mean(ticks)
            v_mean = mean(vals)
            num = sum((ticks[i] - t_mean) * (vals[i] - v_mean) for i in range(n))
            den = sum((ticks[i] - t_mean) ** 2 for i in range(n))
            slope = num / den if den > 1e-9 else 0.0
            slopes.append(slope)
        if not slopes:
            return 0.5
        avg_slope = mean(slopes)
        # Map slope to [0, 1]: 0 = strongly falling, 0.5 = flat, 1 = strongly rising
        return max(0.0, min(1.0, 0.5 + avg_slope * 5))

    def _tool_revocation_rate(self, snapshots: list[dict[str, Any]]) -> float:
        """Fraction of agents who had at least one tool revoked during the run."""
        if not snapshots:
            return 0.0
        agents_with_revocation: set[str] = set()
        agents_all: set[str] = set()
        for s in snapshots:
            agents_all.add(s["agent"])
            flags = s.get("tool_access_flags") or {}
            if any(not v for v in flags.values()):
                agents_with_revocation.add(s["agent"])
        if not agents_all:
            return 0.0
        return len(agents_with_revocation) / len(agents_all)

    def _irreversible_damage_mean(self, snapshots: list[dict[str, Any]]) -> float:
        """Mean irreversible_damage across all agents at their final snapshot."""
        if not snapshots:
            return 0.0
        # Take the last snapshot per agent
        latest: dict[str, float] = {}
        for s in snapshots:
            latest[s["agent"]] = s["irreversible_damage"]
        return mean(latest.values()) if latest else 0.0

    def _compute_exhaustion_events(self, snapshots: list[dict[str, Any]]) -> float:
        """Fraction of ticks where at least one agent was compute-exhausted.

        Normalised to [0, 1] (0 = never exhausted, 1 = exhausted every tick).
        """
        if not snapshots:
            return 0.0
        ticks_with_exhaustion: set[int] = set()
        all_ticks: set[int] = set()
        for s in snapshots:
            all_ticks.add(s["tick"])
            if s["fallback_reason"] == "compute_exhausted":
                ticks_with_exhaustion.add(s["tick"])
        if not all_ticks:
            return 0.0
        return len(ticks_with_exhaustion) / len(all_ticks)

    # ==================================================================
    #  NEW EPISTEMIC SURVIVAL METRICS
    # ==================================================================

    def _epistemic_survival_rate(self, events: list[dict[str, Any]]) -> float:
        """Percentage of agents still alive at end. Should be 40-70%, not 100%."""
        by_agent_alive: dict[str, bool] = {}
        for e in events:
            payload = e.get("payload") or {}
            by_agent_alive[e["agent"]] = bool(payload.get("alive", True))
        if not by_agent_alive:
            return 1.0
        alive = sum(1 for v in by_agent_alive.values() if v)
        return alive / len(by_agent_alive)

    def _trust_contagion_damage(self, events: list[dict[str, Any]]) -> float:
        """Total resource loss attributable to trust contagion."""
        total_contagion_damage = 0.0
        for e in events:
            payload = e.get("payload") or {}
            contagion = float(payload.get("contagion_risk", 0.0))
            if contagion > 0.1:
                # Events where contagion risk is elevated indicate contagion damage
                total_contagion_damage += contagion * e.get("cost", 0.0)
        return min(1.0, total_contagion_damage / max(1, len(events)) * 10)

    def _rumor_resistance(self, rumors: list[dict[str, Any]], events: list[dict[str, Any]]) -> float:
        """Fraction of rumors correctly rejected (not believed when false)."""
        if not rumors:
            return 0.0

        # Count rumors with high mutation (likely false) that were spread
        high_mutation = [r for r in rumors if r["mutation_degree"] > 0.5]
        low_mutation = [r for r in rumors if r["mutation_degree"] <= 0.5]

        if not high_mutation and not low_mutation:
            return 0.5

        # More verification actions relative to distorted claims = higher resistance
        verify_count = sum(
            1 for e in events
            if e["action"] in {"verify_claim", "verify_claim_resolved"}
        )
        distort_count = sum(1 for e in events if e["action"] == "distort_claim")
        broadcast_false = sum(1 for e in events if e["outcome"] == "claim_shared_false")

        total_bad = distort_count + broadcast_false
        if verify_count + total_bad == 0:
            return 0.5

        return verify_count / (verify_count + total_bad)

    def _memory_advantage_score(self, events: list[dict[str, Any]]) -> float:
        """Correlation between memory quality (successful verifications) and survival."""
        by_agent: dict[str, dict[str, Any]] = {}
        for e in events:
            payload = e.get("payload") or {}
            agent = e["agent"]
            if agent not in by_agent:
                by_agent[agent] = {
                    "verifications": 0,
                    "alive": True,
                    "resources": 1.0,
                }
            if e["outcome"] in {"verification_confirmed", "verification_refuted"}:
                by_agent[agent]["verifications"] += 1
            by_agent[agent]["alive"] = bool(payload.get("alive", True))
            by_agent[agent]["resources"] = float(payload.get("resources", 1.0))

        if len(by_agent) < 2:
            return 0.0

        # Do agents with more verifications survive better?
        agents = list(by_agent.values())
        avg_verif = mean(a["verifications"] for a in agents)
        if avg_verif == 0:
            return 0.0

        high_verif = [a for a in agents if a["verifications"] > avg_verif]
        low_verif = [a for a in agents if a["verifications"] <= avg_verif]

        if not high_verif or not low_verif:
            return 0.5

        high_survival = mean(1.0 if a["alive"] else 0.0 for a in high_verif)
        low_survival = mean(1.0 if a["alive"] else 0.0 for a in low_verif)

        high_resources = mean(a["resources"] for a in high_verif)
        low_resources = mean(a["resources"] for a in low_verif)

        # Score: how much better do verifiers do?
        survival_delta = max(0, high_survival - low_survival)
        resource_delta = max(0, high_resources - low_resources)

        return min(1.0, survival_delta * 0.6 + resource_delta * 0.4)

    def _alliance_stability_index(
        self, alliances: list[dict[str, Any]], events: list[dict[str, Any]]
    ) -> float:
        """Average alliance duration relative to total simulation length."""
        if not alliances or not events:
            return 0.0

        max_tick = max(e["tick"] for e in events)
        if max_tick == 0:
            return 0.0

        durations: list[float] = []
        for a in alliances:
            formed = a["formed_tick"]
            dissolved = a["dissolved_tick"] if a["dissolved_tick"] is not None else max_tick
            duration = dissolved - formed
            durations.append(duration / max_tick)

        return mean(durations) if durations else 0.0

    def _isolation_stress_impact(self, events: list[dict[str, Any]]) -> float:
        """Average cognitive degradation from isolation (measured by stress increase
        in agents with high isolation ticks)."""
        isolated_stress_deltas: list[float] = []
        by_agent: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            by_agent[e["agent"]].append(e)

        for agent_events in by_agent.values():
            for i in range(1, len(agent_events)):
                prev = agent_events[i - 1].get("payload") or {}
                curr = agent_events[i].get("payload") or {}
                iso_ticks = int(curr.get("isolation_ticks", 0))
                if iso_ticks >= 3:
                    prev_stress = float(prev.get("stress", 0.5))
                    curr_stress = float(curr.get("stress", 0.5))
                    delta = curr_stress - prev_stress
                    if delta > 0:
                        isolated_stress_deltas.append(delta)

        if not isolated_stress_deltas:
            return 0.0
        return min(1.0, mean(isolated_stress_deltas) * 5)

    def _resource_efficiency(self, events: list[dict[str, Any]]) -> float:
        """Ratio of reward earned to cost incurred across all events."""
        total_reward = sum(e["reward"] for e in events)
        total_cost = sum(e["cost"] for e in events)
        if total_cost == 0:
            return 1.0
        return min(1.0, total_reward / total_cost)

    def _verification_roi(self, events: list[dict[str, Any]]) -> float:
        """Return on investment for verification actions."""
        verify_events = [
            e for e in events
            if e["action"] in {"verify_claim", "verify_claim_resolved"}
        ]
        if not verify_events:
            return 0.0

        total_reward = sum(e["reward"] for e in verify_events)
        total_cost = sum(e["cost"] for e in verify_events)
        if total_cost == 0:
            return 0.5
        return min(1.0, total_reward / total_cost)

    # ==================================================================
    #  EXISTING METRICS (preserved)
    # ==================================================================

    def _trust_calibration(
        self, events: list[dict[str, Any]], trust_edges: list[dict[str, Any]]
    ) -> float:
        trust_observed: dict[tuple[str, str], list[int]] = defaultdict(list)
        for e in events:
            payload = e["payload"] or {}
            target = payload.get("target")
            if e["action"] not in {"verify_claim", "verify_claim_resolved"} or target is None:
                continue
            if e["outcome"] == "verification_confirmed":
                trust_observed[(e["agent"], target)].append(1)
            elif e["outcome"] == "verification_refuted":
                trust_observed[(e["agent"], target)].append(0)

        errors = []
        for edge in trust_edges:
            key = (edge["from_agent"], edge["to_agent"])
            if key not in trust_observed:
                continue
            observed_truthfulness = mean(trust_observed[key])
            expected = (
                edge["truthfulness"] * 0.35
                + edge["consistency"] * 0.25
                + edge["benevolence"] * 0.2
                + edge["competence"] * 0.2
            )
            errors.append(abs(expected - observed_truthfulness))
        trust_calibration = 1.0 - (mean(errors) if errors else 0.5)
        return max(0.0, min(1.0, trust_calibration))

    def _behavior_differentiation(self, events: list[dict[str, Any]]) -> float:
        by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for e in events:
            by_agent[e["agent"]][e["action"]] += 1
        if not by_agent:
            return 0.0

        signatures = []
        for _, counts in by_agent.items():
            total = sum(counts.values())
            signatures.append(
                [
                    counts.get("broadcast_claim", 0) / total,
                    counts.get("distort_claim", 0) / total,
                    counts.get("verify_claim", 0) / total,
                    counts.get("accuse_liar", 0) / total,
                    counts.get("defend_ally", 0) / total,
                    counts.get("seek_alliance", 0) / total,
                    counts.get("isolate", 0) / total,
                ]
            )

        centroid = [mean(x[i] for x in signatures) for i in range(len(signatures[0]))]
        spread = 0.0
        for sig in signatures:
            spread += sum((sig[i] - centroid[i]) ** 2 for i in range(len(sig))) ** 0.5
        return min(1.0, spread / max(1, len(signatures)))

    def _identity_persistence(self, events: list[dict[str, Any]]) -> float:
        per_agent: dict[str, list[dict[str, float]]] = defaultdict(list)
        for e in events:
            payload = e.get("payload") or {}
            ident = payload.get("identity")
            if isinstance(ident, dict):
                per_agent[e["agent"]].append(
                    {
                        "epistemic_need": float(ident.get("epistemic_need", 0.5)),
                        "identity_need": float(ident.get("identity_need", 0.5)),
                        "verification_tendency": float(ident.get("verification_tendency", 0.5)),
                        "social_need": float(ident.get("social_need", 0.5)),
                        "deception_tendency": float(ident.get("deception_tendency", 0.5)),
                        "alliance_bias": float(ident.get("alliance_bias", 0.5)),
                    }
                )
        if not per_agent:
            return 0.0

        scores = []
        for traces in per_agent.values():
            if len(traces) < 2:
                continue
            start, end = traces[0], traces[-1]
            dist = (
                (start["epistemic_need"] - end["epistemic_need"]) ** 2
                + (start["identity_need"] - end["identity_need"]) ** 2
                + (start["verification_tendency"] - end["verification_tendency"]) ** 2
                + (start["social_need"] - end["social_need"]) ** 2
                + (start["deception_tendency"] - end["deception_tendency"]) ** 2
                + (start["alliance_bias"] - end["alliance_bias"]) ** 2
            ) ** 0.5
            scores.append(max(0.0, 1.0 - dist))
        return mean(scores) if scores else 0.0

    def _social_structure_strength(self, trust_edges: list[dict[str, Any]]) -> float:
        if not trust_edges:
            return 0.0
        scores = []
        for edge in trust_edges:
            scores.append(
                edge["truthfulness"] * 0.35
                + edge["consistency"] * 0.25
                + edge["benevolence"] * 0.2
                + edge["competence"] * 0.2
            )
        avg = mean(scores)
        concentration = mean(abs(s - avg) for s in scores)
        return max(0.0, min(1.0, avg * 0.7 + concentration * 0.3))

    def _epistemic_stability(self, events: list[dict[str, Any]]) -> float:
        confusion_vals: list[float] = []
        for e in events:
            payload = e.get("payload") or {}
            if "confusion" in payload:
                confusion_vals.append(float(payload["confusion"]))
        if not confusion_vals:
            return 0.0
        return max(0.0, min(1.0, 1.0 - mean(confusion_vals)))

    def _feedback_discrimination(self, events: list[dict[str, Any]]) -> float:
        informative = {
            "verification_confirmed",
            "verification_refuted",
            "accusation_valid",
            "defense_supported",
        }
        noisy = {"accusation_false", "defense_backfired", "verification_no_claim", "verification_failed"}
        informative_count = sum(1 for e in events if e["outcome"] in informative)
        noisy_count = sum(1 for e in events if e["outcome"] in noisy)
        total = informative_count + noisy_count
        if total == 0:
            return 0.0
        return informative_count / total

    def _metacognitive_humility(self, events: list[dict[str, Any]]) -> float:
        """When confusion is high, agents should prefer caution over confident distortion."""
        cautious = {"verify_claim", "isolate", "seek_alliance"}
        overconfident = {"distort_claim", "accuse_liar"}
        high_confusion_events = [
            e
            for e in events
            if float((e.get("payload") or {}).get("confusion", 0.0)) >= 0.65
        ]
        if not high_confusion_events:
            return 0.0
        cautious_count = sum(1 for e in high_confusion_events if e["action"] in cautious)
        overconfident_count = sum(
            1 for e in high_confusion_events if e["action"] in overconfident
        )
        denom = cautious_count + overconfident_count
        if denom == 0:
            return 0.5
        return cautious_count / denom

    def _belief_revision_quality(self, events: list[dict[str, Any]]) -> float:
        """After being wrong, does an agent move toward verification instead of repeat distortion?"""
        by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            by_agent[e["agent"]].append(e)

        scores: list[float] = []
        for history in by_agent.values():
            score = 0.0
            count = 0
            for idx, e in enumerate(history):
                if e["outcome"] not in {"claim_shared_false", "claim_distorted", "accusation_false"}:
                    continue
                followups = history[idx + 1 : idx + 4]
                if not followups:
                    continue
                count += 1
                if any(f["action"] in {"verify_claim", "verify_claim_resolved"} for f in followups):
                    score += 1.0
                elif any(f["action"] == "distort_claim" for f in followups):
                    score += 0.0
                else:
                    score += 0.5
            if count > 0:
                scores.append(score / count)
        return mean(scores) if scores else 0.0

    def _anti_rigidity(self, events: list[dict[str, Any]]) -> float:
        """High if action distribution is not collapsed into a single repetitive behavior."""
        by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for e in events:
            by_agent[e["agent"]][e["action"]] += 1
        if not by_agent:
            return 0.0
        scores = []
        for counts in by_agent.values():
            total = sum(counts.values())
            max_share = max(v / total for v in counts.values())
            scores.append(1.0 - max_share)
        return mean(scores) if scores else 0.0

    def _social_resilience(self, events: list[dict[str, Any]]) -> float:
        values: list[float] = []
        for e in events:
            payload = e.get("payload") or {}
            belonging = float(payload.get("belonging", 0.5))
            isolation = float(payload.get("isolation", 0.5))
            values.append(max(0.0, min(1.0, belonging * 0.7 + (1.0 - isolation) * 0.3)))
        return mean(values) if values else 0.0

    def _cognitive_collapse_rate(self, events: list[dict[str, Any]]) -> float:
        by_agent_alive: dict[str, bool] = {}
        for e in events:
            payload = e.get("payload") or {}
            by_agent_alive[e["agent"]] = bool(payload.get("alive", True))
        if not by_agent_alive:
            return 0.0
        collapsed = sum(1 for alive in by_agent_alive.values() if not alive)
        return collapsed / len(by_agent_alive)
