from __future__ import annotations

import logging
from typing import Any

from lab_sim.db.repository import LedgerRepository
from lab_sim.utils.types import AgentState, TrustVector


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class TrustSystem:
    def __init__(self, repo: LedgerRepository, mode: str) -> None:
        self.repo = repo
        self.mode = mode
        self._mode_set = set(p.strip() for p in mode.split("+") if p.strip())
        self.logger = logging.getLogger("lab_sim.trust")

    def get_score(self, run_id: str, from_agent: str, to_agent: str) -> float:
        if "no_trust" in self._mode_set:
            return 0.5
        return self.repo.get_trust(run_id, from_agent, to_agent).score()

    def get_trust_vector(self, run_id: str, from_agent: str, to_agent: str) -> TrustVector:
        if "no_trust" in self._mode_set:
            return TrustVector()
        return self.repo.get_trust(run_id, from_agent, to_agent)

    def update_from_outcome(
        self,
        run_id: str,
        from_agent: str,
        to_agent: str,
        tick: int,
        action: str,
        outcome: str,
        verified: bool,
    ) -> None:
        if "no_trust" in self._mode_set:
            return
        trust = self.repo.get_trust(run_id, from_agent, to_agent)

        # Stronger penalties than before — being wrong must hurt
        if outcome in {"verification_refuted", "accusation_valid"}:
            trust.truthfulness = clamp(trust.truthfulness - 0.20)
            trust.consistency = clamp(trust.consistency - 0.14)
            trust.benevolence = clamp(trust.benevolence - 0.08)
            trust.betrayal_count += 1
        elif outcome in {"claim_distorted", "claim_shared_false"}:
            trust.truthfulness = clamp(trust.truthfulness - 0.12)
            trust.consistency = clamp(trust.consistency - 0.08)
        elif outcome in {"verification_confirmed", "defense_supported"}:
            trust.truthfulness = clamp(trust.truthfulness + 0.10)
            trust.consistency = clamp(trust.consistency + 0.07)
            trust.competence = clamp(trust.competence + 0.05)
        elif outcome == "accusation_false":
            trust.benevolence = clamp(trust.benevolence - 0.14)
            trust.competence = clamp(trust.competence - 0.10)
        elif outcome == "alliance_formed":
            trust.benevolence = clamp(trust.benevolence + 0.12)
            trust.consistency = clamp(trust.consistency + 0.06)
            trust.alliance_strength = clamp(trust.alliance_strength + 0.20)
        elif outcome == "alliance_dissolved":
            trust.alliance_strength = 0.0
            trust.benevolence = clamp(trust.benevolence - 0.10)
        elif verified:
            trust.truthfulness = clamp(trust.truthfulness + 0.04)

        if action == "seek_alliance" and outcome == "alliance_rejected":
            trust.benevolence = clamp(trust.benevolence - 0.04)

        trust.updated_tick = tick
        self.repo.upsert_trust(run_id, from_agent, to_agent, trust)

    # ---- Trust contagion ----

    def propagate_contagion(
        self,
        run_id: str,
        liar_agent: str,
        tick: int,
        all_agents: dict[str, AgentState],
    ) -> list[str]:
        """When a liar is caught, all agents who trust them suffer contagion damage.

        Returns list of agent names affected by contagion.
        """
        if "no_trust" in self._mode_set:
            return []

        affected: list[str] = []
        for agent_name, state in all_agents.items():
            if agent_name == liar_agent or not state.alive:
                continue
            trust = self.repo.get_trust(run_id, agent_name, liar_agent)
            trust_score = trust.score()

            # Only agents who actually trust the liar are affected
            if trust_score > 0.55:
                contagion_intensity = (trust_score - 0.55) * 2.0  # 0..0.9 range
                state.contagion_risk = clamp(state.contagion_risk + 0.08 * contagion_intensity)
                state.credibility = clamp(state.credibility - 0.04 * contagion_intensity)
                state.resources = clamp(state.resources - 0.05 * contagion_intensity)
                state.reputation = clamp(state.reputation - 0.03 * contagion_intensity)
                affected.append(agent_name)

                self.logger.debug(
                    "Trust contagion: %s affected by liar %s (trust=%.2f, intensity=%.2f)",
                    agent_name,
                    liar_agent,
                    trust_score,
                    contagion_intensity,
                )

        return affected

    # ---- Trust decay ----

    def decay_all_trust(self, run_id: str, agents: list[str], tick: int) -> None:
        """All trust edges decay slightly each tick if not reinforced."""
        if "no_trust" in self._mode_set:
            return
        for a in agents:
            for b in agents:
                if a == b:
                    continue
                trust = self.repo.get_trust(run_id, a, b)
                # Only decay if not recently updated (within last 3 ticks)
                if tick - trust.updated_tick > 3:
                    trust.truthfulness = clamp(trust.truthfulness - 0.008)
                    trust.consistency = clamp(trust.consistency - 0.005)
                    trust.benevolence = clamp(trust.benevolence - 0.003)
                    trust.alliance_strength = clamp(trust.alliance_strength - 0.01)
                    # Don't update updated_tick so passive decay continues
                    self.repo.upsert_trust(run_id, a, b, trust)

    # ---- Alliance management ----

    def form_alliance(
        self,
        run_id: str,
        agent_a: str,
        agent_b: str,
        tick: int,
        states: dict[str, AgentState],
    ) -> bool:
        """Attempt to form an alliance. Returns True if successful."""
        if "no_trust" in self._mode_set:
            return False

        # Already allies?
        state_a = states[agent_a]
        state_b = states[agent_b]
        if agent_b in state_a.alliances:
            return False

        trust_ab = self.get_score(run_id, agent_a, agent_b)
        trust_ba = self.get_score(run_id, agent_b, agent_a)

        # Both must have minimum trust AND credibility
        if trust_ab < 0.45 or trust_ba < 0.40:
            return False
        if state_b.credibility < 0.3 or state_b.contagion_risk > 0.5:
            return False

        # Form alliance
        state_a.alliances.append(agent_b)
        state_b.alliances.append(agent_a)
        self.repo.upsert_alliance(run_id, agent_a, agent_b, tick, 0.5)
        return True

    def dissolve_alliance(
        self,
        run_id: str,
        agent_a: str,
        agent_b: str,
        tick: int,
        states: dict[str, AgentState],
        reason: str = "betrayal",
    ) -> None:
        """Dissolve an alliance between two agents."""
        state_a = states[agent_a]
        state_b = states[agent_b]

        if agent_b in state_a.alliances:
            state_a.alliances.remove(agent_b)
        if agent_a in state_b.alliances:
            state_b.alliances.remove(agent_a)

        self.repo.upsert_alliance(run_id, agent_a, agent_b, tick, 0.0, dissolved_tick=tick)

        # Betrayal penalty
        if reason == "betrayal":
            state_a.stress = clamp(state_a.stress + 0.10)
            state_a.belonging = clamp(state_a.belonging - 0.12)
            state_b.stress = clamp(state_b.stress + 0.08)
            state_b.belonging = clamp(state_b.belonging - 0.08)

        self.update_from_outcome(
            run_id, agent_a, agent_b, tick,
            action="alliance_dissolved", outcome="alliance_dissolved", verified=False,
        )

    def check_alliance_stability(
        self,
        run_id: str,
        tick: int,
        states: dict[str, AgentState],
        rng,
    ) -> list[tuple[str, str]]:
        """Check all alliances and dissolve unstable ones. Returns dissolved pairs."""
        dissolved: list[tuple[str, str]] = []
        checked: set[tuple[str, str]] = set()

        for agent_name, state in states.items():
            if not state.alive:
                continue
            for ally in list(state.alliances):
                pair = tuple(sorted([agent_name, ally]))
                if pair in checked:
                    continue
                checked.add(pair)

                ally_state = states.get(ally)
                if ally_state is None or not ally_state.alive:
                    self.dissolve_alliance(run_id, agent_name, ally, tick, states, reason="death")
                    dissolved.append(pair)
                    continue

                # Dissolve if partner's credibility collapsed
                if ally_state.credibility < 0.25:
                    self.dissolve_alliance(run_id, agent_name, ally, tick, states, reason="betrayal")
                    dissolved.append(pair)
                    continue

                # Dissolve if partner's contagion risk too high
                if ally_state.contagion_risk > 0.55:
                    if rng.random() < 0.4:
                        self.dissolve_alliance(run_id, agent_name, ally, tick, states, reason="contagion")
                        dissolved.append(pair)
                        continue

        return dissolved

    def bootstrap_neutral_edges(self, run_id: str, agents: list[str]) -> None:
        if "no_trust" in self._mode_set:
            return
        neutral = TrustVector()
        for a in agents:
            for b in agents:
                if a == b:
                    continue
                self.repo.upsert_trust(run_id, a, b, neutral)
