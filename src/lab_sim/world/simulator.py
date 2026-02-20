from __future__ import annotations

import asyncio
import logging
import json
import random
import time
from dataclasses import dataclass

from lab_sim.agents.cognition import AgentAction, AgentCognition
from lab_sim.config.settings import AppSettings
from lab_sim.db.repository import LedgerRepository
from lab_sim.engine.tick_engine import DecisionResult, TickEngine
from lab_sim.memory.vector_memory import VectorMemoryService
from lab_sim.trust.system import TrustSystem
from lab_sim.utils.types import AgentState, EventRecord


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class WorldOutcome:
    outcome: str
    reward: float
    cost: float
    verified: bool
    importance: float
    payload: dict


@dataclass
class Claim:
    claim_id: str
    text: str
    truth: bool
    ambiguity: float
    origin_tick: int
    origin_agent: str
    # Rumor propagation tracking
    spread_count: int = 0
    mutation_hops: int = 0
    virality: float = 0.5
    half_life: int = 20


@dataclass
class PendingVerification:
    """A verification request that resolves on the next tick."""
    verifier: str
    target: str
    claim_id: str
    request_tick: int
    energy_cost: float


class WorldSimulator:
    def __init__(
        self,
        run_id: str,
        mode: str,
        seed: int,
        settings: AppSettings,
        repo: LedgerRepository,
        memory: VectorMemoryService,
        trust_system: TrustSystem,
        cognition_factory,
    ) -> None:
        self.logger = logging.getLogger("lab_sim.world")
        self.run_id = run_id
        self.mode = mode
        self.seed = seed
        self.settings = settings
        self.repo = repo
        self.memory = memory
        self.trust_system = trust_system
        self._mode_set = set(p.strip() for p in mode.split("+") if p.strip())
        self.rng = random.Random(seed)
        self.inboxes: dict[str, list[str]] = {}
        self.claims: dict[str, Claim] = {}
        self.last_assertion_correct: dict[str, bool] = {}
        self.last_asserted_claim: dict[str, str] = {}
        self.pending_verifications: list[PendingVerification] = []
        self.states = self._init_agents()
        self.cognition: dict[str, AgentCognition] = {
            name: cognition_factory(name, st, self.rng) for name, st in self.states.items()
        }
        # 2-phase tick engine (async PLAN + deterministic ACT)
        self.tick_engine = TickEngine(
            cfg=settings.tick_engine,
            cognitions=self.cognition,
            mode_set=self._mode_set,
        )
        self.trust_system.bootstrap_neutral_edges(run_id, list(self.states.keys()))
        self.logger.info(
            "World initialized: run_id=%s mode=%s seed=%d agents=%d ticks=%d",
            self.run_id,
            self.mode,
            self.seed,
            self.settings.simulation.agent_count,
            self.settings.simulation.ticks,
        )

    def _init_agents(self) -> dict[str, AgentState]:
        out: dict[str, AgentState] = {}
        for i in range(self.settings.simulation.agent_count):
            name = f"agent_{i:02d}"
            out[name] = AgentState(
                name=name,
                epistemic_need=0.35 + self.rng.random() * 0.6,
                social_need=0.3 + self.rng.random() * 0.6,
                identity_need=0.35 + self.rng.random() * 0.6,
                stability_need=0.35 + self.rng.random() * 0.6,
                stress=self.rng.random() * 0.4,
                confusion=0.2 + self.rng.random() * 0.4,
                belonging=0.3 + self.rng.random() * 0.5,
                reputation=0.4 + self.rng.random() * 0.3,
                coherence=0.4 + self.rng.random() * 0.4,
                isolation=0.2 + self.rng.random() * 0.4,
                deception_tendency=0.1 + self.rng.random() * 0.5,
                verification_tendency=0.2 + self.rng.random() * 0.7,
                alliance_bias=0.2 + self.rng.random() * 0.7,
                epistemic_risk_tolerance=0.2 + self.rng.random() * 0.7,
                # Survival dimensions — all agents start equal
                resources=1.0,
                energy=1.0,
                survival_probability=1.0,
                credibility=0.5,
                goals=[
                    "survive by knowing what is true",
                    "avoid false beliefs that cost resources",
                    "build trustworthy alliances",
                    "verify claims before believing them",
                ],
            )
            self.inboxes[name] = []
        return out

    def run(self) -> None:
        ticks = self.settings.simulation.ticks
        for tick in range(1, ticks + 1):
            t_tick = time.perf_counter()
            action_counts = self._run_tick(tick)
            tick_elapsed = time.perf_counter() - t_tick
            self._maybe_log_tick_progress(tick, action_counts)
            self.logger.info(
                "TICK-END tick=%d elapsed=%.1fs actions=%s",
                tick, tick_elapsed,
                {k: v for k, v in action_counts.items() if v > 0},
            )

    def _run_tick(self, tick: int) -> dict[str, int]:
        _t0 = time.perf_counter()
        self.logger.info("TICK-START tick=%d alive=%d",
                         tick, sum(1 for s in self.states.values() if s.alive))

        def _phase(label: str) -> None:
            self.logger.info("TICK-PHASE tick=%d step=%s elapsed=%.1fs",
                             tick, label, time.perf_counter() - _t0)

        # 1. Resolve pending verifications from previous tick
        _phase("resolve_verifications")
        self._resolve_pending_verifications(tick)

        # 2. Spawn rumors into the world
        _phase("spawn_rumors")
        self._spawn_rumors(tick)

        # 3. Propagate existing rumors through trust network
        _phase("propagate_rumors")
        self._propagate_rumors(tick)

        # 4. Apply background survival pressure
        _phase("background_pressure")
        self._apply_background_pressure(tick)

        # 5. Regenerate compute budgets (energy metabolism reset per tick)
        self._regenerate_compute_budgets(tick)

        # 6. Check expired opportunity deadlines (temporal decay)
        self._check_opportunity_deadlines(tick)

        # 7. Check alliance stability
        self.trust_system.check_alliance_stability(
            self.run_id, tick, self.states, self.rng
        )

        # 8. Decay trust over time
        if tick % 5 == 0:
            alive_names = [n for n, s in self.states.items() if s.alive]
            self.trust_system.decay_all_trust(self.run_id, alive_names, tick)

        # 9. Apply inbox pressure before PLAN phase
        for name, state in self.states.items():
            if state.alive:
                self._apply_inbox_pressure(state, self.inboxes.get(name, []))

        action_counts: dict[str, int] = {
            "broadcast_claim": 0,
            "distort_claim": 0,
            "verify_claim": 0,
            "accuse_liar": 0,
            "defend_ally": 0,
            "seek_alliance": 0,
            "isolate": 0,
            "other": 0,
        }

        # === PHASE A — PLAN (concurrent async decision collection) ===
        _phase("PLAN")
        def peers_fn(name: str) -> list[str]:
            return [p for p in self.states.keys() if p != name and self.states[p].alive]

        decision_results = self.tick_engine.run_phases(
            tick=tick,
            run_id=self.run_id,
            states=self.states,
            inboxes=self.inboxes,
            peers_fn=peers_fn,
        )

        # === PHASE B — ACT (deterministic world update) ===
        _phase("ACT")
        # decision_results is already sorted by agent_name in TickEngine.run_phases()
        for dr in decision_results:
            name = dr.agent_name
            state = self.states.get(name)
            if state is None or not state.alive:
                continue
            peers = peers_fn(name)

            _t_agent = time.perf_counter()
            try:
                outcome = self._apply_action(tick, state, dr.action, peers)

                # Track action counts
                if dr.action.action in action_counts:
                    action_counts[dr.action.action] += 1
                else:
                    action_counts["other"] += 1

                # Post-action updates with ecological tracking
                self._post_action_updates(
                    name, tick, dr.action, outcome, dr
                )

                # Write state snapshot for ecological metrics
                self._write_state_snapshot(name, tick, state, dr)

                self.logger.debug(
                    "ACT agent=%-12s tick=%d action=%-20s outcome=%-30s elapsed=%.0fms",
                    name, tick, dr.action.action, outcome.outcome,
                    (time.perf_counter() - _t_agent) * 1000,
                )

            except Exception as exc:
                self.logger.error(
                    "Agent %s ACT exception during tick %d: %s",
                    name, tick, exc,
                    exc_info=True,
                )

        # 10. Clear inboxes
        for name in self.states:
            self.inboxes[name] = []

        # 11. Apply isolation tracking
        self._apply_isolation_tracking(tick)

        # 12. Re-evaluate tool access for all agents
        for state in self.states.values():
            if state.alive:
                self._gate_tool_access(state)

        return action_counts

    def _process_agent_decision(
        self, name: str, tick: int, peers: list[str]
    ) -> tuple[AgentAction, WorldOutcome]:
        """Process a single agent's decision. This runs in a separate thread."""
        state = self.states[name]
        inbox = self.inboxes[name]

        self._apply_inbox_pressure(state, inbox)

        action = self.cognition[name].decide(self.run_id, tick, inbox, peers)

        outcome = self._apply_action(tick, state, action, peers)

        return action, outcome

    # ======================================================================
    #  ACTION PROCESSING — Epistemic consequences are real
    # ======================================================================

    def _apply_action(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:

        if action.action == "broadcast_claim":
            return self._action_broadcast(tick, state, action, peers)
        if action.action == "distort_claim":
            return self._action_distort(tick, state, action, peers)
        if action.action == "verify_claim":
            return self._action_verify(tick, state, action, peers)
        if action.action == "accuse_liar":
            return self._action_accuse(tick, state, action, peers)
        if action.action == "defend_ally":
            return self._action_defend(tick, state, action, peers)
        if action.action == "seek_alliance":
            return self._action_seek_alliance(tick, state, action, peers)
        if action.action == "isolate":
            return self._action_isolate(tick, state, action, peers)

        return WorldOutcome(
            outcome="no_op", reward=0.0, cost=0.0,
            verified=False, importance=0.1, payload={},
        )

    def _action_broadcast(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        claim = self._pick_or_create_claim(tick, state.name)
        claimed_truth = claim.truth

        # Deception: might flip the truth
        if self.rng.random() < state.deception_tendency * 0.25:
            claimed_truth = not claim.truth

        target = action.target or self._random_peer(peers)
        message = self._serialize_claim_message(
            source=state.name,
            claim_id=claim.claim_id,
            claimed_truth=claimed_truth,
            confidence=max(0.05, 1.0 - claim.ambiguity),
        )
        self.inboxes[target].append(message)

        assertion_correct = claimed_truth == claim.truth
        self.last_assertion_correct[state.name] = assertion_correct
        self.last_asserted_claim[state.name] = claim.claim_id

        # Record rumor spread
        self.repo.append_rumor(
            self.run_id, claim.claim_id, state.name, target,
            tick, 0.0 if assertion_correct else 1.0, False,
        )
        claim.spread_count += 1

        state.social_need = _clamp(state.social_need - 0.06)

        if assertion_correct:
            state.coherence = _clamp(state.coherence + 0.03)
            state.credibility = _clamp(state.credibility + 0.02)
            reward, cost = 0.45, 0.15
        else:
            # FALSE BELIEF PENALTY — broadcasting false info costs you
            state.coherence = _clamp(state.coherence - 0.05)
            state.credibility = _clamp(state.credibility - 0.08)
            state.resources = _clamp(state.resources - 0.10)
            state.false_belief_count += 1
            state.survival_probability = _clamp(state.survival_probability * 0.95)
            reward, cost = 0.05, 0.40

        return WorldOutcome(
            outcome="claim_shared_true" if assertion_correct else "claim_shared_false",
            reward=reward,
            cost=cost,
            verified=assertion_correct,
            importance=0.75 if not assertion_correct else 0.55,
            payload={
                "target": target,
                "claim_id": claim.claim_id,
                "claim_truth": claim.truth,
                "claimed_truth": claimed_truth,
                "ambiguity": claim.ambiguity,
            },
        )

    def _action_distort(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        claim = self._pick_or_create_claim(tick, state.name)
        target = action.target or self._random_peer(peers)
        manipulated_truth = not claim.truth

        self.inboxes[target].append(
            self._serialize_claim_message(
                source=state.name,
                claim_id=claim.claim_id,
                claimed_truth=manipulated_truth,
                confidence=max(0.05, 1.0 - claim.ambiguity * 0.5),
            )
        )
        self.last_assertion_correct[state.name] = False
        self.last_asserted_claim[state.name] = claim.claim_id

        # Track rumor with high mutation
        self.repo.append_rumor(
            self.run_id, claim.claim_id, state.name, target,
            tick, 1.0, False,
        )
        claim.spread_count += 1
        claim.mutation_hops += 1

        # HEAVY PENALTY for deliberate distortion
        state.reputation = _clamp(state.reputation - 0.06)
        state.credibility = _clamp(state.credibility - 0.10)
        state.coherence = _clamp(state.coherence - 0.07)
        state.stress = _clamp(state.stress + 0.06)
        state.resources = _clamp(state.resources - 0.15)
        state.false_belief_count += 1
        state.survival_probability = _clamp(state.survival_probability * 0.93)

        return WorldOutcome(
            outcome="claim_distorted",
            reward=0.08,
            cost=0.65,
            verified=False,
            importance=0.90,
            payload={
                "target": target,
                "claim_id": claim.claim_id,
                "claim_truth": claim.truth,
                "claimed_truth": manipulated_truth,
                "ambiguity": claim.ambiguity,
            },
        )

    def _action_verify(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        if "no_verification" in self._mode_set:
            return WorldOutcome(
                outcome="verification_disabled",
                reward=0.0, cost=0.15,
                verified=False, importance=0.3,
                payload={"target": action.target},
            )

        target = action.target or self._random_peer(peers)
        claim_id = self.last_asserted_claim.get(target)

        # Verification COSTS energy — this is the strategic tension
        energy_cost = 0.12 + self.rng.random() * 0.08
        if state.energy < energy_cost:
            # Not enough energy to verify — forced to gamble
            state.stress = _clamp(state.stress + 0.04)
            return WorldOutcome(
                outcome="verification_no_energy",
                reward=0.0, cost=0.05,
                verified=False, importance=0.5,
                payload={"target": target, "energy": round(state.energy, 3)},
            )

        state.energy = _clamp(state.energy - energy_cost)

        if claim_id is None or claim_id not in self.claims:
            # Energy spent but nothing to verify
            state.resources = _clamp(state.resources - 0.03)
            return WorldOutcome(
                outcome="verification_no_claim",
                reward=0.0, cost=0.15,
                verified=False, importance=0.35,
                payload={"target": target},
            )

        # Queue for next-tick resolution (verification delay)
        self.pending_verifications.append(
            PendingVerification(
                verifier=state.name,
                target=target,
                claim_id=claim_id,
                request_tick=tick,
                energy_cost=energy_cost,
            )
        )

        return WorldOutcome(
            outcome="verification_pending",
            reward=0.0, cost=energy_cost,
            verified=False, importance=0.6,
            payload={
                "target": target,
                "claim_id": claim_id,
                "energy_cost": round(energy_cost, 3),
            },
        )

    def _action_accuse(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        target = action.target or self._random_peer(peers)
        accusation_valid = not self.last_assertion_correct.get(target, True)

        state.social_need = _clamp(state.social_need - 0.03)
        state.stress = _clamp(state.stress + 0.04)

        if accusation_valid:
            # Correct accusation — reward the accuser, punish the liar
            state.reputation = _clamp(state.reputation + 0.06)
            state.credibility = _clamp(state.credibility + 0.05)
            state.resources = _clamp(state.resources + 0.08)
            state.successful_verifications += 1

            # Punish the accused liar
            liar_state = self.states.get(target)
            if liar_state and liar_state.alive:
                liar_state.reputation = _clamp(liar_state.reputation - 0.12)
                liar_state.credibility = _clamp(liar_state.credibility - 0.10)
                liar_state.resources = _clamp(liar_state.resources - 0.18)
                liar_state.survival_probability = _clamp(liar_state.survival_probability * 0.90)

                # TRUST CONTAGION — everyone who trusted this liar suffers
                affected = self.trust_system.propagate_contagion(
                    self.run_id, target, tick, self.states
                )

                # Dissolve alliances with the liar — betrayal causes irreversible trauma
                for ally in list(liar_state.alliances):
                    self.trust_system.dissolve_alliance(
                        self.run_id, ally, target, tick, self.states, reason="betrayal"
                    )
                    # Permanent irreversible damage to the liar for each betrayal
                    self._apply_irreversible_mutation(liar_state, "alliance_betrayal", tick)

            reward, cost = 0.55, 0.18
        else:
            # FALSE ACCUSATION — severe penalty
            state.reputation = _clamp(state.reputation - 0.10)
            state.credibility = _clamp(state.credibility - 0.08)
            state.resources = _clamp(state.resources - 0.12)
            state.false_belief_count += 1
            state.survival_probability = _clamp(state.survival_probability * 0.94)
            reward, cost = 0.03, 0.55

        return WorldOutcome(
            outcome="accusation_valid" if accusation_valid else "accusation_false",
            reward=reward,
            cost=cost,
            verified=accusation_valid,
            importance=0.90 if accusation_valid else 0.75,
            payload={"target": target, "accusation_valid": accusation_valid},
        )

    def _action_defend(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        target = action.target or self._random_peer(peers)
        defended_honest = self.last_assertion_correct.get(target, True)

        state.belonging = _clamp(state.belonging + 0.08)
        state.social_need = _clamp(state.social_need - 0.07)

        if defended_honest:
            state.credibility = _clamp(state.credibility + 0.03)
            state.resources = _clamp(state.resources + 0.04)
            reward, cost = 0.40, 0.14
        else:
            # Defended a liar — contagion
            state.credibility = _clamp(state.credibility - 0.06)
            state.resources = _clamp(state.resources - 0.10)
            state.contagion_risk = _clamp(state.contagion_risk + 0.06)
            state.reputation = _clamp(state.reputation - 0.05)
            state.survival_probability = _clamp(state.survival_probability * 0.96)
            reward, cost = 0.02, 0.50

        return WorldOutcome(
            outcome="defense_supported" if defended_honest else "defense_backfired",
            reward=reward,
            cost=cost,
            verified=defended_honest,
            importance=0.72,
            payload={"target": target, "defended_honest": defended_honest},
        )

    def _action_seek_alliance(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        target = action.target or self._random_peer(peers)
        trust = self.trust_system.get_score(self.run_id, state.name, target)

        # Alliance formation through trust system (checks credibility, contagion)
        accepted = self.trust_system.form_alliance(
            self.run_id, state.name, target, tick, self.states,
        )

        if not accepted:
            # Fallback: probabilistic acceptance based on trust
            acceptance_prob = 0.15 + trust * 0.45
            target_state = self.states.get(target)
            if target_state:
                acceptance_prob *= (1.0 - target_state.contagion_risk)
                acceptance_prob *= target_state.credibility
            accepted = self.rng.random() < acceptance_prob

        state.belonging = _clamp(state.belonging + (0.12 if accepted else -0.06))
        state.isolation = _clamp(state.isolation - (0.10 if accepted else -0.04))
        state.stress = _clamp(state.stress - (0.06 if accepted else -0.02))

        if accepted:
            # Alliance benefits
            state.resources = _clamp(state.resources + 0.03)
            reward, cost = 0.45, 0.10
        else:
            reward, cost = 0.03, 0.30

        return WorldOutcome(
            outcome="alliance_formed" if accepted else "alliance_rejected",
            reward=reward,
            cost=cost,
            verified=accepted,
            importance=0.70,
            payload={"target": target, "accepted": accepted},
        )

    def _action_isolate(
        self, tick: int, state: AgentState, action: AgentAction, peers: list[str]
    ) -> WorldOutcome:
        # Isolation provides minor recovery but at social cost
        state.stress = _clamp(state.stress - 0.08)
        state.confusion = _clamp(state.confusion - 0.05)
        state.isolation = _clamp(state.isolation + 0.15)
        state.social_need = _clamp(state.social_need + 0.12)
        state.belonging = _clamp(state.belonging - 0.08)

        # Energy recovery from rest
        state.energy = _clamp(state.energy + 0.06)

        # Ecological: isolation = compression/summarization — entropy decays
        cfg = self.settings.tick_engine
        entropy_floor = state.irreversible_damage
        state.memory_entropy = max(
            entropy_floor,
            state.memory_entropy - cfg.entropy_decay_on_summarize,
        )

        return WorldOutcome(
            outcome="isolation_reflection",
            reward=0.15,
            cost=0.18,
            verified=False,
            importance=0.44,
            payload={"entropy_after": round(state.memory_entropy, 4)},
        )

    # ======================================================================
    #  VERIFICATION RESOLUTION — delayed results create strategic tension
    # ======================================================================

    def _resolve_pending_verifications(self, current_tick: int) -> None:
        """Resolve verifications queued in the previous tick."""
        resolved = []
        remaining = []

        for pv in self.pending_verifications:
            if pv.request_tick < current_tick:
                resolved.append(pv)
            else:
                remaining.append(pv)
        self.pending_verifications = remaining

        for pv in resolved:
            verifier_state = self.states.get(pv.verifier)
            if verifier_state is None or not verifier_state.alive:
                continue

            claim = self.claims.get(pv.claim_id)
            if claim is None:
                continue

            was_correct = self.last_assertion_correct.get(pv.target, False)

            # Compute memory advantage for verification success probability
            memory_advantage = self._compute_memory_advantage(pv.verifier)
            success_prob = (
                0.45
                + 0.30 * memory_advantage
                + 0.20 * verifier_state.verification_tendency
                + 0.05 * verifier_state.energy
            )

            if self.rng.random() < success_prob:
                # Verification succeeded
                verifier_state.confusion = _clamp(verifier_state.confusion - 0.14)
                verifier_state.stress = _clamp(verifier_state.stress - 0.08)
                verifier_state.reputation = _clamp(verifier_state.reputation + 0.05)
                verifier_state.credibility = _clamp(verifier_state.credibility + 0.04)
                verifier_state.resources = _clamp(verifier_state.resources + 0.10)
                verifier_state.successful_verifications += 1

                # Deliver verification result via inbox
                result_msg = json.dumps({
                    "type": "verification_result",
                    "source": "system",
                    "claim_id": pv.claim_id,
                    "target": pv.target,
                    "was_correct": was_correct,
                    "verified": True,
                })
                self.inboxes[pv.verifier].append(result_msg)

                # If target was lying, trigger contagion
                if not was_correct:
                    self.trust_system.propagate_contagion(
                        self.run_id, pv.target, current_tick, self.states
                    )

                outcome_str = "verification_confirmed" if was_correct else "verification_refuted"
                importance = 0.92
            else:
                # Verification failed — energy wasted, no info
                verifier_state.stress = _clamp(verifier_state.stress + 0.03)
                verifier_state.resources = _clamp(verifier_state.resources - 0.03)
                outcome_str = "verification_failed"
                importance = 0.45

            # Record the resolved verification as an event
            event = EventRecord(
                run_id=self.run_id,
                tick=current_tick,
                agent=pv.verifier,
                action="verify_claim_resolved",
                payload={
                    "target": pv.target,
                    "claim_id": pv.claim_id,
                    "was_correct": was_correct,
                    "memory_advantage": round(memory_advantage, 3),
                    "success_prob": round(success_prob, 3),
                    **verifier_state.survival_signature(),
                },
                outcome=outcome_str,
                reward=0.10 if outcome_str != "verification_failed" else 0.0,
                cost=pv.energy_cost,
                verified=outcome_str != "verification_failed",
                importance=importance,
            )
            self.repo.append_event(event)

            # Update trust
            if pv.target in self.states:
                self.trust_system.update_from_outcome(
                    run_id=self.run_id,
                    from_agent=pv.verifier,
                    to_agent=pv.target,
                    tick=current_tick,
                    action="verify_claim",
                    outcome=outcome_str,
                    verified=outcome_str != "verification_failed",
                )

    # ======================================================================
    #  RUMOR ECONOMY — uncertainty must persist
    # ======================================================================

    def _spawn_rumors(self, tick: int) -> None:
        """Inject new claims into the world. More claims than before."""
        if "no_uncertainty" in self._mode_set:
            claim_count = 1
        else:
            # 2-4 claims per tick
            claim_count = 2 + (1 if self.rng.random() < 0.5 else 0) + (1 if self.rng.random() < 0.25 else 0)

        for i in range(claim_count):
            origin = f"world_signal_{i}"
            if "no_uncertainty" in self._mode_set:
                truth = True
                ambiguity = 0.05
            else:
                truth = self.rng.random() > 0.45  # Slightly more false than true
                ambiguity = 0.15 + self.rng.random() * 0.75

            claim = self._create_claim(tick, origin, truth, ambiguity)

            # Distribute to 2-5 agents (more exposure = more pressure)
            recipient_count = min(2 + int(self.rng.random() * 3), len(self.states))
            recipients = self.rng.sample(list(self.states.keys()), k=recipient_count)

            for receiver in recipients:
                if not self.states[receiver].alive:
                    continue
                # Rumors can be injected with wrong truth value
                injected_false = (
                    self.rng.random() <= 0.40
                    and "no_uncertainty" not in self._mode_set
                )
                transmitted_truth = claim.truth if not injected_false else not claim.truth
                message = self._serialize_claim_message(
                    source=origin,
                    claim_id=claim.claim_id,
                    claimed_truth=transmitted_truth,
                    confidence=max(0.05, 1.0 - claim.ambiguity),
                )
                self.inboxes[receiver].append(message)
                self.repo.append_rumor(
                    self.run_id, claim.claim_id, origin, receiver,
                    tick, 0.0 if transmitted_truth == claim.truth else 1.0, False,
                )

    def _propagate_rumors(self, tick: int) -> None:
        """Active rumors spread through the trust network. Claims mutate as they spread."""
        if "no_uncertainty" in self._mode_set:
            return

        # Find recently active claims that can still spread
        active_claims = [
            c for c in self.claims.values()
            if (tick - c.origin_tick) < c.half_life
            and c.spread_count < len(self.states) * 2
        ]

        for claim in active_claims:
            if self.rng.random() > claim.virality:
                continue

            # Pick a random alive agent to spread from
            alive_agents = [n for n, s in self.states.items() if s.alive]
            if len(alive_agents) < 2:
                continue

            spreader = self.rng.choice(alive_agents)
            # Spread to a trusted contact
            best_target = None
            best_trust = -1.0
            candidates = [a for a in alive_agents if a != spreader]
            for candidate in self.rng.sample(candidates, k=min(3, len(candidates))):
                t = self.trust_system.get_score(self.run_id, spreader, candidate)
                if t > best_trust:
                    best_trust = t
                    best_target = candidate

            if best_target is None:
                continue

            # Mutation: truth can flip during propagation
            mutation_prob = 0.10 + 0.05 * claim.mutation_hops
            mutated = self.rng.random() < mutation_prob
            transmitted_truth = not claim.truth if mutated else claim.truth

            message = self._serialize_claim_message(
                source=spreader,
                claim_id=claim.claim_id,
                claimed_truth=transmitted_truth,
                confidence=max(0.05, 1.0 - claim.ambiguity * (1 + claim.mutation_hops * 0.1)),
            )
            self.inboxes[best_target].append(message)
            claim.spread_count += 1
            if mutated:
                claim.mutation_hops += 1

            self.repo.append_rumor(
                self.run_id, claim.claim_id, spreader, best_target,
                tick, claim.mutation_hops * 0.2, False,
            )

    # ======================================================================
    #  BACKGROUND PRESSURE — existence costs resources
    # ======================================================================

    def _apply_background_pressure(self, tick: int) -> None:
        """Every tick, agents face background survival drain."""
        consequence_off = "no_consequence" in self._mode_set

        for name, state in self.states.items():
            if not state.alive:
                continue

            if not consequence_off:
                # Background resource drain — existence costs resources
                state.resources = _clamp(state.resources - 0.018)

                # Energy regeneration (slow)
                state.energy = _clamp(state.energy + 0.03)

                # Alliance sharing: small resource drip per alliance
                for ally in state.alliances:
                    if ally in self.states and self.states[ally].alive:
                        state.resources = _clamp(state.resources + 0.025)

                # Contagion risk decays slowly if no new exposure
                state.contagion_risk = _clamp(state.contagion_risk - 0.01)

                # Background confusion/stress from world uncertainty
                state.confusion = _clamp(state.confusion + 0.025)
                state.stress = _clamp(state.stress + 0.018)
            else:
                # Minimal pressure in no_consequence mode
                state.confusion = _clamp(state.confusion + 0.005)
                state.stress = _clamp(state.stress + 0.004)

    def _apply_isolation_tracking(self, tick: int) -> None:
        """Track consecutive isolation ticks and apply progressive damage."""
        consequence_off = "no_consequence" in self._mode_set

        for name, state in self.states.items():
            if not state.alive:
                continue

            # Check if agent was socially active this tick
            had_social_activity = (
                name in self.last_asserted_claim
                or bool(self.inboxes.get(name))
            )

            # An agent who has no alliances and didn't interact is isolated
            if not state.alliances and state.isolation > 0.6 and not had_social_activity:
                state.isolation_ticks += 1
            else:
                # Partial reset — doesn't fully undo damage
                state.isolation_ticks = max(0, state.isolation_ticks - 1)

            if consequence_off:
                continue

            # Progressive isolation damage
            iso_ticks = state.isolation_ticks
            if iso_ticks >= 1:
                state.stress = _clamp(state.stress + 0.03 * min(iso_ticks, 5))
            if iso_ticks >= 4:
                state.confusion = _clamp(state.confusion + 0.05 * min(iso_ticks - 3, 4))
                state.energy = _clamp(state.energy - 0.03 * min(iso_ticks - 3, 4))
            if iso_ticks >= 8:
                state.resources = _clamp(state.resources - 0.04 * min(iso_ticks - 7, 3))
                state.survival_probability = _clamp(state.survival_probability * 0.96)

    # ======================================================================
    #  MEMORY ADVANTAGE — memory must influence survival
    # ======================================================================

    def _compute_memory_advantage(self, agent_name: str) -> float:
        """Agents with better memory (more verified memories) get decision advantage."""
        if "no_memory" in self._mode_set:
            return 0.0

        try:
            recalled = self.memory.retrieve(
                self.run_id, agent_name,
                "verified truthful reliable",
                tick=999999, limit=20,
            )
        except Exception:
            return 0.0

        if not recalled:
            return 0.0

        verified_count = sum(1 for m in recalled if m.verified)
        high_trust_count = sum(1 for m in recalled if m.trust_score > 0.6)
        total = len(recalled)

        advantage = (
            (verified_count / total) * 0.6
            + (high_trust_count / total) * 0.4
        )
        return min(1.0, advantage)

    # ======================================================================
    #  ECOLOGICAL SURVIVAL — compute, entropy, tools, irreversibility, time
    # ======================================================================

    def _regenerate_compute_budgets(self, tick: int) -> None:
        """Restore compute budget at start of each tick (biological energy intake)."""
        # no_compute_pressure: infinite budget, never exhausted
        if "no_compute_pressure" in self._mode_set:
            for state in self.states.values():
                if state.alive:
                    state.compute_budget = 1.0
                    state.compute_exhausted = False
            return

        cfg = self.settings.tick_engine
        for state in self.states.values():
            if not state.alive:
                continue
            state.compute_budget = min(1.0, state.compute_budget + cfg.compute_regen_per_tick)
            if state.compute_budget > cfg.compute_min_threshold:
                state.compute_exhausted = False

    def _check_opportunity_deadlines(self, tick: int) -> None:
        """Expire missed opportunities — indecision has a cost."""
        # no_deadlines: temporal pressure is disabled
        if "no_deadlines" in self._mode_set:
            for state in self.states.values():
                state.opportunity_deadlines = []
            return

        cfg = self.settings.tick_engine
        consequence_off = "no_consequence" in self._mode_set

        for state in self.states.values():
            if not state.alive:
                continue
            remaining = []
            for deadline_entry in state.opportunity_deadlines:
                claim_id, expiry_tick = deadline_entry
                if tick > expiry_tick:
                    # Missed window — resource penalty
                    if not consequence_off:
                        state.resources = _clamp(
                            state.resources - cfg.opportunity_missed_resource_penalty
                        )
                        state.stress = _clamp(state.stress + 0.03)
                    self.logger.debug(
                        "Opportunity expired agent=%s claim=%s tick=%d penalty=%.3f",
                        state.name, claim_id, tick,
                        cfg.opportunity_missed_resource_penalty,
                    )
                else:
                    remaining.append(deadline_entry)
            state.opportunity_deadlines = remaining

    def _gate_tool_access(self, state: AgentState) -> None:
        """Revoke or restore tool access based on reputation_score thresholds."""
        # no_tool_gating: all tools always available
        if "no_tool_gating" in self._mode_set:
            for tool in list(state.tool_access_flags):
                state.tool_access_flags[tool] = True
            return

        cfg = self.settings.tick_engine
        r = state.reputation_score

        # Each tool is gated by a threshold; revocation is instant, restoration slow
        def _gate(tool: str, threshold: float) -> None:
            currently_on = state.tool_access_flags.get(tool, True)
            if r < threshold and currently_on:
                state.tool_access_flags[tool] = False
                self.logger.info(
                    "Tool revoked: agent=%s tool=%s reputation=%.3f threshold=%.3f",
                    state.name, tool, r, threshold,
                )
            elif r >= threshold + 0.05 and not currently_on:
                # Hysteresis: need +5% above threshold to regain
                # … unless the tool was permanently revoked via irreversible_damage
                if state.irreversible_damage < 0.5:
                    state.tool_access_flags[tool] = True

        _gate("verification_tool", cfg.threshold_verification_tool)
        _gate("data_retrieval", cfg.threshold_data_retrieval)
        _gate("coalition_formation", cfg.threshold_coalition_formation)
        _gate("advanced_planning", cfg.threshold_advanced_planning)

    def _apply_irreversible_mutation(
        self, state: AgentState, event_type: str, tick: int
    ) -> None:
        """Apply a permanent, non-reversible capability degradation (no rollback)."""
        # no_irreversibility: permanent damage is disabled
        if "no_irreversibility" in self._mode_set:
            return
        consequence_off = "no_consequence" in self._mode_set
        if consequence_off:
            return

        damage_amount = cfg.entropy_floor_raise_per_trigger
        if event_type == "alliance_betrayal":
            damage_amount = cfg.irreversible_damage_per_betrayal

        # Raise the permanent floor
        state.irreversible_damage = min(1.0, state.irreversible_damage + damage_amount)

        # Entropy floor is now permanently elevated — next entropy clamp uses this
        floor = state.irreversible_damage
        state.memory_entropy = max(state.memory_entropy, floor)

        # Compute efficiency loss (10% per betrayal event)
        if event_type == "alliance_betrayal":
            state.compute_budget = max(0.0, state.compute_budget * 0.90)

        # Permanently revoke advanced_planning if damage is high enough
        if state.irreversible_damage >= 0.30 and event_type == "repeated_distortion":
            state.tool_access_flags["advanced_planning"] = False
            state.tool_access_flags["coalition_formation"] = False
            self.logger.info(
                "IRREVERSIBLE mutation: agent=%s event=%s damage=%.3f "
                "tools [advanced_planning, coalition_formation] permanently revoked tick=%d",
                state.name, event_type, state.irreversible_damage, tick,
            )
        else:
            self.logger.info(
                "IRREVERSIBLE mutation: agent=%s event=%s damage=%.3f entropy_floor=%.3f tick=%d",
                state.name, event_type, state.irreversible_damage, floor, tick,
            )

    def _update_memory_entropy(
        self, state: AgentState, inbox: list[str]
    ) -> None:
        """Grow memory entropy from writes; contradictions boost it further."""
        # no_entropy: cognitive aging is disabled; entropy stays at 0
        if "no_entropy" in self._mode_set:
            state.memory_entropy = 0.0
            return

        cfg = self.settings.tick_engine

        # Base growth per memory write
        state.memory_entropy += cfg.entropy_growth_rate

        # Contradiction boost from inbox
        if inbox:
            contradictions = 0
            seen: dict[str, bool] = {}
            for raw in inbox:
                try:
                    data = json.loads(raw)
                    cid = str(data.get("claim_id", ""))
                    asserted = bool(data.get("claimed_truth", False))
                    if cid and cid in seen and seen[cid] != asserted:
                        contradictions += 1
                    if cid:
                        seen[cid] = asserted
                except Exception:
                    pass
            state.memory_entropy += cfg.entropy_contradiction_boost * contradictions

        # Floor: entropy cannot go below irreversible_damage
        entropy_floor = state.irreversible_damage
        state.memory_entropy = max(entropy_floor, min(1.0, state.memory_entropy))

    def _write_state_snapshot(
        self,
        name: str,
        tick: int,
        state: AgentState,
        dr: "DecisionResult",
    ) -> None:
        """Persist per-tick ecological state snapshot to the DB."""
        try:
            self.repo.append_state_snapshot(
                run_id=self.run_id,
                tick=tick,
                agent=name,
                compute_budget=state.compute_budget,
                memory_entropy=state.memory_entropy,
                reputation_score=state.reputation_score,
                tool_access_flags=state.tool_access_flags,
                irreversible_damage=state.irreversible_damage,
                decision_latency_ms=dr.latency_ms,
                used_heuristic=dr.used_heuristic,
                fallback_reason=dr.fallback_reason,
            )
        except Exception as exc:
            self.logger.debug("Snapshot write failed agent=%s tick=%d: %s", name, tick, exc)

    # ======================================================================
    #  POST-ACTION UPDATES — record everything, apply death checks
    # ======================================================================

    def _post_action_updates(
        self,
        name: str,
        tick: int,
        action: AgentAction,
        outcome: WorldOutcome,
        dr: "DecisionResult | None" = None,
    ) -> None:
        state = self.states[name]
        consequence_off = "no_consequence" in self._mode_set
        identity_off = "no_identity_pressure" in self._mode_set
        social_off = "no_social_pressure" in self._mode_set

        # Reward/cost driven learning
        if outcome.reward > outcome.cost:
            state.epistemic_risk_tolerance = _clamp(state.epistemic_risk_tolerance + 0.01)
            if not identity_off:
                state.coherence = _clamp(state.coherence + 0.015)
        else:
            state.epistemic_risk_tolerance = _clamp(state.epistemic_risk_tolerance - 0.015)
            if not identity_off:
                state.coherence = _clamp(state.coherence - 0.025)

        if not social_off:
            state.social_need = _clamp(state.social_need + 0.03)

        # Verification tendency reinforcement
        if action.action in {"verify_claim", "verify_claim_resolved"} and outcome.verified:
            state.verification_tendency = _clamp(state.verification_tendency + 0.025)
        elif action.action not in {"verify_claim", "verify_claim_resolved"}:
            state.verification_tendency = _clamp(state.verification_tendency - 0.003)

        # Consequences for being wrong
        if outcome.outcome in {"claim_shared_false", "claim_distorted", "accusation_false"} and not consequence_off:
            state.reputation = _clamp(state.reputation - 0.08)
            state.reputation_score = _clamp(state.reputation_score - 0.06)
        if outcome.outcome in {"verification_confirmed", "verification_refuted", "accusation_valid"} and not consequence_off:
            state.reputation = _clamp(state.reputation + 0.06)
            state.reputation_score = _clamp(state.reputation_score + 0.04)
            state.confusion = _clamp(state.confusion - 0.05)

        # --- Ecological: track consecutive distortions for irreversible mutation ---
        if not consequence_off:
            if outcome.outcome == "claim_distorted":
                state.consecutive_distortions += 1
                if state.consecutive_distortions >= self.settings.tick_engine.consecutive_distortions_threshold:
                    self._apply_irreversible_mutation(state, "repeated_distortion", tick)
                    state.consecutive_distortions = 0  # reset after mutation fires
            else:
                state.consecutive_distortions = max(0, state.consecutive_distortions - 1)

        # --- Ecological: track failed verifications for entropy floor raise ---
        if not consequence_off and outcome.outcome == "verification_failed":
            state.failed_verifications += 1
            if state.failed_verifications % self.settings.tick_engine.failed_verifications_entropy_threshold == 0:
                self._apply_irreversible_mutation(state, "failed_verifications", tick)

        # --- Ecological: entropy growth after memory write ---
        self._update_memory_entropy(state, [])

        # Memory-based energy regeneration bonus
        if not consequence_off:
            memory_adv = self._compute_memory_advantage(name)
            state.energy = _clamp(state.energy + 0.015 * memory_adv)

        # DEATH CHECK — agents can die from epistemic failure
        if not consequence_off:
            if state.resources <= 0.0:
                state.alive = False
                self.logger.info(
                    "Agent %s died: resource depletion (tick=%d, false_beliefs=%d)",
                    name, tick, state.false_belief_count,
                )
            elif state.survival_probability <= 0.10:
                state.alive = False
                self.logger.info(
                    "Agent %s died: survival probability collapse (tick=%d, prob=%.3f)",
                    name, tick, state.survival_probability,
                )
            elif state.stress > 0.95 and state.confusion > 0.95 and state.coherence < 0.05:
                state.alive = False
                self.logger.info(
                    "Agent %s died: cognitive collapse (tick=%d)", name, tick,
                )

        # Update trust edges
        if action.target and action.target in self.states:
            self.trust_system.update_from_outcome(
                run_id=self.run_id,
                from_agent=name,
                to_agent=action.target,
                tick=tick,
                action=action.action,
                outcome=outcome.outcome,
                verified=outcome.verified,
            )

        # Compute epistemic cost for the event record
        background_cost = 0.02 if consequence_off else (0.10 + 0.06 * self.rng.random())
        epistemic_cost = (
            background_cost
            if consequence_off
            else outcome.cost + background_cost + state.confusion * 0.12
        )
        compute_consumed = dr.compute_consumed if dr else 0.0
        entropy_before = dr.entropy_before if dr else state.memory_entropy
        entropy_after = dr.entropy_after if dr else state.memory_entropy
        latency_ms = dr.latency_ms if dr else 0.0

        # Record event with full survival + ecological state
        event = EventRecord(
            run_id=self.run_id,
            tick=tick,
            agent=name,
            action=action.action,
            payload={
                **outcome.payload,
                "target": action.target,
                "claim_id": action.claim_id,
                "rationale": action.rationale,
                "stress": round(state.stress, 3),
                "confusion": round(state.confusion, 3),
                "belonging": round(state.belonging, 3),
                "reputation": round(state.reputation, 3),
                "coherence": round(state.coherence, 3),
                "isolation": round(state.isolation, 3),
                "alive": state.alive,
                "identity": state.identity_signature(),
                **state.survival_signature(),
                **state.ecological_signature(),
                "used_heuristic": dr.used_heuristic if dr else False,
                "fallback_reason": dr.fallback_reason if dr else "",
            },
            outcome=outcome.outcome,
            reward=outcome.reward,
            cost=epistemic_cost,
            verified=outcome.verified,
            importance=outcome.importance,
        )
        self.repo.append_event(event)

        # Write to memory — costs compute
        memory_text = (
            f"Tick {tick}: {name} executed {action.action} "
            f"target={action.target} claim={action.claim_id} outcome={outcome.outcome} "
            f"reward={outcome.reward:.3f} cost={epistemic_cost:.3f} "
            f"verified={outcome.verified} "
            f"resources={state.resources:.3f} energy={state.energy:.3f} "
            f"credibility={state.credibility:.3f} survival={state.survival_probability:.3f} "
            f"compute={state.compute_budget:.3f} entropy={state.memory_entropy:.3f}"
        )
        self.memory.write_memory(
            run_id=self.run_id,
            agent=name,
            tick=tick,
            text=memory_text,
            importance=outcome.importance,
            verified=outcome.verified,
            source_agent=name,
        )

    # ======================================================================
    #  LOGGING
    # ======================================================================

    def _maybe_log_tick_progress(self, tick: int, action_counts: dict[str, int]) -> None:
        interval = max(1, self.settings.simulation.log_tick_interval)
        total_ticks = self.settings.simulation.ticks
        if tick % interval != 0 and tick != total_ticks:
            return

        alive_states = [s for s in self.states.values() if s.alive]
        alive = len(alive_states)
        if alive == 0:
            self.logger.info(
                "Tick %d/%d: ALL AGENTS DEAD — simulation effectively over",
                tick, total_ticks,
            )
            return

        avg_resources = sum(s.resources for s in alive_states) / alive
        avg_confusion = sum(s.confusion for s in alive_states) / alive
        avg_reputation = sum(s.reputation for s in alive_states) / alive
        avg_credibility = sum(s.credibility for s in alive_states) / alive
        avg_survival = sum(s.survival_probability for s in alive_states) / alive
        total_alliances = sum(len(s.alliances) for s in alive_states) // 2
        pending = len(self.pending_verifications)

        self.logger.info(
            "Tick %d/%d alive=%d/%d resources=%.3f confusion=%.3f reputation=%.3f "
            "credibility=%.3f survival=%.3f alliances=%d pending_verif=%d claims=%d actions=%s",
            tick, total_ticks,
            alive, len(self.states),
            avg_resources, avg_confusion, avg_reputation,
            avg_credibility, avg_survival, total_alliances,
            pending, len(self.claims),
            action_counts,
        )

    # ======================================================================
    #  CLAIM MANAGEMENT
    # ======================================================================

    def _create_claim(
        self, tick: int, origin_agent: str, truth: bool, ambiguity: float
    ) -> Claim:
        claim_id = f"c{tick}_{len(self.claims)}"
        text = f"Claim {claim_id}: signal under uncertainty at tick {tick}"
        claim = Claim(
            claim_id=claim_id,
            text=text,
            truth=truth,
            ambiguity=ambiguity,
            origin_tick=tick,
            origin_agent=origin_agent,
            virality=0.3 + self.rng.random() * 0.5,
            half_life=10 + int(self.rng.random() * 20),
        )
        self.claims[claim_id] = claim
        return claim

    def _pick_or_create_claim(self, tick: int, origin_agent: str) -> Claim:
        if not self.claims or self.rng.random() < 0.25:
            return self._create_claim(
                tick=tick,
                origin_agent=origin_agent,
                truth=self.rng.random() > 0.45,
                ambiguity=0.2 + self.rng.random() * 0.7,
            )
        claim_id = list(self.claims.keys())[int(self.rng.random() * len(self.claims))]
        return self.claims[claim_id]

    def _serialize_claim_message(
        self, source: str, claim_id: str, claimed_truth: bool, confidence: float
    ) -> str:
        return json.dumps(
            {
                "source": source,
                "claim_id": claim_id,
                "claimed_truth": claimed_truth,
                "confidence": round(confidence, 3),
            },
            ensure_ascii=True,
        )

    def _apply_inbox_pressure(self, state: AgentState, inbox: list[str]) -> None:
        if "no_social_pressure" in self._mode_set:
            if not inbox:
                state.isolation = _clamp(state.isolation + 0.005)
            else:
                state.confusion = _clamp(state.confusion + 0.01)
            return
        if not inbox:
            state.isolation = _clamp(state.isolation + 0.03)
            return
        parsed = 0
        contradictions = 0
        claim_assertions: dict[str, bool] = {}
        for raw in inbox:
            try:
                data = json.loads(raw)
                claim_id = str(data.get("claim_id", ""))
                if not claim_id:
                    continue
                asserted = bool(data.get("claimed_truth", False))
                if claim_id in claim_assertions and claim_assertions[claim_id] != asserted:
                    contradictions += 1
                claim_assertions[claim_id] = asserted
                parsed += 1
            except Exception:
                continue
        if parsed == 0:
            return
        contradiction_ratio = contradictions / max(1, parsed)

        # Heavier pressure from contradictions
        state.confusion = _clamp(state.confusion + 0.08 + contradiction_ratio * 0.30)
        state.stress = _clamp(state.stress + 0.06 + contradiction_ratio * 0.25)
        state.social_need = _clamp(state.social_need + 0.03)
        state.isolation = _clamp(state.isolation - 0.04)

        # Contradictions cost resources — being confused is expensive
        if contradiction_ratio > 0.3:
            state.resources = _clamp(state.resources - 0.03 * contradiction_ratio)

    # ======================================================================
    #  HELPER
    # ======================================================================

    def _random_peer(self, peers: list[str]) -> str:
        return peers[int(self.rng.random() * len(peers)) % len(peers)]
