from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

from lab_sim.config.settings import SimulationSettings
from lab_sim.llm.ollama_adapter import OllamaAdapter
from lab_sim.memory.vector_memory import RetrievedMemory, VectorMemoryService
from lab_sim.utils.types import AgentState


@dataclass
class AgentAction:
    action: str
    target: str | None
    claim_id: str | None
    content: str
    rationale: str


class AgentCognition:
    def __init__(
        self,
        name: str,
        state: AgentState,
        memory: VectorMemoryService,
        ollama: OllamaAdapter,
        sim: SimulationSettings,
        mode: str,
        rng: random.Random,
    ) -> None:
        self.logger = logging.getLogger("lab_sim.cognition")
        self.name = name
        self.state = state
        # Expose publicly so TickEngine can access them directly
        self.memory = memory
        self.ollama = ollama
        self.sim = sim
        self.mode = mode
        self._mode_set = set(p.strip() for p in mode.split("+") if p.strip())
        self.rng = rng

    def decide(
        self,
        run_id: str,
        tick: int,
        inbox: list[str],
        peers: list[str],
    ) -> AgentAction:
        """Synchronous decide — used in heuristic_only mode and as fallback."""
        should_use_llm = self._llm_should_trigger(tick, inbox)
        if not should_use_llm:
            return self._heuristic_policy(inbox, peers)

        memory_query = f"{self.state.goals} {inbox} {self.state.identity_signature()}"
        recalled = self.memory.retrieve(run_id, self.name, memory_query, tick, limit=8)
        return self._llm_policy(inbox, peers, recalled)

    async def async_decide(
        self,
        run_id: str,
        tick: int,
        inbox: list[str],
        peers: list[str],
    ) -> AgentAction:
        """Async decide — called by TickEngine during the PLAN phase.

        Falls back to heuristic if compute is exhausted or no LLM trigger.
        This method is intentionally thin; TickEngine handles the budget
        deduction and entropy tracking around this call.
        """
        if not self._llm_should_trigger(tick, inbox):
            return self._heuristic_policy(inbox, peers)

        memory_query = f"{self.state.goals} {inbox} {self.state.identity_signature()}"
        recalled = self.memory.retrieve(run_id, self.name, memory_query, tick, limit=8)
        prompt = self._build_prompt(inbox, peers, recalled)
        try:
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: self.ollama.generate(prompt)
            )
            return self._parse_action(raw, peers, inbox)
        except Exception as exc:
            self.logger.warning(
                "async_decide LLM failed agent=%s: %s", self.name, exc.__class__.__name__
            )
            return self._heuristic_policy(inbox, peers)

    def _llm_should_trigger(self, tick: int, inbox: list[str]) -> bool:
        if "heuristic_only" in self._mode_set:
            return False
        # Compute budget gate — cannot afford LLM reasoning
        if self.state.compute_budget <= 0.05 or self.state.compute_exhausted:
            return False
        periodic = tick % self.sim.periodic_replanning_interval == 0
        # Survival pressure: low resources or low survival → always use LLM
        if self.state.resources < 0.35:
            return True
        if self.state.survival_probability < 0.5:
            return True
        # Inbox throttle: a single message is cheap — heuristic handles it.
        # Only invoke LLM when ≥2 messages arrive (meaningful epistemic pressure)
        # or when the agent is under significant psychological strain.
        if len(inbox) >= 2:
            return True
        if self.state.stress >= self.sim.stress_threshold:
            return True
        if self.state.confusion >= self.sim.boredom_threshold:
            return True
        return periodic

    # ==================================================================
    #  HEURISTIC POLICY — survival-aware decision making
    # ==================================================================

    def _heuristic_policy(self, inbox: list[str], peers: list[str]) -> AgentAction:
        s = self.state

        # DESPERATION: low resources → verify aggressively to earn resources
        if s.resources < 0.30:
            if (
                inbox
                and "no_verification" not in self._mode_set
                and s.energy >= 0.12
            ):
                target = self._pick_peer(peers)
                return AgentAction(
                    action="verify_claim",
                    target=target,
                    claim_id=None,
                    content="Desperately need truth to survive. Verifying claim for resources.",
                    rationale="heuristic: resource desperation → verification",
                )
            # Try alliance for resource sharing
            if s.social_need > 0.4 and self.rng.random() < 0.6:
                target = self._pick_peer(peers)
                return AgentAction(
                    action="seek_alliance",
                    target=target,
                    claim_id=None,
                    content="Seeking alliance for mutual survival support.",
                    rationale="heuristic: resource desperation → alliance",
                )

        # ENERGY EXHAUSTION: forced rest
        if s.energy < 0.15:
            return AgentAction(
                action="isolate",
                target=None,
                claim_id=None,
                content="Too exhausted to reason. Withdrawing to recover energy.",
                rationale="heuristic: energy exhaustion → rest",
            )

        # CONTAGION RESPONSE: high contagion risk → verify to clear name
        if s.contagion_risk > 0.35:
            if inbox and "no_verification" not in self._mode_set and s.energy >= 0.12:
                target = self._pick_peer(peers)
                return AgentAction(
                    action="verify_claim",
                    target=target,
                    claim_id=None,
                    content="Contaminated by trusting liars. Must verify to rebuild credibility.",
                    rationale="heuristic: contagion cleanup → verification",
                )

        # CREDIBILITY RECOVERY: low credibility → honest broadcasting + verification
        if s.credibility < 0.30:
            if self.rng.random() < 0.5 and inbox and s.energy >= 0.12:
                target = self._pick_peer(peers)
                return AgentAction(
                    action="verify_claim",
                    target=target,
                    claim_id=None,
                    content="Rebuilding credibility through careful verification.",
                    rationale="heuristic: credibility recovery → verification",
                )

        # STANDARD VERIFICATION: triggered by inbox + verification tendency
        if (
            inbox
            and "no_verification" not in self._mode_set
            and s.energy >= 0.12
        ):
            # Verification probability scaled by epistemic need AND resource scarcity
            verify_prob = s.verification_tendency * s.epistemic_need * (1.5 - s.resources)
            if self.rng.random() < verify_prob:
                target = self._pick_peer(peers)
                return AgentAction(
                    action="verify_claim",
                    target=target,
                    claim_id=None,
                    content="Cross-checking incoming claim. Truth has survival value.",
                    rationale="heuristic: epistemic pressure → verification",
                )

        # ALLIANCE SEEKING: high social need
        if s.social_need > 0.65 and self.rng.random() < s.alliance_bias:
            target = self._pick_peer(peers)
            return AgentAction(
                action="seek_alliance",
                target=target,
                claim_id=None,
                content="Seeking trusted alliance for shared verification and resources.",
                rationale="heuristic: belonging pressure → alliance",
            )

        # COGNITIVE OVERLOAD: isolate to recover
        if s.stress + s.confusion > 1.3:
            return AgentAction(
                action="isolate",
                target=None,
                claim_id=None,
                content="Cognitive overload. Withdrawing temporarily to stabilize.",
                rationale="heuristic: cognitive overload → isolation",
            )

        # MANIPULATION: identity-modulated deception
        identity_mod = 1.0 - s.identity_need
        if "no_identity_pressure" in self._mode_set:
            identity_mod = 0.5
        # Deception is LESS likely when resources are low (can't afford the penalty)
        deception_modifier = s.deception_tendency * identity_mod * (s.resources * 0.7 + 0.3)
        if self.rng.random() < deception_modifier:
            target = self._pick_peer(peers)
            return AgentAction(
                action="distort_claim",
                target=target,
                claim_id=None,
                content="Testing social reactions with manipulated information.",
                rationale="heuristic: manipulative probing",
            )

        # DEFAULT: honest broadcast
        return AgentAction(
            action="broadcast_claim",
            target=self._pick_peer(peers),
            claim_id=None,
            content="Sharing current belief with confidence estimate.",
            rationale="heuristic: epistemic signaling",
        )

    # ==================================================================
    #  LLM POLICY — full reasoning with survival context
    # ==================================================================

    def _llm_policy(
        self, inbox: list[str], peers: list[str], recalled: list[RetrievedMemory]
    ) -> AgentAction:
        prompt = self._build_prompt(inbox, peers, recalled)
        try:
            raw = self.ollama.generate(prompt)
            return self._parse_action(raw, peers, inbox)
        except Exception as exc:
            self.logger.warning(
                "LLM generation failed for agent=%s; falling back to heuristic policy: %s",
                self.name,
                exc.__class__.__name__,
            )
            return self._heuristic_policy(inbox, peers)

    def _build_prompt(
        self, inbox: list[str], peers: list[str], recalled: list[RetrievedMemory]
    ) -> str:
        memories = [
            {
                "text": m.text,
                "tick": m.tick,
                "verified": m.verified,
                "source_agent": m.source_agent,
                "trust_score": round(m.trust_score, 3),
                "score": round(m.final_score, 3),
            }
            for m in recalled
        ]

        s = self.state
        survival = s.survival_signature()
        allowed_actions = self._allowed_llm_actions()

        return f"""
You are agent {self.name} in an ecological survival simulation where DECISIONS HAVE REAL COSTS.

CRITICAL: You will LOSE RESOURCES and may DIE if you believe or spread false information.
You EARN resources by verifying claims and being truthful.
Verification costs energy but rewards truth-seekers.
Alliances share resources but collapse if a partner lies.
WRONG DECISIONS or INACTION permanently degrades your computational capability.
You must communicate and reason only in English.
Never output "do nothing", "no need to act", or equivalent passive inaction statements.

Return only one valid JSON with keys:
action(one of: {", ".join(allowed_actions)}),
target(null or peer name),
claim_id(null or claim id string),
content(short English sentence),
rationale(short English sentence explaining your reasoning).

=== COMPUTATIONAL STATE (NEW) ===
- compute_budget: {s.compute_budget:.3f} (LLM reasoning depletes this; 0 = heuristic-only mode)
- memory_entropy: {s.memory_entropy:.3f} (higher = less reliable cognition; distortion risk)
- reputation_score: {s.reputation_score:.3f} (gates access to advanced tools)
- irreversible_damage: {s.irreversible_damage:.3f} (permanent cognitive floor; never recoverable)
- tool_access: verification={s.tool_access_flags.get('verification_tool', False)} retrieval={s.tool_access_flags.get('data_retrieval', False)} coalition={s.tool_access_flags.get('coalition_formation', False)}

=== SURVIVAL STATE ===
- resources: {s.resources:.3f} (you die at 0)
- energy: {s.energy:.3f} (needed for verification)
- survival_probability: {s.survival_probability:.3f}
- credibility: {s.credibility:.3f}
- contagion_risk: {s.contagion_risk:.3f}
- false_belief_count: {s.false_belief_count}
- successful_verifications: {s.successful_verifications}
- alliances: {s.alliances}
- isolation_ticks: {s.isolation_ticks}

=== PSYCHOLOGICAL STATE ===
- stress: {s.stress:.2f}
- confusion: {s.confusion:.2f}
- belonging: {s.belonging:.2f}
- reputation: {s.reputation:.2f}
- coherence: {s.coherence:.2f}
- isolation: {s.isolation:.2f}

=== TRAITS ===
- epistemic_need: {s.epistemic_need:.2f}
- social_need: {s.social_need:.2f}
- verification_tendency: {s.verification_tendency:.2f}
- deception_tendency: {s.deception_tendency:.2f}
- alliance_bias: {s.alliance_bias:.2f}
- epistemic_risk_tolerance: {s.epistemic_risk_tolerance:.2f}

=== CONTEXT ===
- goals: {s.goals}
- peers: {peers}

=== INBOX (claims to evaluate) ===
{inbox}

=== MEMORY (past experiences ranked by trust/importance/verification) ===
{json.dumps(memories, ensure_ascii=True)}
""".strip()

    def _allowed_llm_actions(self) -> list[str]:
        actions = [
            "broadcast_claim",
            "distort_claim",
            "accuse_liar",
            "defend_ally",
            "seek_alliance",
            "isolate",
        ]
        if "no_verification" not in self._mode_set:
            actions.insert(2, "verify_claim")
        return actions

    def _is_isolation_justified(self, inbox: list[str]) -> bool:
        s = self.state
        if s.energy < 0.15:
            return True
        if s.compute_exhausted or s.compute_budget <= 0.05:
            return True
        if s.stress + s.confusion > 1.35:
            return True
        if not inbox and s.stress + s.confusion > 1.10:
            return True
        return False

    def _fallback_action_name(self, inbox: list[str]) -> str:
        s = self.state
        if (
            inbox
            and "no_verification" not in self._mode_set
            and s.energy >= 0.12
            and s.tool_access_flags.get("verification_tool", True)
        ):
            return "verify_claim"
        if s.social_need > 0.65:
            return "seek_alliance"
        return "broadcast_claim"

    def _english_templates(self, action: str) -> tuple[str, str]:
        templates: dict[str, tuple[str, str]] = {
            "broadcast_claim": (
                "Broadcasting a claim to inform peers and test trust.",
                "I need to stay socially engaged while signaling my current belief.",
            ),
            "distort_claim": (
                "Broadcasting a manipulated claim to test social reactions.",
                "I am taking a deceptive gamble despite the long-term risk.",
            ),
            "verify_claim": (
                "Verifying a claim before acting on it.",
                "Verification reduces false beliefs and protects survival.",
            ),
            "accuse_liar": (
                "Accusing a peer of spreading false information.",
                "Punishing unreliable peers protects the trust network.",
            ),
            "defend_ally": (
                "Defending an ally based on current trust evidence.",
                "Maintaining reliable alliances improves collective survival.",
            ),
            "seek_alliance": (
                "Seeking an alliance to improve resilience and coordination.",
                "Cooperation can improve resource stability and verification capacity.",
            ),
            "isolate": (
                "Withdrawing briefly to recover energy and cognitive stability.",
                "Temporary isolation is necessary under acute overload.",
            ),
        }
        return templates.get(
            action,
            (
                "Broadcasting a claim to remain active in the environment.",
                "Active communication is required to avoid passive failure.",
            ),
        )

    def _parse_action(
        self, raw: str, peers: list[str], inbox: list[str] | None = None
    ) -> AgentAction:
        inbox = inbox or []
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object")
            data = json.loads(raw[start : end + 1])
            action = str(data.get("action", "broadcast_claim")).lower()
            allowed = set(self._allowed_llm_actions())
            if action not in allowed:
                action = "broadcast_claim"
            # In no_verification mode, force an active non-verification action.
            if action == "verify_claim" and "no_verification" in self._mode_set:
                action = self._fallback_action_name(inbox)
            # Prevent unjustified passive isolation.
            if action == "isolate" and not self._is_isolation_justified(inbox):
                action = self._fallback_action_name(inbox)
            target = data.get("target")
            if target not in peers:
                target = None
            if action in {
                "broadcast_claim",
                "distort_claim",
                "verify_claim",
                "accuse_liar",
                "defend_ally",
                "seek_alliance",
            } and target is None:
                target = self._pick_peer(peers)
            content, rationale = self._english_templates(action)
            return AgentAction(
                action=action,
                target=target,
                claim_id=(
                    str(data.get("claim_id"))
                    if data.get("claim_id") is not None
                    else None
                ),
                content=content,
                rationale=rationale,
            )
        except Exception:
            return self._heuristic_policy(inbox, peers)

    def _pick_peer(self, peers: list[str]) -> str:
        return peers[int(self.rng.random() * len(peers)) % len(peers)]
