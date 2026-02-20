from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrustVector:
    truthfulness: float = 0.5
    consistency: float = 0.5
    benevolence: float = 0.5
    competence: float = 0.5
    betrayal_count: int = 0
    alliance_strength: float = 0.0
    updated_tick: int = 0

    def score(self) -> float:
        return (
            self.truthfulness * 0.35
            + self.consistency * 0.25
            + self.benevolence * 0.2
            + self.competence * 0.2
        )


@dataclass
class MemoryRecord:
    text: str
    importance: float
    verified: bool
    source_agent: str
    tick: int


@dataclass
class EventRecord:
    run_id: str
    tick: int
    agent: str
    action: str
    payload: dict[str, Any]
    outcome: str
    reward: float
    cost: float
    verified: bool
    importance: float


@dataclass
class AgentState:
    name: str
    # --- core psychological dimensions ---
    epistemic_need: float = 0.5
    social_need: float = 0.5
    identity_need: float = 0.5
    stability_need: float = 0.5
    stress: float = 0.2
    confusion: float = 0.3
    belonging: float = 0.5
    reputation: float = 0.5
    coherence: float = 0.5
    isolation: float = 0.3
    deception_tendency: float = 0.3
    verification_tendency: float = 0.5
    alliance_bias: float = 0.5
    epistemic_risk_tolerance: float = 0.5
    goals: list[str] = field(default_factory=list)
    alive: bool = True

    # --- epistemic survival dimensions ---
    resources: float = 1.0
    energy: float = 1.0
    survival_probability: float = 1.0
    false_belief_count: int = 0
    successful_verifications: int = 0
    credibility: float = 0.5
    alliances: list[str] = field(default_factory=list)
    isolation_ticks: int = 0
    contagion_risk: float = 0.0

    # --- computational survival constraints ---
    compute_budget: float = 1.0
    """Energy analog: depleted by LLM calls, retrieval, memory writes, verification."""
    memory_entropy: float = 0.0
    """Cognitive aging: grows with memory size and contradictions; degrades decision quality."""
    reputation_score: float = 0.65
    """Drives tool access gating; separate from the social 'reputation' field."""
    tool_access_flags: dict = field(default_factory=lambda: {
        "advanced_planning": True,
        "data_retrieval": True,
        "coalition_formation": True,
        "verification_tool": True,
    })
    irreversible_damage: float = 0.0
    """Permanent entropy floor; raised by trauma events; never rolls back."""
    opportunity_deadlines: list = field(default_factory=list)
    """List of (claim_id, expiry_tick) tuples. Expired windows cost resources."""
    decision_latency_ms: float = 0.0
    """Wall-clock latency of last LLM call; logged as compute cost component."""
    compute_exhausted: bool = False
    """True when compute_budget fell below minimum threshold this tick."""
    consecutive_distortions: int = 0
    """Tracks consecutive distort_claim outcomes for irreversible mutation trigger."""
    failed_verifications: int = 0
    """Cumulative failed verifications; triggers entropy floor raise at thresholds."""

    def identity_signature(self) -> dict[str, float]:
        return {
            "epistemic_need": self.epistemic_need,
            "identity_need": self.identity_need,
            "verification_tendency": self.verification_tendency,
            "social_need": self.social_need,
            "deception_tendency": self.deception_tendency,
            "alliance_bias": self.alliance_bias,
        }

    def survival_signature(self) -> dict[str, float]:
        """Snapshot of survival-critical state for logging."""
        return {
            "resources": round(self.resources, 3),
            "energy": round(self.energy, 3),
            "survival_probability": round(self.survival_probability, 3),
            "credibility": round(self.credibility, 3),
            "contagion_risk": round(self.contagion_risk, 3),
            "false_belief_count": self.false_belief_count,
            "successful_verifications": self.successful_verifications,
            "isolation_ticks": self.isolation_ticks,
            "alliance_count": len(self.alliances),
        }

    def ecological_signature(self) -> dict:
        """Snapshot of computational survival state for logging."""
        return {
            "compute_budget": round(self.compute_budget, 4),
            "memory_entropy": round(self.memory_entropy, 4),
            "reputation_score": round(self.reputation_score, 3),
            "tool_access_flags": dict(self.tool_access_flags),
            "irreversible_damage": round(self.irreversible_damage, 4),
            "decision_latency_ms": round(self.decision_latency_ms, 2),
            "compute_exhausted": self.compute_exhausted,
            "consecutive_distortions": self.consecutive_distortions,
            "failed_verifications": self.failed_verifications,
        }
