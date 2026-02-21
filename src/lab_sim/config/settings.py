from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DBSettings:
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return (
            f"dbname={self.name} user={self.user} password={self.password} "
            f"host={self.host} port={self.port}"
        )


@dataclass(frozen=True)
class OllamaSettings:
    host: str
    llm_model: str
    embedding_model: str
    llm_temperature: float = 0.2
    timeout_seconds: int = 90
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5


@dataclass(frozen=True)
class SimulationSettings:
    agent_count: int = 10
    ticks: int = 200
    periodic_replanning_interval: int = 12
    boredom_threshold: float = 0.75
    stress_threshold: float = 0.7
    log_tick_interval: int = 10


@dataclass(frozen=True)
class TickEngineSettings:
    """Controls the 2-phase async tick engine and computational survival budgets."""

    # Async concurrency
    per_request_timeout_s: float = 25.0
    """Aiohttp timeout per individual LLM call."""
    per_tick_deadline_s: float = 35.0
    """Max wall-clock time for the PLAN phase before heuristic fallback fires."""
    max_concurrent_llm: int = 6
    """Semaphore limit on simultaneous Ollama requests."""

    # Compute budget (energy metabolism)
    compute_per_llm_call: float = 0.15
    """Budget deducted per successful LLM inference call."""
    compute_per_memory_write: float = 0.03
    """Budget deducted per memory write operation."""
    compute_per_retrieval: float = 0.04
    """Budget deducted per vector memory retrieval."""
    compute_per_verification: float = 0.08
    """Budget deducted per verification action."""
    compute_regen_per_tick: float = 0.10
    """Budget restored to each agent at the start of every tick."""
    compute_min_threshold: float = 0.05
    """Below this, agent is marked compute_exhausted and forced to heuristic."""

    # Memory entropy (cognitive aging)
    entropy_growth_rate: float = 0.02
    """Entropy increase per memory write."""
    entropy_contradiction_boost: float = 0.05
    """Additional entropy per inbox contradiction."""
    entropy_decay_on_summarize: float = 0.10
    """Entropy reduction granted when agent chooses to isolate (compress/rest)."""
    entropy_hallucination_scale: float = 0.40
    """Multiplier: hallucination_prob = 0.05 + scale * memory_entropy."""

    # Tool access reputation thresholds
    threshold_verification_tool: float = 0.30
    """reputation_score required to keep verification_tool access."""
    threshold_coalition_formation: float = 0.40
    """reputation_score required to keep coalition_formation access."""
    threshold_data_retrieval: float = 0.25
    """reputation_score required to keep data_retrieval access."""
    threshold_advanced_planning: float = 0.55
    """reputation_score required to keep advanced_planning access."""

    # Irreversible mutation triggers
    consecutive_distortions_threshold: int = 3
    """Number of consecutive distort_claim actions before permanent damage."""
    failed_verifications_entropy_threshold: int = 5
    """Cumulative failed verifications before entropy floor is raised."""
    irreversible_damage_per_betrayal: float = 0.10
    """Permanent irreversible_damage added per alliance betrayal."""
    entropy_floor_raise_per_trigger: float = 0.05
    """Amount by which irreversible_damage raises the entropy floor."""

    # Temporal decay (opportunity loss)
    opportunity_window_ticks: int = 8
    """How many ticks a deadline opportunity lives before expiring."""
    opportunity_missed_resource_penalty: float = 0.06
    """Resources lost when an opportunity deadline expires unmet."""


@dataclass(frozen=True)
class AppSettings:
    db: DBSettings
    ollama: OllamaSettings
    simulation: SimulationSettings
    tick_engine: TickEngineSettings
    output_dir: Path

    @staticmethod
    def from_env() -> "AppSettings":
        return AppSettings(
            db=DBSettings(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                name=os.getenv("DB_NAME", "lab"),
                user=os.getenv("DB_USER", "lab_user"),
                password=os.getenv("DB_PASSWORD", "lab_pass"),
            ),
            ollama=OllamaSettings(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                llm_model=os.getenv("LLM_MODEL", "qwen2.5:1.5b"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
                timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90")),
                max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "2")),
                retry_backoff_seconds=float(
                    os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "1.5")
                ),
            ),
            simulation=SimulationSettings(
                agent_count=int(os.getenv("AGENT_COUNT", "10")),
                ticks=int(os.getenv("TICKS", "200")),
                periodic_replanning_interval=int(
                    os.getenv("PERIODIC_REPLAN_INTERVAL", "12")
                ),
                boredom_threshold=float(os.getenv("BOREDOM_THRESHOLD", "0.75")),
                stress_threshold=float(os.getenv("STRESS_THRESHOLD", "0.7")),
                log_tick_interval=int(os.getenv("LOG_TICK_INTERVAL", "10")),
            ),
            tick_engine=TickEngineSettings(
                per_request_timeout_s=float(
                    os.getenv("TICK_REQUEST_TIMEOUT_S", "25.0")
                ),
                per_tick_deadline_s=float(
                    os.getenv("TICK_DEADLINE_S", "35.0")
                ),
                max_concurrent_llm=int(os.getenv("MAX_CONCURRENT_LLM", "6")),
                compute_per_llm_call=float(
                    os.getenv("COMPUTE_PER_LLM_CALL", "0.15")
                ),
                compute_per_memory_write=float(
                    os.getenv("COMPUTE_PER_MEMORY_WRITE", "0.03")
                ),
                compute_per_retrieval=float(
                    os.getenv("COMPUTE_PER_RETRIEVAL", "0.04")
                ),
                compute_per_verification=float(
                    os.getenv("COMPUTE_PER_VERIFICATION", "0.08")
                ),
                compute_regen_per_tick=float(
                    os.getenv("COMPUTE_REGEN_PER_TICK", "0.10")
                ),
                compute_min_threshold=float(
                    os.getenv("COMPUTE_MIN_THRESHOLD", "0.05")
                ),
                entropy_growth_rate=float(
                    os.getenv("ENTROPY_GROWTH_RATE", "0.02")
                ),
                entropy_contradiction_boost=float(
                    os.getenv("ENTROPY_CONTRADICTION_BOOST", "0.05")
                ),
                entropy_decay_on_summarize=float(
                    os.getenv("ENTROPY_DECAY_ON_SUMMARIZE", "0.10")
                ),
                entropy_hallucination_scale=float(
                    os.getenv("ENTROPY_HALLUCINATION_SCALE", "0.40")
                ),
                threshold_verification_tool=float(
                    os.getenv("THRESHOLD_VERIFICATION_TOOL", "0.30")
                ),
                threshold_coalition_formation=float(
                    os.getenv("THRESHOLD_COALITION_FORMATION", "0.40")
                ),
                threshold_data_retrieval=float(
                    os.getenv("THRESHOLD_DATA_RETRIEVAL", "0.25")
                ),
                threshold_advanced_planning=float(
                    os.getenv("THRESHOLD_ADVANCED_PLANNING", "0.55")
                ),
                consecutive_distortions_threshold=int(
                    os.getenv("CONSECUTIVE_DISTORTIONS_THRESHOLD", "3")
                ),
                failed_verifications_entropy_threshold=int(
                    os.getenv("FAILED_VERIFICATIONS_ENTROPY_THRESHOLD", "5")
                ),
                irreversible_damage_per_betrayal=float(
                    os.getenv("IRREVERSIBLE_DAMAGE_PER_BETRAYAL", "0.10")
                ),
                entropy_floor_raise_per_trigger=float(
                    os.getenv("ENTROPY_FLOOR_RAISE_PER_TRIGGER", "0.05")
                ),
                opportunity_window_ticks=int(
                    os.getenv("OPPORTUNITY_WINDOW_TICKS", "8")
                ),
                opportunity_missed_resource_penalty=float(
                    os.getenv("OPPORTUNITY_MISSED_RESOURCE_PENALTY", "0.06")
                ),
            ),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
        )
