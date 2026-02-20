"""EcologicalAgents tick engine — 2-phase async execution model.

This package provides a clean, modular 2-phase simulation tick:

  PHASE A — PLAN: all agent decisions are dispatched concurrently via asyncio.
  PHASE B — ACT:  decisions are applied to the world in a deterministic order.
"""
from lab_sim.engine.tick_engine import TickEngine

__all__ = ["TickEngine"]
