# EcologicalAgents

This project aims to enable LLM-based agents to adapt their decision-making behavior through accumulated experience rather than through prompts or fine-tuning.

Instead of relying on surface-level conditioning (e.g., prompt changes, system roles, or retrieval context), the system introduces persistent internal state variables such as memory history, social relationships, resource access, risk exposure, and irreversible consequences. These lived trajectories dynamically influence the agent's decision policy over time, even when the underlying model weights and prompts remain fixed.

As a result, two agents instantiated from the same base model and given identical prompts may produce different decisions due to differences in their experiential history, social environment, or past outcomes. The agent's behavior evolves as a function of its interaction with environmental constraints and long-term consequences.

The primary goal is to shift behavioral adaptation away from inference-time prompting and toward developmental trajectory, allowing agents to make decisions based on prior experience rather than solely on instruction.