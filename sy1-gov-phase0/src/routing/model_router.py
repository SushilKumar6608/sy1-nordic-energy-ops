"""
Model router for SY1-Gov.

Routes agent calls between:
  - claude-haiku-5-5   : fast, cheap - for read queries and status checks
  - claude-sonnet-4-6  : capable    - for diagnosis, recommendation, supervision

Routing logic (Phase 2.5 - rule-based):
  HAIKU  : demand agent, RAG simple retrieval, node status reads
  SONNET : diagnosis agent, recommendation agent, supervisor agent, complex RAG (HyDE)

Phase 3 upgrade path: replace rule-based classifier with a small trained classifier
on query complexity features (token count, intent type, requires_tools).
"""

from __future__ import annotations

from enum import Enum


class ModelTier(str, Enum):
    HAIKU  = "claude-haiku-5-5"
    SONNET = "claude-sonnet-4-6"


# Agent-to-model mapping (rule-based)
_AGENT_MODEL_MAP: dict[str, ModelTier] = {
    "demand_agent":          ModelTier.HAIKU,
    "rag_retrieval_agent":   ModelTier.HAIKU,   # simple retrieval; HyDE uses SONNET
    "rag_hyde_agent":        ModelTier.SONNET,  # HyDE needs stronger generation
    "gnn_localisation_agent": ModelTier.HAIKU,  # structured output, low complexity
    "diagnosis_agent":       ModelTier.SONNET,
    "recommendation_agent":  ModelTier.SONNET,
    "supervisor_agent":      ModelTier.SONNET,
}

# Cost per 1M tokens (USD) - for Langfuse cost tracking
# Update if Anthropic pricing changes
MODEL_COST_PER_1M: dict[str, dict[str, float]] = {
    "claude-haiku-5-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
}


class ModelRouter:
    """
    Returns the appropriate model for a given agent and query.
    """

    def route(self, agent_name: str, query: str | None = None) -> ModelTier:
        """
        Rule-based routing by agent name.
        query is reserved for the future classifier upgrade.
        """
        if agent_name not in _AGENT_MODEL_MAP:
            # Unknown agent - default to Sonnet (safe, capable)
            return ModelTier.SONNET
        return _AGENT_MODEL_MAP[agent_name]

    def estimate_cost_usd(
        self,
        model: ModelTier,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost in USD for a single call.
        Used by the Langfuse tracer for per-agent cost tracking.
        """
        pricing = MODEL_COST_PER_1M.get(model.value, MODEL_COST_PER_1M["claude-sonnet-4-6"])
        return (
            input_tokens  / 1_000_000 * pricing["input"] +
            output_tokens / 1_000_000 * pricing["output"]
        )

    @staticmethod
    def get_model_id(agent_name: str) -> str:
        """Convenience: return the model ID string directly."""
        router = ModelRouter()
        return router.route(agent_name).value
