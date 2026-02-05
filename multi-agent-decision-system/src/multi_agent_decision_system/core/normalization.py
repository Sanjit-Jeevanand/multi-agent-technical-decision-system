

from typing import Any, Dict, Optional, Literal

from multi_agent_decision_system.core.schemas import DecisionConstraints


DecisionContext = Literal[
    "batch_vs_online",
    "build_vs_buy",
    "architecture_choice",
    "tool_selection",
    "feature_rollout",
]


def normalize_constraints(
    raw: Dict[str, Any],
    decision_context: DecisionContext,
) -> DecisionConstraints:
    """
    Normalize user-provided constraints using intent-first, numbers-second logic.

    Design principles:
    - Semantic intent ALWAYS wins if provided.
    - Numeric values are interpreted relative to the decision context.
    - No numeric values propagate beyond this boundary.
    - Downstream agents only see categorical intent signals.
    """

    # -------------------------
    # Intent-first passthrough
    # -------------------------
    intent: Dict[str, Optional[str]] = raw.get("intent", {})

    def intent_or_none(key: str) -> Optional[str]:
        return intent.get(key)

    # -------------------------
    # Context-aware heuristics
    # -------------------------
    def normalize_budget(budget_usd: Optional[int]) -> Optional[str]:
        if budget_usd is None:
            return None

        if decision_context in {"feature_rollout", "tool_selection"}:
            if budget_usd < 100_000:
                return "high"
            elif budget_usd < 400_000:
                return "medium"
            return "low"

        if decision_context in {"batch_vs_online", "architecture_choice"}:
            if budget_usd < 200_000:
                return "high"
            elif budget_usd < 800_000:
                return "medium"
            return "low"

        # Conservative default
        if budget_usd < 50_000:
            return "high"
        elif budget_usd < 200_000:
            return "medium"
        return "low"

    def normalize_latency(latency_ms: Optional[int]) -> Optional[str]:
        if latency_ms is None:
            return None

        if latency_ms <= 100:
            return "high"
        elif latency_ms <= 500:
            return "medium"
        return "low"

    def normalize_team_size(team_size: Optional[int]) -> Optional[str]:
        if team_size is None:
            return None

        if team_size <= 3:
            return "small"
        elif team_size <= 8:
            return "medium"
        return "large"

    def normalize_risk(risk_score: Optional[float]) -> Optional[str]:
        if risk_score is None:
            return None

        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.7:
            return "medium"
        return "high"

    # -------------------------
    # Construct normalized constraints
    # -------------------------
    return DecisionConstraints(
        budget_sensitivity=(
            intent_or_none("budget_sensitivity")
            or normalize_budget(raw.get("budget_usd"))
        ),
        latency_sensitivity=(
            intent_or_none("latency_sensitivity")
            or normalize_latency(raw.get("latency_ms"))
        ),
        team_size=(
            intent_or_none("team_size")
            or normalize_team_size(raw.get("team_size"))
        ),
        risk_tolerance=(
            intent_or_none("risk_tolerance")
            or normalize_risk(raw.get("risk_score"))
        ),
    )