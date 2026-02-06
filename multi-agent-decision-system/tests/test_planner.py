import json
from unittest.mock import patch

from multi_agent_decision_system.agents.planner import run_planner
from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.schemas import PlannerOutput, DecisionInput
from multi_agent_decision_system.core.normalization import normalize_constraints


def test_planner_produces_valid_bounded_output():

    # ------------------
    # Arrange
    # ------------------
    raw_constraints = {
        "latency_ms": 200,
        "team_size": 4,
        "risk_tolerance": "low",
    }

    normalized_constraints = normalize_constraints(
        raw_constraints,
        decision_context="batch_vs_online",
    )

    decision_input = DecisionInput(
        decision_question="Should we use batch or online inference for this ML system?",
        constraints=normalized_constraints,
    )

    state = create_initial_state(decision_input)

    mock_planner_response = {
        "systems": {
            "question": "Can the system meet latency and reliability requirements under expected load?",
            "why_it_matters": "Online inference introduces stricter latency and availability constraints.",
            "key_unknowns": ["peak traffic volume", "acceptable degradation modes"],
        },
        "ml": {
            "question": "Does the model require real-time features or can it tolerate staleness?",
            "why_it_matters": "Batch inference may introduce prediction drift if features lag.",
            "key_unknowns": ["feature freshness sensitivity"],
        },
        "cost": {
            "question": "What is the long-term operational cost of maintaining real-time inference?",
            "why_it_matters": "Online systems increase infrastructure and on-call burden.",
            "key_unknowns": ["cloud serving costs at scale"],
        },
        "product_risk": {
            "question": "What is the user impact if predictions are delayed or stale?",
            "why_it_matters": "User trust may be affected by incorrect or delayed predictions.",
            "key_unknowns": [],
        },
        "assumptions": ["Traffic patterns are relatively stable"],
        "clarifying_questions": [],
        "model_used": "gpt-5.1",
        "estimated_tokens_in": 120,
        "estimated_tokens_out": 220,
    }

    # ------------------
    # Act
    # ------------------
    with patch("multi_agent_decision_system.agents.planner.ChatOpenAI") as MockChat:
        mock_llm = MockChat.return_value

        mock_llm.invoke.return_value.content = json.dumps(mock_planner_response)
        mock_llm.invoke.return_value.response_metadata = {
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 220,
            }
        }

        updates = run_planner(state)

    # ------------------
    # Assert
    # ------------------

    assert "plan" in updates
    assert isinstance(updates["plan"], PlannerOutput)

    dimensions = [
        updates["plan"].systems,
        updates["plan"].ml,
        updates["plan"].cost,
        updates["plan"].product_risk,
    ]
    assert sum(d is not None for d in dimensions) <= 4

    assert updates["plan"].model_used == "gpt-5.1"
    assert updates["plan"].estimated_tokens_in > 0
    assert updates["plan"].estimated_tokens_out > 0

    planner_dump = updates["plan"].model_dump_json()
    forbidden_terms = ["should choose", "recommend", "best option"]
    assert not any(term in planner_dump.lower() for term in forbidden_terms)