import json
from unittest.mock import patch

from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.core.schemas import AgentOutput
from multi_agent_decision_system.core.state import (
    DecisionInput,
    PlannerOutput,
    create_initial_state,
)
from multi_agent_decision_system.core.normalization import normalize_constraints


def test_systems_agent_produces_valid_partial_update():
    """
    Systems Agent should:
    - Consume DecisionState
    - Call the LLM once
    - Return a LangGraph-safe partial update
    - Produce a valid AgentOutput under agent_outputs["systems"]
    """

    # ------------------
    # Arrange
    # ------------------
    raw_constraints = {
        "latency_ms": 200,
        "team_size": 4,
        "risk_tolerance": "low",
    }

    constraints = normalize_constraints(
        raw_constraints,
        decision_context="batch_vs_online",
    )

    decision_input = DecisionInput(
        decision_question="Should we use batch or online inference?",
        constraints=constraints,
    )

    state = create_initial_state(decision_input)

    # Inject planner slice (systems only)
    state.plan = PlannerOutput(
        systems={
            "question": "Can the system meet latency requirements?",
            "why_it_matters": "Latency constraints differ between batch and online systems.",
            "key_unknowns": ["peak traffic"],
        },
        assumptions=["Traffic is expected to grow steadily"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_llm_response = {
        "agent_name": "systems",
        "recommendation": "defer",
        "confidence": 0.6,
        "benefits": [
            "Avoids premature operational complexity",
            "Reduces risk of brittle real-time systems",
        ],
        "risks": [
            "Delayed real-time capabilities",
            "Potential future re-architecture",
        ],
    }

    # ------------------
    # Act
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.systems_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_llm_response)

        updates = run_systems_agent(state)

    # ------------------
    # Assert
    # ------------------
    assert isinstance(updates, dict)
    assert "agent_outputs" in updates
    assert "systems" in updates["agent_outputs"]

    output = updates["agent_outputs"]["systems"]
    assert isinstance(output, AgentOutput)

    # Schema sanity
    assert output.agent_name == "systems"
    assert output.recommendation in {
        "option_a",
        "option_b",
        "hybrid",
        "defer",
        "insufficient_information",
    }
    assert 0.0 <= output.confidence <= 1.0
    assert isinstance(output.benefits, list)
    assert isinstance(output.risks, list)