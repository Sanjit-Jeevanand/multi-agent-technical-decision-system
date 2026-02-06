import json
from unittest.mock import patch

from multi_agent_decision_system.agents.ml_ai_agent import run_ml_agent
from multi_agent_decision_system.core.schema import AgentOutput
from multi_agent_decision_system.core.state import (
    DecisionInput,
    PlannerOutput,
    create_initial_state,
)
from multi_agent_decision_system.core.normalization import normalize_constraints


def test_ml_agent_produces_valid_partial_update():
    """
    ML Agent should:
    - Consume DecisionState
    - Call the LLM once
    - Return a LangGraph-safe partial update
    - Produce a valid AgentOutput under agent_outputs["ml"]
    """

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

    state.plan = PlannerOutput(
        ml={
            "question": "Does the model require real-time features?",
            "why_it_matters": "Feature freshness impacts model accuracy.",
            "key_unknowns": ["staleness tolerance"],
        },
        assumptions=["Feature pipelines are moderately stable"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_llm_response = {
        "agent_name": "ml",
        "recommendation": "defer",
        "confidence": 0.6,
        "benefits": ["Avoids training–serving skew"],
        "risks": ["Delayed deployment of online features"],
    }

    with patch(
        "multi_agent_decision_system.agents.ml_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_llm_response)

        updates = run_ml_agent(state)

    assert "agent_outputs" in updates
    assert "ml" in updates["agent_outputs"]

    output = updates["agent_outputs"]["ml"]
    assert isinstance(output, AgentOutput)
    assert output.agent_name == "ml"
    assert 0.0 <= output.confidence <= 1.0