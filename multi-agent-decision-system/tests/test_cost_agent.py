import json
from unittest.mock import patch

from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.core.schema import AgentOutput
from multi_agent_decision_system.core.state import (
    DecisionInput,
    PlannerOutput,
    create_initial_state,
)
from multi_agent_decision_system.core.normalization import normalize_constraints


def test_cost_agent_produces_valid_partial_update():
    """
    Cost Agent should:
    - Consume DecisionState
    - Call the LLM once
    - Return a LangGraph-safe partial update
    - Produce a valid AgentOutput under agent_outputs["cost"]
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
        cost={
            "question": "What is the long-term operational cost?",
            "why_it_matters": "Online systems introduce persistent infra and on-call costs.",
            "key_unknowns": ["traffic scale"],
        },
        assumptions=["Team is cost-sensitive"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_llm_response = {
        "agent_name": "cost",
        "recommendation": "defer",
        "confidence": 0.55,
        "benefits": ["Avoids premature infra spend"],
        "risks": ["Slower feature rollout"],
    }

    with patch(
        "multi_agent_decision_system.agents.cost_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_llm_response)

        updates = run_cost_agent(state)

    assert "agent_outputs" in updates
    assert "cost" in updates["agent_outputs"]

    output = updates["agent_outputs"]["cost"]
    assert isinstance(output, AgentOutput)
    assert output.agent_name == "cost"
    assert 0.0 <= output.confidence <= 1.0