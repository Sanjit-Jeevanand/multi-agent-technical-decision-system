import json
from unittest.mock import patch

from multi_agent_decision_system.agents.product_agent import run_product_risk_agent
from multi_agent_decision_system.core.schema import AgentOutput
from multi_agent_decision_system.core.state import (
    DecisionInput,
    PlannerOutput,
    create_initial_state,
)
from multi_agent_decision_system.core.normalization import normalize_constraints


def test_product_risk_agent_produces_valid_partial_update():
    """
    Product & Risk Agent should:
    - Consume DecisionState
    - Call the LLM once
    - Return a LangGraph-safe partial update
    - Produce a valid AgentOutput under agent_outputs["product_risk"]
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
        product_risk={
            "question": "What is the user impact of failures?",
            "why_it_matters": "Silent failures erode trust.",
            "key_unknowns": ["failure visibility"],
        },
        assumptions=["Users expect consistent behavior"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_llm_response = {
        "agent_name": "product_risk",
        "recommendation": "defer",
        "confidence": 0.5,
        "benefits": ["Avoids user-facing instability"],
        "risks": ["Delayed feature delivery"],
    }

    with patch(
        "multi_agent_decision_system.agents.product_risk_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_llm_response)

        updates = run_product_risk_agent(state)

    assert "agent_outputs" in updates
    assert "product_risk" in updates["agent_outputs"]

    output = updates["agent_outputs"]["product_risk"]
    assert isinstance(output, AgentOutput)
    assert output.agent_name == "product_risk"
    assert 0.0 <= output.confidence <= 1.0