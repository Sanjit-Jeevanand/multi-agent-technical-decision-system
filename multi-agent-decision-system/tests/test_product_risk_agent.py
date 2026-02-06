import json
from unittest.mock import patch

from multi_agent_decision_system.core.schemas import (
    DecisionInput,
    PlannerOutput,
    AgentOutput,
)
from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.normalization import normalize_constraints
from multi_agent_decision_system.agents.product_risk_agent import (
    run_product_risk_agent,
)


def test_product_risk_agent_logs_input_and_output_correctly():
    """
    Product & Risk Agent should:
    - Log its inputs explicitly
    - Produce a schema-valid AgentOutput
    - Log its output
    - Not depend on other agents
    """

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
        decision_question="Should we use batch or online inference?",
        constraints=normalized_constraints,
    )

    state = create_initial_state(decision_input)

    # Inject planner output (product/risk slice only)
    state.plan = PlannerOutput(
        systems=None,
        ml=None,
        cost=None,
        product_risk={
            "question": "What is the user impact if predictions are delayed or incorrect?",
            "why_it_matters": "User trust and compliance depend on predictable behavior.",
            "key_unknowns": ["failure visibility"],
        },
        assumptions=["Users expect consistent behavior"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=120,
        estimated_tokens_out=200,
    )

    mock_product_risk_response = {
        "agent_name": "product_risk",
        "recommendation": "defer",
        "confidence": 0.55,
        "benefits": [
            "Avoids exposing users to unvalidated behavior",
            "Reduces risk of silent failures",
        ],
        "risks": [
            "Delayed feature delivery",
            "Potential loss of competitive advantage",
        ],
    }

    # ------------------
    # Act
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.product_risk_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(
            mock_product_risk_response
        )

        updated_state = run_product_risk_agent(state)

    # ------------------
    # Assert
    # ------------------

    # Agent output exists and is valid
    assert "product_risk" in updated_state.agent_outputs
    output = updated_state.agent_outputs["product_risk"]
    assert isinstance(output, AgentOutput)

    # Recommendation + confidence sanity
    assert output.recommendation in {
        "option_a",
        "option_b",
        "hybrid",
        "defer",
        "insufficient_information",
    }
    assert 0.0 <= output.confidence <= 1.0

    # Input logging exists
    assert "product_risk" in updated_state.input_log.agent_inputs
    input_log = updated_state.input_log.agent_inputs["product_risk"]
    assert input_log["agent_name"] == "product_risk"
    assert "decision_question" in input_log
    assert "constraints" in input_log
    assert "planner_slice" in input_log
    assert "iteration" in input_log

    # Output logging exists
    assert "product_risk" in updated_state.output_log.agent_outputs
    logged_output = updated_state.output_log.agent_outputs["product_risk"]
    assert logged_output["agent_name"] == "product_risk"