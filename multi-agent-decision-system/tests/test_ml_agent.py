import json
from unittest.mock import patch

from multi_agent_decision_system.core.schemas import (
    DecisionInput,
    PlannerOutput,
    AgentOutput,
)
from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.normalization import normalize_constraints
from multi_agent_decision_system.agents.ml_agent import run_ml_agent


def test_ml_agent_logs_input_and_output_correctly():
    """
    ML Agent should:
    - Log its inputs explicitly
    - Produce a schema-valid AgentOutput
    - Log its output
    - Remain independent of other agents
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

    # Inject minimal planner output (ML slice only)
    state.plan = PlannerOutput(
        ml={
            "question": "Does the model require real-time features?",
            "why_it_matters": "Online inference requires fresh features.",
            "key_unknowns": ["feature latency", "label delay"],
        },
        assumptions=["Feature pipelines exist but are not real-time"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_ml_response = {
        "agent_name": "ml",
        "recommendation": "defer",
        "confidence": 0.55,
        "benefits": [
            "Allows time to validate feature freshness",
            "Reduces risk of training-serving skew",
        ],
        "risks": [
            "Delayed model impact",
            "May require additional data engineering work",
        ],
    }

    # ------------------
    # Act
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.ml_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_ml_response)

        updated_state = run_ml_agent(state)

    # ------------------
    # Assert: input logging
    # ------------------
    assert "ml" in updated_state.input_log.agent_inputs

    agent_input = updated_state.input_log.agent_inputs["ml"]
    assert agent_input["agent_name"] == "ml"
    assert agent_input["decision_question"] == decision_input.decision_question
    assert agent_input["planner_slice"] is not None
    assert "feature latency" in str(agent_input["planner_slice"])

    # ------------------
    # Assert: output logging
    # ------------------
    assert "ml" in updated_state.agent_outputs
    assert "ml" in updated_state.output_log.agent_outputs

    output = updated_state.agent_outputs["ml"]
    assert isinstance(output, AgentOutput)

    # ------------------
    # Assert: schema + epistemic bounds
    # ------------------
    assert 0.0 <= output.confidence <= 1.0
    assert output.confidence <= 0.7  # pessimistic ML bias
    assert output.recommendation in {
        "option_a",
        "option_b",
        "hybrid",
        "defer",
        "insufficient_information",
    }
    assert len(output.benefits) > 0
    assert len(output.risks) > 0