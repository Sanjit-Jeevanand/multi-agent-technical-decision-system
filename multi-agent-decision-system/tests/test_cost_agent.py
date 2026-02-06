import json
from unittest.mock import patch

from multi_agent_decision_system.core.schemas import (
    DecisionInput,
    PlannerOutput,
    AgentOutput,
)
from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.normalization import normalize_constraints
from multi_agent_decision_system.agents.cost_agent import run_cost_agent


def test_cost_agent_logs_input_and_output_correctly():


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

    # Inject minimal planner output (cost slice only)
    state.plan = PlannerOutput(
        cost={
            "question": "What is the long-term operational cost of online inference?",
            "why_it_matters": "Online systems introduce ongoing infra and on-call burden.",
            "key_unknowns": ["serving scale", "on-call load"],
        },
        assumptions=["Traffic volume is expected to grow steadily"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_cost_response = {
        "agent_name": "cost",
        "recommendation": "defer",
        "confidence": 0.55,
        "benefits": [
            "Avoids premature operational complexity",
            "Preserves engineering capacity for core work",
        ],
        "risks": [
            "Delayed system capabilities",
            "Potential rework if requirements change",
        ],
    }

    # ------------------
    # Act
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.cost_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_cost_response)

        updates = run_cost_agent(state)
        updated_state = state.model_copy(update=updates)

    # ------------------
    # Assert: input logging
    # ------------------
    assert "cost" in updated_state.input_log["agent_inputs"]

    agent_inputs = updated_state.input_log["agent_inputs"]["cost"]
    assert isinstance(agent_inputs, list)
    agent_input = agent_inputs[-1]
    assert agent_input["agent_name"] == "cost"
    assert agent_input["decision_question"] == decision_input.decision_question
    assert agent_input["planner_slice"] is not None
    assert "operational cost" in str(agent_input["planner_slice"])

    # ------------------
    # Assert: output logging
    # ------------------
    assert "cost" in updated_state.agent_outputs
    assert "cost" in updated_state.output_log["agent_outputs"]

    output = updated_state.agent_outputs["cost"]
    assert isinstance(output, AgentOutput)

    # ------------------
    # Assert: schema + cost conservatism
    # ------------------
    assert 0.0 <= output.confidence <= 1.0
    assert output.confidence <= 0.6  # cost agent pessimism
    assert output.recommendation in {
        "option_a",
        "option_b",
        "hybrid",
        "defer",
        "insufficient_information",
    }
    assert len(output.benefits) > 0
    assert len(output.risks) > 0