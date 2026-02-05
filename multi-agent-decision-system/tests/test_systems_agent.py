import json
from unittest.mock import patch

from multi_agent_decision_system.core.schemas import (
    DecisionInput,
    PlannerOutput,
    AgentOutput,
)
from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.normalization import normalize_constraints
from multi_agent_decision_system.agents.systems_agent import run_systems_agent


def test_systems_agent_logs_input_and_output_correctly():
    """
    Systems Agent should:
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

    # Inject a minimal planner output (systems slice only)
    state.plan = PlannerOutput(
        systems={
            "question": "Can the system meet latency requirements?",
            "why_it_matters": "Online systems impose stricter latency SLAs.",
            "key_unknowns": ["peak traffic"],
        },
        assumptions=["Traffic is relatively stable"],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=100,
        estimated_tokens_out=150,
    )

    mock_systems_response = {
        "agent_name": "systems",
        "recommendation": "hybrid",
        "confidence": 0.6,
        "benefits": [
            "Graceful degradation under load",
            "Operational isolation between batch and online paths",
        ],
        "risks": [
            "Increased deployment complexity",
            "More complex monitoring requirements",
        ],
    }

    # ------------------
    # Act
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.systems_agent.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_systems_response)

        updated_state = run_systems_agent(state)

    # ------------------
    # Assert: input logging
    # ------------------
    assert "systems" in updated_state.input_log.agent_inputs

    agent_input = updated_state.input_log.agent_inputs["systems"]
    assert agent_input["agent_name"] == "systems"
    assert agent_input["decision_question"] == decision_input.decision_question
    assert agent_input["planner_slice"] is not None

    # ------------------
    # Assert: output logging
    # ------------------
    assert "systems" in updated_state.agent_outputs
    assert "systems" in updated_state.output_log.agent_outputs

    output = updated_state.agent_outputs["systems"]
    assert isinstance(output, AgentOutput)

    # ------------------
    # Assert: schema + bounds
    # ------------------
    assert 0.0 <= output.confidence <= 1.0
    assert output.recommendation in {
        "option_a",
        "option_b",
        "hybrid",
        "defer",
        "insufficient_information",
    }
    assert len(output.benefits) > 0
    assert len(output.risks) > 0