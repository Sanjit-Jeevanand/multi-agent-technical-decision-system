import json
from unittest.mock import patch

from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.schemas import (
    DecisionInput,
    PlannerOutput,
    AgentOutput,
)
from multi_agent_decision_system.core.normalization import normalize_constraints
from multi_agent_decision_system.graph.specialist_graph import build_specialist_graph


def test_specialist_graph_executes_all_agents_in_parallel():

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

    # Inject full planner output so all agents are eligible
    state.plan = PlannerOutput(
        systems={
            "question": "Can the system meet latency requirements?",
            "why_it_matters": "Latency differs between batch and online systems.",
            "key_unknowns": ["peak traffic"],
        },
        ml={
            "question": "Does the model require real-time features?",
            "why_it_matters": "Feature freshness impacts accuracy.",
            "key_unknowns": ["staleness tolerance"],
        },
        cost={
            "question": "What are the infra cost implications?",
            "why_it_matters": "Online inference increases baseline cost.",
            "key_unknowns": ["cloud pricing at scale"],
        },
        product_risk={
            "question": "What is the user impact of delayed predictions?",
            "why_it_matters": "Trust and compliance depend on predictability.",
            "key_unknowns": [],
        },
        assumptions=[],
        clarifying_questions=[],
        model_used="gpt-5.1",
        estimated_tokens_in=200,
        estimated_tokens_out=300,
    )

    graph = build_specialist_graph()

    # ------------------
    # Mock all agents
    # ------------------
    def mock_agent_output(agent_name: str):
        return json.dumps(
            {
                "agent_name": agent_name,
                "recommendation": "defer",
                "confidence": 0.5,
                "benefits": ["Conservative default"],
                "risks": ["Delayed decision"],
            }
        )

    with patch(
        "multi_agent_decision_system.agents.systems_agent.ChatOpenAI"
    ) as MockSystems, patch(
        "multi_agent_decision_system.agents.ml_agent.ChatOpenAI"
    ) as MockML, patch(
        "multi_agent_decision_system.agents.cost_agent.ChatOpenAI"
    ) as MockCost, patch(
        "multi_agent_decision_system.agents.product_risk_agent.ChatOpenAI"
    ) as MockProduct:

        MockSystems.return_value.invoke.return_value.content = mock_agent_output(
            "systems"
        )
        MockML.return_value.invoke.return_value.content = mock_agent_output("ml")
        MockCost.return_value.invoke.return_value.content = mock_agent_output("cost")
        MockProduct.return_value.invoke.return_value.content = mock_agent_output(
            "product_risk"
        )

        # ------------------
        # Act
        # ------------------
        final_state = graph.invoke(state)

    # ------------------
    # Assert: agent outputs
    # ------------------
    assert final_state.agent_outputs is not None

    for agent in ["systems", "ml", "cost", "product_risk"]:
        assert agent in final_state.agent_outputs
        assert isinstance(final_state.agent_outputs[agent], AgentOutput)

    # ------------------
    # Assert: input logging (parallel-safe)
    # ------------------
    assert "agent_inputs" in final_state.input_log
    for agent in ["systems", "ml", "cost", "product_risk"]:
        assert agent in final_state.input_log["agent_inputs"]
        assert isinstance(final_state.input_log["agent_inputs"][agent], list)
        assert len(final_state.input_log["agent_inputs"][agent]) == 1

    # ------------------
    # Assert: output logging (append-only)
    # ------------------
    assert "agent_outputs" in final_state.output_log
    for agent in ["systems", "ml", "cost", "product_risk"]:
        outputs = final_state.output_log["agent_outputs"][agent]
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0]["agent_name"] == agent