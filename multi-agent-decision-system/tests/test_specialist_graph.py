import json
from unittest.mock import patch

from multi_agent_decision_system.graph.specialist_graph import build_specialist_graph
from multi_agent_decision_system.core.state import (
    DecisionInput,
    PlannerOutput,
    create_initial_state,
)
from multi_agent_decision_system.core.schemas import AgentOutput
from multi_agent_decision_system.core.normalization import normalize_constraints


def test_specialist_graph_executes_all_agents_in_parallel():
    """
    Specialist graph should:
    - Execute all specialist agents from the same initial state
    - Allow each agent to write independently into agent_outputs
    - Merge partial updates correctly
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

    # ------------------
    # Mock all agents
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.systems_agent.ChatOpenAI"
    ) as MockSystems, patch(
        "multi_agent_decision_system.agents.ml_agent.ChatOpenAI"
    ) as MockML, patch(
        "multi_agent_decision_system.agents.cost_agent.ChatOpenAI"
    ) as MockCost, patch(
        "multi_agent_decision_system.agents.product_risk_agent.ChatOpenAI"
    ) as MockProduct:

        MockSystems.return_value.invoke.return_value.content = mock_agent_output("systems")
        MockML.return_value.invoke.return_value.content = mock_agent_output("ml")
        MockCost.return_value.invoke.return_value.content = mock_agent_output("cost")
        MockProduct.return_value.invoke.return_value.content = mock_agent_output("product_risk")

        # ------------------
        # Act
        # ------------------
        final_state = graph.invoke(state)

    # ------------------
    # Assert
    # ------------------
    assert final_state is not None
    assert isinstance(final_state, dict)
    assert "agent_outputs" in final_state
    assert isinstance(final_state["agent_outputs"], dict)

    for agent in ["systems", "ml", "cost", "product_risk"]:
        assert agent in final_state["agent_outputs"]
        output = final_state["agent_outputs"][agent]
        assert isinstance(output, AgentOutput)
        assert output.agent_name == agent
        assert output.recommendation in {
            "option_a",
            "option_b",
            "hybrid",
            "defer",
            "insufficient_information",
        }
        assert 0.0 <= output.confidence <= 1.0