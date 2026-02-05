import json
from unittest.mock import patch

from multi_agent_decision_system.core.schemas import DecisionInput
from multi_agent_decision_system.core.state import DecisionState, create_initial_state
from multi_agent_decision_system.core.normalization import normalize_constraints
from multi_agent_decision_system.graph.planner_graph import build_planner_graph


def test_planner_graph_executes_planner_node():
    """
    LangGraph should:
    - Accept an initial DecisionState
    - Execute the planner node
    - Return a DecisionState with a populated plan
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

    initial_state = create_initial_state(decision_input)

    mock_planner_response = {
        "systems": {
            "question": "Can the system meet latency requirements?",
            "why_it_matters": "Latency constraints differ between batch and online systems.",
            "key_unknowns": ["peak traffic"],
        },
        "ml": None,
        "cost": None,
        "product_risk": None,
        "assumptions": [],
        "clarifying_questions": [],
        "model_used": "gpt-4.1",
        "estimated_tokens_in": 100,
        "estimated_tokens_out": 150,
    }

    graph = build_planner_graph()

    # ------------------
    # Act
    # ------------------
    with patch(
        "multi_agent_decision_system.agents.planner.ChatOpenAI"
    ) as MockChat:
        mock_llm = MockChat.return_value
        mock_llm.invoke.return_value.content = json.dumps(mock_planner_response)
        mock_llm.invoke.return_value.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 150,
            }
        }

        final_state_dict = graph.invoke(initial_state)
        final_state = DecisionState(**final_state_dict)

        assert final_state.plan is not None

    # ------------------
    # Assert
    # ------------------
    assert final_state is not None
    assert final_state.plan is not None
    assert final_state.plan.systems is not None

    # Graph should preserve and enrich state
    assert final_state.input == initial_state.input
    assert final_state.termination.iteration_count == 0