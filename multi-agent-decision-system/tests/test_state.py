import pytest

from multi_agent_decision_system.core.state import create_initial_state
from multi_agent_decision_system.core.schemas import DecisionInput


def test_state_initializes_correctly():
    decision_input = DecisionInput(
        decision_question="Should we use batch or online inference?",
        decision_type="ml_architecture_decision",
        decision_template="operational_mode",
        constraints={
            "latency_sensitivity": "high",
            "budget_sensitivity": "medium",
        },
    )

    state = create_initial_state(decision_input)

    # ---- Input correctness ----
    assert state.input.decision_question == decision_input.decision_question

    # ---- Iteration semantics ----
    assert state.termination.iteration_count == 0

    # ---- Authoritative state should be empty ----
    assert state.plan is None
    assert state.agent_outputs == {}
    assert state.disagreements is None
    assert state.critic_feedback is None
    assert state.synthesis is None
    assert state.confidence is None

    # ---- Termination defaults ----
    assert state.termination.can_terminate is False
    assert state.termination.max_iterations > 0

    # ---- Metadata ----
    assert state.metadata.run_id is not None
    assert "created_at" in state.metadata.timestamps

# Added property to DecisionState in core.schemas (assuming the file location)
# This snippet should be added inside the DecisionState class definition:

#     @property
#     def iteration(self) -> int:
#         return self.termination.iteration_count