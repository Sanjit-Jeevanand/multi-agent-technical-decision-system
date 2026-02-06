import pytest
from pprint import pprint

from multi_agent_decision_system.core.state_v2 import (
    DecisionState,
    DecisionDelta,
)
from multi_agent_decision_system.core.schema import (
    SynthesizerOutput,
    CriticOutput,
    GateOutput,
)

from multi_agent_decision_system.agents.planner_agent import run_planner_agent
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_ai_agent import run_ml_ai_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_agent import run_product_agent
from multi_agent_decision_system.agents.detector_agent import run_detector_agent
from multi_agent_decision_system.agents.critic_agent import run_critic_agent
from multi_agent_decision_system.agents.synthesizer_agent import run_synthesizer_agent
from multi_agent_decision_system.agents.gate_agent import run_gate


# =============================================================================
# Test constants
# =============================================================================

DECISION_QUESTION = "Should we use batch or online inference?"

OPTIONS = {
    "option_a": "batch inference",
    "option_b": "online inference",
}

CONSTRAINTS = {
    "latency_sensitivity": "medium",
    "team_size": "small",
    "risk_tolerance": "medium",
}


# =============================================================================
# Helper
# =============================================================================

def run_full_iteration(state: DecisionState):
    state.current.planner = run_planner_agent(state)["planner"]

    for agent, runner in [
        ("systems", run_systems_agent),
        ("ml_ai", run_ml_ai_agent),
        ("cost", run_cost_agent),
        ("product", run_product_agent),
    ]:
        setattr(state.current, agent, runner(state)[agent])

    state.current.detector = run_detector_agent(state)["detector"]
    state.current.critic = run_critic_agent(state)["critic"]
    state.current.synthesizer = run_synthesizer_agent(state)["synthesizer"]
    state.current.gate = run_gate(state)


# =============================================================================
# Fixture
# =============================================================================

@pytest.fixture(scope="module")
def state():
    return DecisionState(
        input={
            "decision_question": DECISION_QUESTION,
            "options": OPTIONS,
            "constraints": CONSTRAINTS,
        },
        max_iterations=2,
    )


# =============================================================================
# Iteration 1
# =============================================================================

def test_iteration_1(state):
    print("\n" + "=" * 80)
    print("ITERATION 1")
    print("=" * 80)

    run_full_iteration(state)

    pprint(state.current.synthesizer.model_dump())
    pprint(state.current.critic.model_dump())
    pprint(state.current.gate.model_dump())

    assert isinstance(state.current.synthesizer, SynthesizerOutput)
    assert isinstance(state.current.critic, CriticOutput)
    assert isinstance(state.current.gate, GateOutput)

    # Expected: critic blocks
    assert state.current.critic.requires_revision is True
    assert state.current.gate.decision.approved is False


# =============================================================================
# Iteration 2 — Critic rerun with user input
# =============================================================================

def test_iteration_2_with_critic_rerun(state):
    print("\n" + "=" * 80)
    print("ITERATION 2 (USER ACCEPTS RISK, CRITIC RERUN)")
    print("=" * 80)

    delta = DecisionDelta(
        accepted_risks=[
            "Model performance may degrade between batch runs due to drift."
        ],
        rejected_recommendations=[
            "hybrid"
        ],
        notes="We accept staleness risk for v1 and prioritize operational simplicity."
    )

    state.start_next_iteration(delta)

    # Sanity checks
    assert state.iteration == 2
    assert len(state.history) == 1

    run_full_iteration(state)

    pprint(state.current.synthesizer.model_dump())
    pprint(state.current.critic.model_dump())
    pprint(state.current.gate.model_dump())

    # Critic MUST be rerun
    assert isinstance(state.current.critic, CriticOutput)

    # Synthesizer MUST respect rejection
    assert state.current.synthesizer.final_recommendation != "hybrid"

    # Gate behavior is deterministic but user-controlled
    assert isinstance(state.current.gate.decision.approved, bool)
    # Gate must not re-block on previously accepted risks
    assert (
        state.current.gate.decision.approved
        or state.current.gate.decision.required_actions
    ), "Gate must respect accepted risks and not hard-block repeatedly"
    # History integrity
    snapshot = state.history[0]
    assert snapshot.critic is not None
    assert snapshot.delta_applied is not None