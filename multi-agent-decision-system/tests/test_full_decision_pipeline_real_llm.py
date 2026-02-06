import pytest
from pprint import pprint

from multi_agent_decision_system.core.state import (
    create_initial_state,
    start_new_iteration,
)
from multi_agent_decision_system.core.schema import (
    PlannerOutput,
    SpecialistOutput,
    DetectorOutput,
    CriticOutput,
    SynthesizerOutput,
    GateOutput,
    Recommendation,
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
# Pretty printing helpers
# =============================================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_raw_parsed(title: str, raw, parsed):
    print_section(f"{title} INPUT (RAW)")
    pprint(raw)
    print_section(f"{title} INPUT (PARSED)")
    pprint(parsed)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def state():
    state = create_initial_state(
        decision_question=DECISION_QUESTION,
        options=OPTIONS,
        constraints=CONSTRAINTS,
        max_iterations=1,
    )
    return start_new_iteration(state)


# =============================================================================
# Planner
# =============================================================================

def test_planner(state):
    output = run_planner_agent(state)["planner"]
    state.current.planner = output

    print_raw_parsed(
        "PLANNER",
        {
            "decision_question": state.input.decision_question,
            "options": state.input.options,
            "constraints": state.input.constraints,
        },
        output.model_dump(),
    )

    assert isinstance(output, PlannerOutput)
    assert output.agent_name == "planner"


# =============================================================================
# Specialist agents
# =============================================================================

@pytest.mark.parametrize(
    "agent_name, runner",
    [
        ("systems", run_systems_agent),
        ("ml_ai", run_ml_ai_agent),
        ("cost", run_cost_agent),
        ("product", run_product_agent),
    ],
)
def test_specialist_agents(state, agent_name, runner):
    planner = state.current.planner
    output = runner(state)[agent_name]
    setattr(state.current, agent_name, output)

    print_raw_parsed(
        agent_name.upper(),
        {
            "decision_question": state.input.decision_question,
            "options": state.input.options,
            "constraints": state.input.constraints,
            "planner_slice": getattr(planner, agent_name),
        },
        output.model_dump(),
    )

    assert isinstance(output, SpecialistOutput)
    assert output.agent_name == agent_name
    assert output.recommendation in Recommendation.__args__


# =============================================================================
# Detector
# =============================================================================

def test_detector(state):
    output = run_detector_agent(state)["detector"]
    state.current.detector = output

    print_raw_parsed(
        "DETECTOR",
        {
            "specialist_outputs": {
                k: getattr(state.current, k)
                for k in ["systems", "ml_ai", "cost", "product"]
            }
        },
        output.model_dump(),
    )

    assert isinstance(output, DetectorOutput)
    assert output.agent_name == "detector"


# =============================================================================
# Critic
# =============================================================================

def test_critic(state):
    output = run_critic_agent(state)["critic"]
    state.current.critic = output

    print_raw_parsed(
        "CRITIC",
        {
            "specialist_outputs": {
                k: getattr(state.current, k)
                for k in ["systems", "ml_ai", "cost", "product"]
            },
            "detector": state.current.detector,
        },
        output.model_dump(),
    )

    assert isinstance(output, CriticOutput)
    assert output.agent_name == "critic"


# =============================================================================
# Synthesizer
# =============================================================================

def test_synthesizer(state):
    output = run_synthesizer_agent(state)["synthesizer"]
    state.current.synthesizer = output

    print_raw_parsed(
        "SYNTHESIZER",
        {
            "specialist_outputs": {
                k: getattr(state.current, k)
                for k in ["systems", "ml_ai", "cost", "product"]
            },
            "detector": state.current.detector,
            "critic": state.current.critic,
        },
        output.model_dump(),
    )

    assert isinstance(output, SynthesizerOutput)
    assert output.agent_name == "synthesizer"
    assert output.final_recommendation in Recommendation.__args__


# =============================================================================
# Gate (internal, deterministic)
# =============================================================================

def test_gate(state):
    output = run_gate(state)
    state.current.gate = output

    print_section("GATE OUTPUT (PARSED)")
    pprint(output.model_dump())

    assert isinstance(output, GateOutput)
    assert output.agent_name == "gate"
    assert isinstance(output.decision.approved, bool)