import pytest
from pprint import pprint

from multi_agent_decision_system.core.state import (
    create_initial_state,
    start_new_iteration,
)
from multi_agent_decision_system.core.schema import (
    PlannerOutput,
    SpecialistOutput,
    Recommendation,
)

from multi_agent_decision_system.agents.planner_agent import run_planner_agent
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_ai_agent import run_ml_ai_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_agent import run_product_agent


# =============================================================================
# Test constants
# =============================================================================

DECISION_QUESTION = "Should we use batch or online inference?"

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


def print_planner_io(state, planner: PlannerOutput):
    print_section("PLANNER INPUT (RAW)")
    pprint(
        {
            "decision_question": state.input.decision_question,
            "constraints": state.input.constraints,
        }
    )

    print_section("PLANNER OUTPUT (PARSED)")
    pprint(planner.model_dump())


def print_specialist_input(agent_name: str, state, planner: PlannerOutput):
    print_section(f"{agent_name.upper()} INPUT (RAW)")

    raw_input = {
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints,
        "planner_slice": getattr(planner, agent_name, None),
    }
    pprint(raw_input)

    print_section(f"{agent_name.upper()} INPUT (PARSED)")

    parsed_input = {
        "decision_question": state.input.decision_question,
        "constraints": (
            state.input.constraints.model_dump()
            if hasattr(state.input.constraints, "model_dump")
            else state.input.constraints
        ),
        "planner_slice": (
            getattr(planner, agent_name).model_dump()
            if getattr(planner, agent_name, None)
            else None
        ),
    }
    pprint(parsed_input)


def print_specialist_output(agent_name: str, output: SpecialistOutput):
    print_section(f"{agent_name.upper()} OUTPUT (PARSED)")
    pprint(output.model_dump())


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def base_state():
    state = create_initial_state(
        decision_question=DECISION_QUESTION,
        constraints=CONSTRAINTS,
        max_iterations=1,
    )
    return start_new_iteration(state)


@pytest.fixture(scope="module")
def planner_output(base_state):
    result = run_planner_agent(base_state)
    planner = result["planner"]

    base_state.current.planner = planner

    print_planner_io(base_state, planner)

    return planner


# =============================================================================
# Assertions
# =============================================================================

def assert_specialist_output(output: SpecialistOutput, agent_name: str):
    assert isinstance(output, SpecialistOutput)
    assert output.agent_name == agent_name
    assert output.recommendation in Recommendation.__args__
    assert 0.0 <= output.confidence <= 1.0
    assert 2 <= len(output.benefits) <= 4
    assert 2 <= len(output.risks) <= 4


# =============================================================================
# Planner test
# =============================================================================

def test_planner_real_llm(planner_output):
    planner = planner_output

    assert isinstance(planner, PlannerOutput)
    assert planner.agent_name == "planner"

    assert any(
        [
            planner.systems,
            planner.ml_ai,
            planner.cost,
            planner.product,
        ]
    )


# =============================================================================
# Systems Agent
# =============================================================================

def test_systems_agent_real_llm(base_state, planner_output):
    print_specialist_input("systems", base_state, planner_output)

    result = run_systems_agent(base_state)
    output = result["systems"]

    print_specialist_output("systems", output)
    assert_specialist_output(output, "systems")


# =============================================================================
# ML / AI Agent
# =============================================================================

def test_ml_ai_agent_real_llm(base_state, planner_output):
    print_specialist_input("ml_ai", base_state, planner_output)

    result = run_ml_ai_agent(base_state)
    output = result["ml_ai"]

    print_specialist_output("ml_ai", output)
    assert_specialist_output(output, "ml_ai")


# =============================================================================
# Cost Agent
# =============================================================================

def test_cost_agent_real_llm(base_state, planner_output):
    print_specialist_input("cost", base_state, planner_output)

    result = run_cost_agent(base_state)
    output = result["cost"]

    print_specialist_output("cost", output)
    assert_specialist_output(output, "cost")


# =============================================================================
# Product & Risk Agent
# =============================================================================

def test_product_agent_real_llm(base_state, planner_output):
    print_specialist_input("product", base_state, planner_output)

    result = run_product_agent(base_state)
    output = result["product"]

    print_specialist_output("product", output)
    assert_specialist_output(output, "product")