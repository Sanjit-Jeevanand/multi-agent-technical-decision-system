import pytest
from pprint import pprint
import tiktoken

from multi_agent_decision_system.core.schema import DecisionInput
from multi_agent_decision_system.core.state_v2 import GatePolicyTier

from multi_agent_decision_system.core.state_v2 import (
    CurrentIteration,
    DecisionState,
    DecisionDelta,
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
# Constants
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

MODEL_NAME = "gpt-5"

# GPT-5 pricing (per 1M tokens)
GPT5_INPUT_PER_M = 1.25
GPT5_OUTPUT_PER_M = 10.00


# =============================================================================
# Token estimation helper
# =============================================================================

encoder = tiktoken.encoding_for_model("gpt-4")  # close enough for estimation


def estimate_cost(text: str, *, is_output: bool):
    tokens = len(encoder.encode(text))
    if is_output:
        cost = (tokens / 1_000_000) * GPT5_OUTPUT_PER_M
    else:
        cost = (tokens / 1_000_000) * GPT5_INPUT_PER_M
    return tokens, round(cost, 4)


def print_block(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# Full iteration runner with tracing
# =============================================================================

def run_iteration_with_trace(state: DecisionState, iteration_label: str):
    print_block(iteration_label)

    def run_and_trace(name, fn):
        raw = fn(state)
        parsed = list(raw.values())[0]

        raw_text = parsed.model_dump_json(indent=2)

        # Approximate input tokens from prompt context
        input_text = str(state.input) + str(state.current.model_dump())

        print_block(f"{name.upper()} — INPUT CONTEXT")
        print(input_text)

        # Output tokens
        out_tokens, out_cost = estimate_cost(raw_text, is_output=True)

        in_tokens, in_cost = estimate_cost(input_text, is_output=False)

        print_block(f"{name.upper()} — RAW OUTPUT")
        print(raw_text)

        print_block(f"{name.upper()} — PARSED OUTPUT")
        pprint(parsed.model_dump())

        print(
            f"\n[{name}] "
            f"input_tokens≈{in_tokens} (${in_cost}) | "
            f"output_tokens≈{out_tokens} (${out_cost})"
        )

        return parsed, in_tokens + out_tokens, in_cost + out_cost

    total_tokens = 0
    total_cost = 0.0

    if state.iteration == 1:
        state.current.planner, t, c = run_and_trace("planner", run_planner_agent)
        total_tokens += t; total_cost += c

        for name, runner in [
            ("systems", run_systems_agent),
            ("ml_ai", run_ml_ai_agent),
            ("cost", run_cost_agent),
            ("product", run_product_agent),
        ]:
            output, t, c = run_and_trace(name, runner)
            setattr(state.current, name, output)
            total_tokens += t; total_cost += c

    state.current.detector, t, c = run_and_trace("detector", run_detector_agent)
    total_tokens += t; total_cost += c

    state.current.critic, t, c = run_and_trace("critic", run_critic_agent)
    total_tokens += t; total_cost += c

    state.current.synthesizer, t, c = run_and_trace("synthesizer", run_synthesizer_agent)
    total_tokens += t; total_cost += c

    state.current.gate = run_gate(state)

    print_block("GATE — OUTPUT")
    pprint(state.current.gate.model_dump())

    print_block("ITERATION SUMMARY")
    print(f"Total tokens this iteration: {total_tokens}")
    print(f"Estimated cost this iteration: ${round(total_cost, 4)}")

    return total_tokens, total_cost


# =============================================================================
# Test
# =============================================================================

def test_two_iteration_cost_trace():
    # planner/specialists frozen after iteration 1
    state = DecisionState(
        input=DecisionInput(
            decision_question=DECISION_QUESTION,
            options=OPTIONS,
            constraints=CONSTRAINTS,
        ),
        iteration=1,
        current=CurrentIteration(iteration=1),
        gate_tier=GatePolicyTier.EXPLORATION,
    )

    # ------------------
    # Iteration 1
    # ------------------
    tokens_1, cost_1 = run_iteration_with_trace(state, "ITERATION 1")

    # ------------------
    # Iteration 2 (user input)
    # ------------------
    delta = DecisionDelta(
        accepted_risks=[
            "Model performance may degrade between batch runs due to drift."
        ],
        rejected_recommendations=["hybrid"],
        notes="We accept staleness risk for v1 and prioritize operational simplicity."
    )

    state.start_next_iteration(delta)
    assert state.gate_tier == GatePolicyTier.COMMITMENT

    tokens_2, cost_2 = run_iteration_with_trace(state, "ITERATION 2")

    print_block("RUN TOTAL")
    print(f"Iteration 1 cost: ${cost_1}")
    print(f"Iteration 2 cost: ${cost_2}")
    print(f"TOTAL cost: ${round(cost_1 + cost_2, 4)}")

    # This is an observability test, not a correctness gate
    assert tokens_1 > 0
    assert tokens_2 > 0