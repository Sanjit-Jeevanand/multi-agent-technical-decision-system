from multi_agent_decision_system.core.state_v2 import (
    DecisionState,
    DecisionDelta,
)
from multi_agent_decision_system.agents.gate_agent import GatePolicyTier
from multi_agent_decision_system.agents.planner_agent import run_planner_agent
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_ai_agent import run_ml_ai_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_agent import run_product_agent
from multi_agent_decision_system.agents.detector_agent import run_detector_agent
from multi_agent_decision_system.agents.critic_agent import run_critic_agent
from multi_agent_decision_system.agents.synthesizer_agent import run_synthesizer_agent
from multi_agent_decision_system.agents.gate_agent import run_gate


def authority_frozen(state: DecisionState) -> bool:
    # Authority is frozen after iteration 1 by gate tier, not counters.
    return state.gate_tier != GatePolicyTier.EXPLORATION


def run_single_iteration(state: DecisionState) -> DecisionState:
    """
    Runs exactly ONE full decision iteration.
    Does NOT loop.
    """

    if state.gate_tier == GatePolicyTier.EXPLORATION:
        # Planner
        state.current.planner = run_planner_agent(state)["planner"]

        # Specialists
        state.current.systems = run_systems_agent(state)["systems"]
        state.current.ml_ai = run_ml_ai_agent(state)["ml_ai"]
        state.current.cost = run_cost_agent(state)["cost"]
        state.current.product = run_product_agent(state)["product"]

    # Reasoning
    state.current.detector = run_detector_agent(state)["detector"]

    # Critic is advisory only after iteration 1 (authority frozen by gate tier)
    state.current.critic = run_critic_agent(state)["critic"]

    # After iteration 1, Synthesizer + Gate have final authority
    # Decision
    state.current.synthesizer = run_synthesizer_agent(state)["synthesizer"]

    # Gate (pure logic, no LLM)
    state.current.gate = run_gate(state)

    # Terminal bookkeeping
    if state.current.gate.decision.approved:
        state.approved = True
        state.final_decision = state.current.synthesizer

    return state

def start_decision(state: DecisionState) -> DecisionState:
    return run_single_iteration(state)

def continue_decision(
    state: DecisionState,
    delta: DecisionDelta,
) -> DecisionState:

    if state.approved:
        return state  

    if delta is None:
        return state  

    # Snapshot + mutate input
    state.start_next_iteration(delta)

    # Run ONE more iteration
    return run_single_iteration(state)
