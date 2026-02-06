# src/multi_agent_decision_system/graph/decision_graph.py

from langgraph.graph import StateGraph, END

from multi_agent_decision_system.core.state import DecisionState
from multi_agent_decision_system.agents import (
    run_planner,
    run_systems_agent,
    run_ml_agent,
    run_cost_agent,
    run_product_agent,
    run_detector,
    run_critic,
    run_synthesizer,
    run_gate,
)

MAX_ITERATIONS = 3


# -------------------------------------------------
# Iteration condition
# -------------------------------------------------

def needs_iteration(state: DecisionState) -> str:
    if state.iteration >= MAX_ITERATIONS:
        return "synthesize"

    if state.detector_output and state.detector_output.has_conflicts:
        return "iterate"

    if state.critic_feedback and state.critic_feedback.requires_revision:
        return "iterate"

    return "synthesize"


def gate_decision(state: DecisionState) -> str:
    if state.gate.approved:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "iterate"


# -------------------------------------------------
# Build Graph
# -------------------------------------------------

def build_decision_graph() -> StateGraph:
    graph = StateGraph(DecisionState)

    # Nodes
    graph.add_node("planner", run_planner)

    graph.add_node("systems", run_systems_agent)
    graph.add_node("ml_ai", run_ml_agent)
    graph.add_node("cost", run_cost_agent)
    graph.add_node("product", run_product_agent)

    graph.add_node("detector", run_detector)
    graph.add_node("critic", run_critic)
    graph.add_node("synthesizer", run_synthesizer)
    graph.add_node("gate", run_gate)

    # Edges
    graph.set_entry_point("planner")

    # Planner → parallel specialists
    graph.add_edge("planner", "systems")
    graph.add_edge("planner", "ml_ai")
    graph.add_edge("planner", "cost")
    graph.add_edge("planner", "product")

    # Specialists → detector
    graph.add_edge("systems", "detector")
    graph.add_edge("ml_ai", "detector")
    graph.add_edge("cost", "detector")
    graph.add_edge("product", "detector")

    # Detector → critic
    graph.add_edge("detector", "critic")

    # Critic → conditional
    graph.add_conditional_edges(
        "critic",
        needs_iteration,
        {
            "iterate": "systems",
            "synthesize": "synthesizer",
        },
    )

    # Synthesizer → gate
    graph.add_edge("synthesizer", "gate")

    # Gate → conditional
    graph.add_conditional_edges(
        "gate",
        gate_decision,
        {
            "iterate": "systems",
            "end": END,
        },
    )

    return graph.compile()