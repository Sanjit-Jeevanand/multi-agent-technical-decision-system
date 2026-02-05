from langgraph.graph import StateGraph, END

from multi_agent_decision_system.core.state import DecisionState
from multi_agent_decision_system.agents.systems_agent import run_systems_agent


def build_specialist_graph():
    graph = StateGraph(DecisionState)

    graph.add_node("systems", run_systems_agent)

    # Future:
    # graph.add_node("ml", run_ml_agent)
    # graph.add_node("cost", run_cost_agent)
    # graph.add_node("product_risk", run_product_agent)

    graph.set_entry_point("systems")
    graph.add_edge("systems", END)

    return graph.compile()