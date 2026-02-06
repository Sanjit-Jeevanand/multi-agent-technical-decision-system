from langgraph.graph import StateGraph, END

from multi_agent_decision_system.core.state import DecisionState
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_agent import run_ml_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_risk_agent import run_product_risk_agent


def build_specialist_graph():
    """
    Parallel execution graph for all specialist agents.
    Each agent operates independently and writes into DecisionState.
    """
    graph = StateGraph(DecisionState)

    # Specialist agent nodes
    graph.add_node("systems", run_systems_agent)
    graph.add_node("ml", run_ml_agent)
    graph.add_node("cost", run_cost_agent)
    graph.add_node("product_risk", run_product_risk_agent)

    # Entry point fans out to all specialists
    graph.set_entry_point("systems")

    graph.add_edge("systems", "ml")
    graph.add_edge("systems", "cost")
    graph.add_edge("systems", "product_risk")

    # All specialists terminate independently
    graph.add_edge("ml", END)
    graph.add_edge("cost", END)
    graph.add_edge("product_risk", END)

    return graph.compile()