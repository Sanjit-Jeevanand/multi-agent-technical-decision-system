from langgraph.graph import StateGraph
from multi_agent_decision_system.core.state import DecisionState
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_agent import run_ml_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_risk_agent import run_product_risk_agent


def build_specialist_graph():
    graph = StateGraph(DecisionState)

    # Start (identity)
    graph.add_node("start", lambda state: state)

    # Specialists
    graph.add_node("systems", run_systems_agent)
    graph.add_node("ml", run_ml_agent)
    graph.add_node("cost", run_cost_agent)
    graph.add_node("product_risk", run_product_risk_agent)

    # Join (identity)
    graph.add_node("join", lambda state: state)

    graph.set_entry_point("start")

    # Fan-out
    graph.add_edge("start", "systems")
    graph.add_edge("start", "ml")
    graph.add_edge("start", "cost")
    graph.add_edge("start", "product_risk")

    # Fan-in
    graph.add_edge("systems", "join")
    graph.add_edge("ml", "join")
    graph.add_edge("cost", "join")
    graph.add_edge("product_risk", "join")

    # 🔴 THIS IS THE LINE YOU ARE MISSING
    graph.set_finish_point("join")

    return graph.compile()