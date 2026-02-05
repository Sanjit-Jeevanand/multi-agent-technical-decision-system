from langgraph.graph import StateGraph, END

from multi_agent_decision_system.core.state import DecisionState
from multi_agent_decision_system.agents.planner import run_planner


def build_planner_graph():

    graph = StateGraph(DecisionState)

    # ------------------
    # Nodes
    # ------------------
    graph.add_node("planner", run_planner)

    # ------------------
    # Edges
    # ------------------
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)

    return graph.compile()