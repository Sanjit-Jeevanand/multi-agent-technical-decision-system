from multi_agent_decision_system.graph.specialist_graph import build_specialist_graph
from multi_agent_decision_system.core.state import (
    DecisionInput,
    PlannerOutput,
    create_initial_state,
)
from multi_agent_decision_system.core.normalization import normalize_constraints

# ---- Input ----
constraints = normalize_constraints(
    {
        "latency_ms": 150,
        "team_size": 3,
        "risk_tolerance": "low",
    },
    decision_context="batch_vs_online",
)

decision_input = DecisionInput(
    decision_question="Should we use batch or online inference?",
    constraints=constraints,
)

state = create_initial_state(decision_input)

# ---- Planner output (normally produced by planner agent) ----
state.plan = PlannerOutput(
    systems={
        "question": "Can the system meet latency requirements?",
        "why_it_matters": "Online inference has tighter latency SLAs.",
        "key_unknowns": ["peak traffic"],
    },
    ml={
        "question": "Do we need real-time features?",
        "why_it_matters": "Feature freshness affects model quality.",
        "key_unknowns": ["feature staleness tolerance"],
    },
    cost={
        "question": "What is the long-term infra cost?",
        "why_it_matters": "Always-on infra increases baseline cost.",
        "key_unknowns": ["traffic growth"],
    },
    product_risk={
        "question": "What happens if predictions are delayed?",
        "why_it_matters": "User trust depends on predictability.",
        "key_unknowns": [],
    },
    assumptions=["Traffic is moderate"],
    clarifying_questions=[],
    model_used="gpt-5.1",
    estimated_tokens_in=200,
    estimated_tokens_out=300,
)

# ---- Run ----
graph = build_specialist_graph()
final_state = graph.invoke(state)

print("\n=== FINAL AGENT OUTPUTS ===")
for k, v in final_state["agent_outputs"].items():
    print(f"\n[{k.upper()}]")
    print(v.model_dump())