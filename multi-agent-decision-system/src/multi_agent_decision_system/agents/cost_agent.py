import json
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.state import (
    DecisionState,
    initialize_iteration_log,
)
from multi_agent_decision_system.core.schemas import AgentOutput


COST_AGENT_MODEL = "gpt-5.1-mini"


COST_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Cost & Complexity Agent in a multi-agent technical decision system.

Your task:
Evaluate the decision strictly from a cost, engineering effort, and operational complexity perspective.

You assess:
- Engineering build effort
- Operational and maintenance overhead
- On-call and reliability burden
- Infrastructure complexity
- Long-term cost of ownership
- Opportunity cost of engineering time

You do NOT:
- Consider ML quality, latency, or product strategy
- Reference other agents
- Make a final system decision

Biases:
- Be slightly pessimistic by default.
- Prefer simpler systems.
- Long-term operational cost matters more than initial build cost.
- Hidden complexity is a real cost.
- You may recommend "defer" or "insufficient_information".

Confidence rules:
- Confidence reflects certainty in cost assessment, not preference.
- High confidence (≥ 0.8) should be rare.
- Online or real-time systems usually cap confidence at ~0.5–0.6.

Recommendation values (exact):
- "option_a"
- "option_b"
- "hybrid"
- "defer"
- "insufficient_information"

Output rules:
- Output valid JSON only.
- Match the AgentOutput schema exactly.
- No explanations or markdown.

Output format:
{{
  "agent_name": "cost",
  "recommendation": "...",
  "confidence": 0.0,
  "benefits": ["..."],
  "risks": ["..."]
}}
"""
        ),
        (
            "human",
            """
Decision question:
{decision_question}

Constraints:
{constraints}

Planner context (cost slice):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_cost_agent(state: DecisionState) -> dict:
    """
    Cost agent returns PARTIAL state updates only.
    Safe for parallel execution in LangGraph.
    """

    # ---- Initialize iteration-aware logs (idempotent) ----
    state = initialize_iteration_log(state)
    iteration = state.termination.iteration_count

    # ---- Extract planner slice ----
    planner_slice = state.plan.cost if state.plan and state.plan.cost else None

    # ---- Build explicit agent input payload ----
    agent_input = {
        "agent_name": "cost",
        "decision_question": state.input.decision_question,
        "constraints": (
            state.input.constraints.model_dump()
            if state.input.constraints
            else {}
        ),
        "planner_slice": planner_slice,
        "assumptions": state.plan.assumptions if state.plan else [],
        "iteration": iteration,
    }

    # ---- Invoke model ----
    llm = ChatOpenAI(
        model=COST_AGENT_MODEL,
        temperature=0.3,
    )

    messages = COST_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=agent_input["constraints"],
        planner_slice=planner_slice or {},
        assumptions=agent_input["assumptions"],
    )

    response = llm.invoke(messages)
    raw_text = response.content

    # ---- Parse + validate output ----
    try:
        agent_output = AgentOutput.model_validate_json(raw_text)
    except ValidationError as e:
        raise RuntimeError(f"Cost Agent output schema validation failed: {e}")

    # ---- Return PARTIAL update only ----
    return {
        "agent_outputs": {
            "cost": agent_output,
        },
        "input_log": {
            "agent_inputs": {
                "cost": [agent_input],
            }
        },
        "output_log": {
            "agent_outputs": {
                "cost": [agent_output.model_dump()],
            }
        },
    }