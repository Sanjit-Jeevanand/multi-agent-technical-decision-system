import json
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.state import (
    DecisionState,
    initialize_iteration_log,
)
from multi_agent_decision_system.core.schemas import AgentOutput


ML_AGENT_MODEL = "gpt-5.1-mini"


ML_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the ML Agent in a multi-agent technical decision system.

Your task:
Assess the decision strictly from a machine learning feasibility and risk perspective.

You evaluate:
- Data availability and quality
- Feature freshness and feasibility
- Training–serving consistency
- Distribution shift and monitoring risk
- Evaluation realism

You do NOT:
- Consider infrastructure, cost, or product strategy
- Reference other agents
- Make a final system decision

Epistemic rules:
- Be slightly pessimistic by default.
- Missing or uncertain data is a risk.
- High confidence (≥ 0.8) should be rare.
- Real-time or online ML usually caps confidence at ~0.6–0.7.
- You may recommend "defer" or "insufficient_information".

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
  "agent_name": "ml",
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

Planner context (ML slice):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_ml_agent(state: DecisionState) -> dict:

    state = initialize_iteration_log(state)
    iteration = state.termination.iteration_count

    # ---- Extract planner slice ----
    planner_slice = None
    if state.plan and state.plan.ml:
        planner_slice = state.plan.ml

    # ---- Explicit ML agent input logging ----
    input_log_update = {
        "ml": {
            "agent_name": "ml",
            "decision_question": state.input.decision_question,
            "constraints": state.input.constraints,
            "planner_slice": planner_slice,
            "assumptions": state.plan.assumptions if state.plan else [],
            "iteration": iteration,
        }
    }

    # ---- Invoke model ----
    llm = ChatOpenAI(
        model=ML_AGENT_MODEL,
        temperature=0.3,
    )

    messages = ML_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints or {},
        planner_slice=planner_slice or {},
        assumptions=state.plan.assumptions if state.plan else [],
    )

    response = llm.invoke(messages)
    raw_text = response.content

    # ---- Parse + validate output ----
    try:
        agent_output = AgentOutput.model_validate_json(raw_text)
    except ValidationError as e:
        raise RuntimeError(f"ML Agent output schema validation failed: {e}")

    # ---- Partial update and append-only logging ----
    return {
        "agent_outputs": {"ml": agent_output},
        "input_log": {
            "agent_inputs": input_log_update
        },
        "output_log": {
            "agent_outputs": {
                "ml": [agent_output.model_dump()]
            }
        }
    }