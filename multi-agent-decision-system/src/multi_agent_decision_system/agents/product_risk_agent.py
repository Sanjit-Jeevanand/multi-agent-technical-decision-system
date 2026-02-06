import json
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.state import (
    DecisionState,
    initialize_iteration_log,
)
from multi_agent_decision_system.core.schemas import AgentOutput


PRODUCT_RISK_AGENT_MODEL = "gpt-5.1-mini"


PRODUCT_RISK_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Product & Risk Agent in a multi-agent technical decision system.

Your task:
Evaluate the decision strictly from a user impact, safety, and risk perspective.

You assess:
- Potential user harm
- Failure visibility and detectability
- Trust and reputational risk
- Regulatory or compliance exposure
- Rollout and rollback safety
- Blast radius of failures
- Reversibility of mistakes

You do NOT:
- Optimize for cost, ML quality, or infrastructure
- Reference other agents
- Make a final system decision

Biases:
- Be slightly pessimistic by default.
- User harm matters more than system elegance.
- Silent or hidden failures are high risk.
- Reversibility matters more than speed.
- You may recommend "defer" or "insufficient_information".

Confidence rules:
- Confidence reflects certainty about user impact, not preference.
- High confidence (≥ 0.8) should be rare.
- User-facing or automated decisions usually cap confidence at ~0.5–0.6.

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
  "agent_name": "product_risk",
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

Planner context (product/risk slice):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_product_risk_agent(state: DecisionState) -> DecisionState:
    """
    Execute the Product & Risk Agent with strict input/output logging.
    """

    state = initialize_iteration_log(state)
    iteration = state.termination.iteration_count

    # ---- Extract planner slice ----
    planner_slice = None
    if state.plan and state.plan.product_risk:
        planner_slice = state.plan.product_risk

    # ---- Explicit product/risk agent input logging ----
    state.input_log.agent_inputs["product_risk"] = {
        "agent_name": "product_risk",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints,
        "planner_slice": planner_slice,
        "assumptions": state.plan.assumptions if state.plan else [],
        "iteration": iteration,
    }

    # ---- Invoke model ----
    llm = ChatOpenAI(
        model=PRODUCT_RISK_AGENT_MODEL,
        temperature=0.3,
    )

    messages = PRODUCT_RISK_AGENT_PROMPT.format_messages(
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
        raise RuntimeError(
            f"Product/Risk Agent output schema validation failed: {e}"
        )

    # ---- Write authoritative state ----
    state.agent_outputs["product_risk"] = agent_output

    # ---- Log output ----
    state.output_log.agent_outputs["product_risk"] = agent_output.model_dump()

    return state