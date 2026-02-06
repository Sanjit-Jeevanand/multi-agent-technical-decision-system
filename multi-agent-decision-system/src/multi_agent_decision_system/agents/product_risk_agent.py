import json
from pydantic import ValidationError
from openai import OpenAI

from multi_agent_decision_system.core.schemas import AgentOutput
from multi_agent_decision_system.core.state import DecisionState


PRODUCT_RISK_MODEL = "gpt-5-mini"


PRODUCT_RISK_PROMPT_STR = """
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

List constraints (IMPORTANT):
- benefits: 2–4 items maximum
- risks: 2–4 items maximum
- Each list item must be a single, atomic sentence
- Do NOT use semicolons or conjunctions ("and", "or") inside list items

JSON correctness rules (CRITICAL):
- Do not include trailing commas.
- Do not include comments.
- Do not include extra whitespace outside JSON.
- Ensure all arrays and objects are properly closed.

Output format:
{{
  "agent_name": "product_risk",
  "recommendation": "...",
  "confidence": 0.0,
  "benefits": ["..."],
  "risks": ["..."]
}}

Decision question:
{decision_question}

Constraints:
{constraints}

Planner context (product/risk slice):
{planner_slice}

Assumptions:
{assumptions}
"""


def run_product_risk_agent(state: DecisionState) -> dict:
    """
    Execute the Product & Risk Agent.
    Returns a LangGraph-safe partial update.
    """

    planner_slice = (
        state.plan.product_risk.model_dump()
        if state.plan and state.plan.product_risk
        else None
    )

    client = OpenAI()
    prompt = PRODUCT_RISK_PROMPT_STR.format(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints.model_dump(),
        planner_slice=planner_slice or {},
        assumptions=state.plan.assumptions if state.plan else [],
    )
    response = client.responses.create(
        model=PRODUCT_RISK_MODEL,
        input=prompt,
        reasoning={"effort": "minimal"},
    )
    content = response.output_text

    try:
        agent_output = AgentOutput.model_validate_json(content)
    except ValidationError as e:
        raise RuntimeError(f"Product/Risk Agent output invalid: {e}")

    # 🔹 Side-channel JSON logging (NOT LangGraph state)
    product_risk_log = {
        "agent": "product_risk",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints.model_dump(),
        "planner_slice": planner_slice,
        "assumptions": state.plan.assumptions if state.plan else [],
        "output": agent_output.model_dump(),
        "model": PRODUCT_RISK_MODEL,
    }
    print(json.dumps(product_risk_log))

    # 🔹 LangGraph-safe partial update
    return {
        "agent_outputs": {
            "product_risk": agent_output
        }
    }