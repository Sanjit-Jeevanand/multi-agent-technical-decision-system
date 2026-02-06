import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import SpecialistOutput
from multi_agent_decision_system.core.state import DecisionState


PRODUCT_MODEL = "gpt-5-mini"


PRODUCT_AGENT_PROMPT = ChatPromptTemplate.from_messages(
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
- Match the SpecialistOutput schema exactly.
- No explanations, markdown, or extra text.

List constraints:
- benefits: 2–4 items maximum
- risks: 2–4 items maximum
- Each list item must be a single atomic sentence.
- Do NOT use semicolons.
- Do NOT use conjunctions ("and", "or").

────────────────────────────
Output Format (STRICT)
────────────────────────────
{{
  "agent_name": "product",

  "recommendation": "<option_a | option_b | hybrid | defer | insufficient_information>",
  "confidence": 0.0,

  "benefits": [
    "...",
    "..."
  ],

  "risks": [
    "...",
    "..."
  ]
}}
"""
        ),
        (
            "human",
            """
Decision question:
{decision_question}

Options:
{options}

Constraints:
{constraints}

Planner context (product):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_product_agent(state: DecisionState) -> dict:
    """
    Product & risk specialist agent.
    Produces SpecialistOutput only.
    """

    planner = state.current.planner

    planner_slice = (
        planner.product.model_dump()
        if planner and planner.product
        else None
    )

    assumptions = planner.assumptions if planner else []

    constraints_dump = state.input.constraints.model_dump() if state.input.constraints else {}

    messages = PRODUCT_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        options=state.input.options,
        constraints=constraints_dump,
        planner_slice=planner_slice or {},
        assumptions=assumptions,
    )

    client = OpenAI()

    response = client.responses.create(
        model=PRODUCT_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        agent_output = SpecialistOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"Product Agent output invalid: {e}")

    return {
        "product": agent_output
    }