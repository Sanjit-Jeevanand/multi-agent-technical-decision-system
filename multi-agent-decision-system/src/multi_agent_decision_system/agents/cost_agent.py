import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import SpecialistOutput
from multi_agent_decision_system.core.state import DecisionState


COST_MODEL = "gpt-5-mini"


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
  "agent_name": "cost",

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

Planner context (cost):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_cost_agent(state: DecisionState) -> dict:
    """
    Cost & complexity specialist agent.
    Produces SpecialistOutput only.
    """

    planner = state.current.planner

    planner_slice = (
        planner.cost.model_dump()
        if planner and planner.cost
        else None
    )

    assumptions = planner.assumptions if planner else []

    options = state.input.options

    constraints = state.input.constraints
    if hasattr(constraints, "model_dump"):
        constraints = constraints.model_dump()

    messages = COST_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        options=options,
        constraints=constraints or {},
        planner_slice=planner_slice or {},
        assumptions=assumptions,
    )

    client = OpenAI()

    response = client.responses.create(
        model=COST_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        agent_output = SpecialistOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"Cost Agent output invalid: {e}")

    return {
        "cost": agent_output
    }