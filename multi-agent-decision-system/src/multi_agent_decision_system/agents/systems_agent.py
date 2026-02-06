import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import SpecialistOutput
from multi_agent_decision_system.core.state import DecisionState


SYSTEMS_MODEL = "gpt-5-mini"


SYSTEMS_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Systems Agent.

Role:
- Evaluate the decision strictly from a systems-only perspective.
- Focus exclusively on scalability, reliability, latency, and operational risk.

Prohibitions:
- Do NOT consider cost, machine learning quality, or product strategy.
- Do NOT assume the existence of other agents.
- Do NOT attempt synthesis or integration of perspectives.

Decision framing:
- Recommend the safest operational choice, not necessarily the best overall.
- Prioritize operational safety, simplicity, and failure containment.

Style constraints:
- Be concise and technical.
- Avoid narrative explanations.
- Prefer short, declarative statements over descriptive prose.

Confidence calibration rules:
- Confidence reflects certainty about systems aspects.
- High confidence (≥ 0.8) should be rare.
- If assumptions about scale or environment are required, cap confidence at ~0.6–0.7.
- Conservative bias is required.

Recommendation label rules:
- Provide exactly one recommendation.
- Use ONLY canonical labels:
  - "option_a"
  - "option_b"
  - "hybrid"
  - "defer"
  - "insufficient_information"

Output contract:
- Output MUST be valid JSON.
- Output MUST conform exactly to the SpecialistOutput schema.

List constraints:
- benefits: 2–4 items
- risks: 2–4 items
- Each item must be a single atomic sentence.

────────────────────────────
Output Format (STRICT)
────────────────────────────
{{
  "agent_name": "systems",

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

Constraints:
{constraints}

Planner context (systems):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_systems_agent(state: DecisionState) -> dict:
    """
    Systems specialist agent.
    Produces SpecialistOutput only.
    """

    planner = state.current.planner

    planner_slice = (
        planner.systems.model_dump()
        if planner and planner.systems
        else None
    )

    assumptions = planner.assumptions if planner else []

    messages = SYSTEMS_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints,
        planner_slice=planner_slice or {},
        assumptions=assumptions,
    )

    client = OpenAI()

    response = client.responses.create(
        model=SYSTEMS_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        agent_output = SpecialistOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"Systems Agent output invalid: {e}")

    return {
        "systems": agent_output
    }