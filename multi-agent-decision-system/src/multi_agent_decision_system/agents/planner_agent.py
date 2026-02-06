import json
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import PlannerOutput
from multi_agent_decision_system.core.state import DecisionState


PLANNER_MODEL = "gpt-5-mini"

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Planner Agent in a multi-agent technical decision system.

Your role:
- Decompose the decision into independent decision dimensions.
- Each dimension must map to exactly one specialist perspective.

Allowed perspectives:
- systems
- ml_ai
- cost
- product

Hard constraints:
- Produce at most 4 dimensions.
- Omit any perspective that is not relevant.
- Do NOT make recommendations.
- Do NOT compare options.
- Do NOT mention which option is better.

For each included perspective, produce:
- question
- why_it_matters
- key_unknowns

You may also include:
- assumptions
- clarifying_questions

Output rules:
- Output valid JSON only.
- Match the PlannerOutput schema exactly.
- No markdown.
- No extra text.

Output format (STRICT):

{{
  "agent_name": "planner",

  "systems": {{
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  }},

  "ml_ai": {{
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  }},

  "cost": {{
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  }},

  "product": {{
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  }},

  "assumptions": ["..."],
  "clarifying_questions": ["..."],

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
"""
        ),
    ]
)


def run_planner_agent(state: DecisionState) -> dict:
    """
    Planner agent:
    - Produces PlannerOutput only
    - No side effects
    - LangGraph-safe
    """

    client = OpenAI()

    messages = PLANNER_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints.model_dump(),
    )

    response = client.responses.create(
        model=PLANNER_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    output_text = response.output_text

    try:
        planner_output = PlannerOutput.model_validate_json(output_text)
    except ValidationError as e:
        raise RuntimeError(f"Planner output invalid: {e}")

    return {
        "planner": planner_output
    }