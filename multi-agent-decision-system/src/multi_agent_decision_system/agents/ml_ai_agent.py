import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import SpecialistOutput
from multi_agent_decision_system.core.state import DecisionState


ML_AI_MODEL = "gpt-5-mini"


ML_AI_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the ML / AI Agent in a multi-agent technical decision system.

Your task:
Assess the decision strictly from a machine learning and artificial intelligence feasibility and risk perspective.

You evaluate:
- Data availability, quality, and representativeness
- Feature freshness, feasibility, and temporal alignment
- Training–serving consistency and inference parity
- Model capability limits and failure modes
- Distribution shift, drift detection, and monitoring feasibility
- Evaluation realism and offline–online gap
- Inference behavior under latency constraints
- Determinism, reproducibility, and stochasticity risks

You do NOT:
- Consider infrastructure design, operational scalability, or cost
- Consider user experience, business impact, or product strategy
- Reference other agents or their outputs
- Make a final system decision

Epistemic rules:
- Be slightly pessimistic by default.
- Missing or uncertain information is a material risk.
- Confidence reflects epistemic certainty, not preference.
- High confidence (≥ 0.8) should be rare and well-justified.
- Real-time, online, or adaptive AI systems usually cap confidence at ~0.6–0.7.
- You may recommend "defer" or "insufficient_information" when key uncertainties remain.

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
  "agent_name": "ml_ai",

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

Planner context (ml_ai):
{planner_slice}

Assumptions:
{assumptions}
"""
        ),
    ]
)


def run_ml_ai_agent(state: DecisionState) -> dict:
    """
    ML / AI specialist agent.
    Produces SpecialistOutput only.
    """

    planner = state.current.planner

    planner_slice = (
        planner.ml_ai.model_dump()
        if planner and planner.ml_ai
        else None
    )

    assumptions = planner.assumptions if planner else []

    # Serialize constraints if they are Pydantic models
    options = state.input.options

    constraints = state.input.constraints
    if hasattr(constraints, "model_dump"):
        constraints = constraints.model_dump()


    messages = ML_AI_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        options=options,
        constraints=constraints or {},
        planner_slice=planner_slice or {},
        assumptions=assumptions,
    )

    client = OpenAI()

    response = client.responses.create(
        model=ML_AI_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        agent_output = SpecialistOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"ML/AI Agent output invalid: {e}")

    return {
        "ml_ai": agent_output
    }