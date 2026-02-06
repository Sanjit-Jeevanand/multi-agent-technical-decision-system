import json
from pydantic import ValidationError
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schemas import AgentOutput
from multi_agent_decision_system.core.state import DecisionState


ML_MODEL = "gpt-5-mini"


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

    planner_slice = (
        state.plan.ml.model_dump()
        if state.plan and state.plan.ml
        else None
    )

    client = OpenAI()

    messages = ML_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints.model_dump(),
        planner_slice=planner_slice or {},
        assumptions=state.plan.assumptions if state.plan else [],
    )

    response = client.responses.create(
        model=ML_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    output_text = response.output_text

    try:
        agent_output = AgentOutput.model_validate_json(output_text)
    except ValidationError as e:
        raise RuntimeError(f"ML Agent output invalid: {e}")

    # 🔹 Side-channel JSON logging (NOT LangGraph state)
    ml_log = {
        "agent": "ml",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints.model_dump(),
        "planner_slice": planner_slice,
        "assumptions": state.plan.assumptions if state.plan else [],
        "output": agent_output.model_dump(),
        "model": ML_MODEL,
    }
    print(json.dumps(ml_log))

    # 🔹 LangGraph-safe partial update
    return {
        "agent_outputs": {
            "ml": agent_output
        }
    }