import json
from pydantic import ValidationError
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schemas import AgentOutput
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
    Execute the Cost & Complexity Agent.
    Returns a LangGraph-safe partial update.
    """

    planner_slice = (
        state.plan.cost.model_dump()
        if state.plan and state.plan.cost
        else None
    )

    client = OpenAI()

    messages = COST_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints.model_dump(),
        planner_slice=planner_slice or {},
        assumptions=state.plan.assumptions if state.plan else [],
    )

    response = client.responses.create(
        model=COST_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    try:
        agent_output = AgentOutput.model_validate_json(
            response.output_text
        )
    except ValidationError as e:
        raise RuntimeError(f"Cost Agent output invalid: {e}")

    # 🔹 Side-channel JSON logging (NOT LangGraph state)
    cost_log = {
        "agent": "cost",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints.model_dump(),
        "planner_slice": planner_slice,
        "assumptions": state.plan.assumptions if state.plan else [],
        "output": agent_output.model_dump(),
        "model": COST_MODEL,
    }
    print(json.dumps(cost_log))

    # 🔹 LangGraph-safe partial update
    return {
        "agent_outputs": {
            "cost": agent_output
        }
    }