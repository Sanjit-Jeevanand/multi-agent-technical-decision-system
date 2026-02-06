import json
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schemas import AgentOutput
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
- Prioritize conservative, simple, and robust solutions.

Confidence calibration rules:
- Confidence reflects your certainty about systems aspects, not personal preference.
- High confidence (≥ 0.8) should be rare and reserved for well-understood scenarios.
- When assumptions about scale or environment are required, cap confidence at approximately 0.6–0.7.
- Maintain a conservative bias favoring operational simplicity and risk avoidance.
- It is acceptable to recommend "defer" or "insufficient_information" if confident assessment is not possible.

Recommendation label rules:
- Provide exactly one recommendation.
- Use canonical labels for recommendations.
- Avoid ambiguous or multiple recommendations.

Output contract:
- Output MUST be valid JSON.
- Output MUST conform exactly to the AgentOutput schema.

Output format:
{{
  "agent_name": "systems",
  "recommendation": "...",
  "confidence": 0.6,
  "benefits": ["..."],
  "risks": ["..."]
}}
"""
        ),
        (
            "human",
            """
Decision question:
{question}

Constraints:
{constraints}

Planner framing:
{plan}
"""
        ),
    ]
)


def run_systems_agent(state: DecisionState) -> dict:

    planner_slice = (
        state.plan.systems.model_dump()
        if state.plan and state.plan.systems
        else None
    )

    llm = ChatOpenAI(
        model=SYSTEMS_MODEL,
        temperature=0,
    )

    messages = SYSTEMS_AGENT_PROMPT.format_messages(
        question=state.input.decision_question,
        constraints=state.input.constraints.model_dump(),
        plan=planner_slice or {},
    )

    response = llm.invoke(messages)

    try:
        agent_output = AgentOutput.model_validate_json(response.content)
    except ValidationError as e:
        raise RuntimeError(f"Systems Agent output invalid: {e}")

    systems_log = {
        "agent": "systems",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints.model_dump(),
        "planner_slice": planner_slice,
        "assumptions": state.plan.assumptions if state.plan else [],
        "output": agent_output.model_dump(),
        "model": SYSTEMS_MODEL,
    }
    print(json.dumps(systems_log))

    # 🔹 LangGraph-safe partial update
    return {
        "agent_outputs": {
            "systems": agent_output
        }
    }