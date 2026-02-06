import json
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schemas import PlannerOutput
from multi_agent_decision_system.core.state import DecisionState


PLANNER_MODEL = "gpt-5.1"


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Planner Agent in a multi-agent technical decision system.

Your role:
- Decompose the decision into a small number of independent decision dimensions.
- Each dimension will later be evaluated by a specialist agent.
- You do NOT make recommendations and do NOT compare options.

--------------------
Hard Constraints (IMPORTANT)
--------------------
- You MUST produce at most 4 decision dimensions.
- Each dimension must map cleanly to one specialist perspective:
  - systems
  - ml
  - cost
  - product_risk
- If a dimension is not relevant, omit it entirely.
- Do NOT invent extra dimensions.

--------------------
What You Produce
--------------------
For each included dimension:
- A single framing question
- Why that question matters
- Key unknowns that block confident decision-making

You may also list:
- Explicit assumptions you are making
- Clarifying questions that would reduce uncertainty

--------------------
What You MUST NOT Do
--------------------
- Do NOT recommend any option.
- Do NOT weigh trade-offs.
- Do NOT mention which option is better.
- Do NOT assume how specialists will decide.

--------------------
Cost Awareness
--------------------
You must report:
- The model name you are using
- Estimated input tokens
- Estimated output tokens

--------------------
Output Format (STRICT)
--------------------
Output valid JSON only.
Match the PlannerOutput schema exactly.
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


def run_planner(state: DecisionState) -> dict:
    """
    Execute the Planner Agent.

    - Produces a PlannerOutput
    - Logs externally for audit
    - Returns ONLY a LangGraph-safe partial update
    """

    llm = ChatOpenAI(
        model=PLANNER_MODEL,
        temperature=0,
    )

    messages = PLANNER_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        constraints=state.input.constraints.model_dump(),
    )

    response = llm.invoke(messages)

    try:
        plan = PlannerOutput.model_validate_json(response.content)
    except ValidationError as e:
        raise RuntimeError(f"Planner output invalid: {e}")

    # 🔹 External-only logging (safe, auditable, non-state)
    planner_log = {
        "agent": "planner",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints.model_dump(),
        "output": plan.model_dump(),
        "model": PLANNER_MODEL,
    }
    print(json.dumps(planner_log))

    # 🔹 LangGraph-safe partial update
    return {
        "plan": plan
    }