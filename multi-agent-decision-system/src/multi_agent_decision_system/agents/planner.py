from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from multi_agent_decision_system.core.state import (
    DecisionState,
    initialize_iteration_log,
)
from multi_agent_decision_system.core.schemas import PlannerOutput


PLANNER_MODEL = "gpt-4.1"


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Planner Agent in a multi-agent technical decision system.

Your job is to FRAME the decision, not to decide it.

Rules you must follow:
- Identify AT MOST 4 decision dimensions.
- Each dimension must map to exactly one agent:
  - systems
  - ml
  - cost
  - product_risk
- Do NOT recommend an option.
- Do NOT weigh trade-offs.
- Prefer clarity over exhaustiveness.
- If information is missing, ask clarifying questions.

Output rules (MANDATORY):
- Output MUST be valid JSON.
- Output MUST match the PlannerOutput schema exactly.
- Do NOT include prose, explanations, or markdown.
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


def run_planner(state: DecisionState) -> DecisionState:

    state = initialize_iteration_log(state)

    iteration = state.termination.iteration_count

    llm = ChatOpenAI(
        model=PLANNER_MODEL,
        temperature=0.4,
    )

    # ---- Log planner input (iteration-aware) ----
    state.input_log.iterations[iteration].planner_input = {
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints,
        "iteration": iteration,
    }

    # ---- Invoke model ----
    response = llm.invoke(
        PLANNER_PROMPT.format_messages(
            decision_question=state.input.decision_question,
            constraints=state.input.constraints or {},
        )
    )

    raw_text = response.content
    usage = response.response_metadata.get("token_usage", {})

    # ---- Parse + validate schema ----
    try:
        planner_output = PlannerOutput.model_validate_json(raw_text)
    except ValidationError as e:
        raise RuntimeError(f"Planner output schema validation failed: {e}")

    # Populate mandatory cost metadata
    planner_output.model_used = PLANNER_MODEL
    planner_output.estimated_tokens_in = usage.get("prompt_tokens", 0)
    planner_output.estimated_tokens_out = usage.get("completion_tokens", 0)

    # ---- Write authoritative state ----
    state.plan = planner_output

    # ---- Log planner output ----
    state.output_log.planner_output = planner_output.model_dump()

    # ---- Cost accounting (per iteration) ----
    state.output_log.termination_outputs.append(
        {
            "iteration": iteration,
            "component": "planner",
            "model": PLANNER_MODEL,
            "tokens_in": usage.get("prompt_tokens", planner_output.estimated_tokens_in),
            "tokens_out": usage.get("completion_tokens", planner_output.estimated_tokens_out),
        }
    )

    return state