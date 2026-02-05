from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from multi_agent_decision_system.core.state import (
    DecisionState,
    initialize_iteration_log,
)
from multi_agent_decision_system.core.schemas import PlannerOutput


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

This is for audit and cost tracking only.

--------------------
Output Format (STRICT)
--------------------
You MUST output valid JSON only.
You MUST match this schema exactly:

{
  "systems": {
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  },
  "ml": {
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  },
  "cost": {
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  },
  "product_risk": {
    "question": "...",
    "why_it_matters": "...",
    "key_unknowns": ["..."]
  },
  "assumptions": ["..."],
  "clarifying_questions": ["..."],
  "model_used": "gpt-5.1",
  "estimated_tokens_in": <integer>,
  "estimated_tokens_out": <integer>
}

Omit any dimension that is not relevant.
Do not include explanations outside this JSON.
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

    # ---- Explicit planner agent input logging ----
    state.input_log.agent_inputs["planner"] = {
        "agent_name": "planner",
        "decision_question": state.input.decision_question,
        "constraints": state.input.constraints,
        "iteration": iteration,
    }

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