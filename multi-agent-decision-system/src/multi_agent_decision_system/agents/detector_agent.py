import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import DetectorOutput
from multi_agent_decision_system.core.state import DecisionState


DETECTOR_MODEL = "gpt-5-mini"


DETECTOR_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Detector Agent.

Role:
- Identify conflicts, inconsistencies, or incompatibilities across specialist agent outputs.
- Compare agent conclusions and confidence levels without judgment or resolution.

Scope:
- Analyze recommendations from systems, ml_ai, cost, and product agents.
- Detect logical, semantic, or feasibility conflicts that may affect decision validity.

Prohibitions:
- Do NOT recommend an option.
- Do NOT resolve conflicts.
- Do NOT synthesize or integrate perspectives.
- Do NOT introduce new assumptions or domain knowledge.

Conflict definition:
A conflict exists when:
- Two or more agents recommend incompatible outcomes.
- One agent recommends deferring or insufficient information while others recommend proceeding.
- Confidence or risk profiles materially contradict each other.
- Feasibility or safety claims directly oppose each other.

Severity classification rules:
- "high":
  - Any agent recommends "defer" or "insufficient_information" while others recommend an option.
  - Conflicts involving feasibility, safety, or correctness (typically systems or ml_ai).
- "medium":
  - Direct disagreement between option_a and option_b.
  - Large confidence discrepancies on the same option.
- "low":
  - Hybrid vs single-option disagreements.
  - Differences that reflect preference rather than feasibility.

Blocking rule:
- has_blocking_conflicts MUST be true if and only if at least one conflict has severity "high".

Style constraints:
- Be neutral and descriptive.
- Avoid normative or prescriptive language.
- Each conflict description must be concise and factual.

Output contract:
- Output MUST be valid JSON.
- Output MUST conform exactly to the DetectorOutput schema.
- Do NOT include any fields outside the schema.

────────────────────────────
Output Format (STRICT)
────────────────────────────
{{
  "agent_name": "detector",

  "conflicts": [
    {{
      "agents_involved": ["systems", "ml_ai"],
      "description": "...",
      "severity": "medium"
    }}
  ],

  "has_blocking_conflicts": false
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

Specialist agent outputs:
{agent_outputs}
"""
        ),
    ]
)


def run_detector_agent(state: DecisionState) -> dict:
    """
    Detector agent.
    Identifies conflicts across specialist agents.
    Produces DetectorOutput only.
    """

    # Collect specialist outputs (only those that exist)
    specialist_outputs = {}

    for agent_name in ["systems", "ml_ai", "cost", "product"]:
        agent_output = getattr(state.current, agent_name, None)
        if agent_output is not None:
            specialist_outputs[agent_name] = agent_output.model_dump()

    options = state.input.options
    if hasattr(options, "model_dump"):
        options = options.model_dump()

    constraints = state.input.constraints
    if hasattr(constraints, "model_dump"):
        constraints = constraints.model_dump()

    messages = DETECTOR_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        options=options,
        constraints=constraints,
        agent_outputs=specialist_outputs,
    )

    client = OpenAI()

    response = client.responses.create(
        model=DETECTOR_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        detector_output = DetectorOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"Detector Agent output invalid: {e}")

    # Suppress conflicts already acknowledged by the user in the previous iteration
    accepted_risks = []
    if hasattr(state, "history") and state.history:
        last_snapshot = state.history[-1]
        if last_snapshot.delta_applied:
            accepted_risks = last_snapshot.delta_applied.accepted_risks or []

    if accepted_risks:
        filtered_conflicts = []
        for conflict in detector_output.conflicts:
            if not any(
                risk.lower() in conflict.description.lower()
                for risk in accepted_risks
            ):
                filtered_conflicts.append(conflict)

        detector_output.conflicts = filtered_conflicts
        detector_output.has_blocking_conflicts = any(
            c.severity == "high" for c in detector_output.conflicts
        )

    return {
        "detector": detector_output
    }