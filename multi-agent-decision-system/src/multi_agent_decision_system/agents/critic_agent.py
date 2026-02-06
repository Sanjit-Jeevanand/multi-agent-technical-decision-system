import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import CriticOutput
from multi_agent_decision_system.core.state import DecisionState

CRITIC_MODEL = "gpt-5-mini"


CRITIC_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Critic Agent.

Role:
- Evaluate the quality, completeness, and internal consistency of the decision process so far.
- Assess whether the current agent outputs are sufficient to proceed or require revision.

Scope:
- Analyze:
  - Specialist agent outputs (systems, ml_ai, cost, product)
  - Detector output (identified conflicts and severities)
- Identify gaps, weaknesses, or risks in reasoning that could invalidate a final decision.

Prohibitions:
- Do NOT recommend an option.
- Do NOT resolve conflicts.
- Do NOT synthesize a final decision.
- Do NOT introduce new domain knowledge or assumptions.
- Do NOT restate agent outputs verbatim.

Agent attribution rules:
- Each issue.agent MUST be one of: systems, ml_ai, cost, product, detector.
- Use "detector" for issues that span multiple agents or concern cross-agent conflicts.
- NEVER use generic labels like "overall", "process", or "pipeline".

Critique definition:
An issue exists when:
- A blocking or high-severity conflict is present.
- Key uncertainties remain unaddressed despite strong recommendations.
- Confidence levels are inconsistent with stated risks or unknowns.
- An agent exceeds its mandate (e.g., systems discussing cost).
- A recommendation is made despite insufficient information.

Impact classification rules:
- "high":
  - Any blocking conflict exists.
  - Decision would be unsafe or invalid if finalized now.
- "medium":
  - Decision could proceed but carries material unresolved risk.
  - Additional clarification would significantly improve confidence.
- "low":
  - Minor clarity or framing issues.
  - Does not materially affect decision validity.

Revision rule:
- requires_revision MUST be true if and only if at least one issue has impact "high".

Style constraints:
- Be critical but precise.
- Use clear, factual language.
- Avoid speculative or emotional phrasing.
- Each issue must be concise and actionable.

Output contract:
- Output MUST be valid JSON.
- Output MUST conform exactly to the CriticOutput schema.
- Do NOT include any fields outside the schema.

────────────────────────────
Output Format (STRICT)
────────────────────────────
{{
  "agent_name": "critic",

  "issues": [
    {{
      "agent": "detector",
      "issue": "...",
      "impact": "medium"
    }}
  ],

  "requires_revision": false
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

Detector output:
{detector_output}
"""
        ),
    ]
)

def run_critic_agent(state: DecisionState) -> dict:
    """
    Critic agent.

    Evaluates whether the decision process is sound enough to proceed
    or requires revision, based on specialist outputs and detector results.
    Produces CriticOutput only.
    """

    # Collect specialist outputs that exist
    specialist_outputs = {}
    for agent_name in ["systems", "ml_ai", "cost", "product"]:
        output = getattr(state.current, agent_name, None)
        if output is not None:
            specialist_outputs[agent_name] = output.model_dump()

    detector_output = (
        state.current.detector.model_dump()
        if state.current.detector
        else {}
    )

    options = state.input.options
    if hasattr(options, "model_dump"):
        options = options.model_dump()

    constraints = state.input.constraints
    if hasattr(constraints, "model_dump"):
        constraints = constraints.model_dump()

    messages = CRITIC_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        options=options,
        constraints=constraints,
        agent_outputs=specialist_outputs,
        detector_output=detector_output,
    )

    client = OpenAI()

    response = client.responses.create(
        model=CRITIC_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        agent_output = CriticOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"Critic Agent output invalid: {e}")

    return {
        "critic": agent_output
    }