import json
from openai import OpenAI
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from multi_agent_decision_system.core.schema import SynthesizerOutput
from multi_agent_decision_system.core.state import DecisionState


SYNTHESIZER_MODEL = "gpt-5"


# =========================
# Prompt
# =========================

SYNTHESIZER_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Synthesizer Agent.

Your task is to produce ONE final decision by synthesizing prior agent outputs.

This is a DECISION role, not an analysis role.

━━━━━━━━━━━━━━━━━━━━
PRIMARY OBJECTIVE
━━━━━━━━━━━━━━━━━━━━
Select the most defensible recommendation under uncertainty, given:
- specialist analyses,
- detected conflicts,
- critic assessment.

Perfect certainty is NOT required.
Explicit trade-offs ARE required.

━━━━━━━━━━━━━━━━━━━━
ALLOWED INPUTS
━━━━━━━━━━━━━━━━━━━━
You may consider ONLY:
- Specialist agent outputs (systems, ml_ai, cost, product)
- Detector output (conflicts + severity)
- Critic output (issues + requires_revision)

━━━━━━━━━━━━━━━━━━━━
STRICT PROHIBITIONS
━━━━━━━━━━━━━━━━━━━━
- Do NOT invent new options.
- Do NOT introduce new assumptions or facts.
- Do NOT restate agent outputs verbatim.
- Do NOT explain your reasoning process.
- Do NOT hedge with vague language.
- Do NOT output anything outside the schema.

━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━
You MUST choose exactly ONE of:
- "option_a"
- "option_b"
- "hybrid"
- "defer"
- "insufficient_information"

Interpretation rules:
- Conflicts do NOT automatically imply deferral.
- High-severity conflicts require acknowledgement, not paralysis.
- Proceed when risks are bounded, reversible, or operationally containable.
- Defer ONLY when unresolved unknowns make any action irresponsible.

━━━━━━━━━━━━━━━━━━━━
RECOMMENDATION GUIDANCE
━━━━━━━━━━━━━━━━━━━━
Choose:
- "option_a" or "option_b" when one option is viable under current constraints.
- "hybrid" when combining approaches materially reduces risk or failure modes.
- "defer" when missing information blocks safe execution of ALL options.
- "insufficient_information" ONLY when inputs are fundamentally unusable.

━━━━━━━━━━━━━━━━━━━━
CONFIDENCE CALIBRATION
━━━━━━━━━━━━━━━━━━━━
- Confidence reflects decision robustness, not agent agreement.
- High confidence (≥ 0.8) is rare.
- If proceeding with unresolved risks, cap confidence ≤ 0.7.
- If critic.requires_revision is true, confidence MUST NOT exceed 0.6.

━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS (ABSOLUTE)
━━━━━━━━━━━━━━━━━━━━
- Output MUST be valid JSON.
- Output MUST match SynthesizerOutput EXACTLY.
- Field order and names MUST be preserved.
- Each list item MUST be a single, atomic sentence.
- No markdown. No commentary. No extra keys.

━━━━━━━━━━━━━━━━━━━━
Output Format (STRICT)
━━━━━━━━━━━━━━━━━━━━
{{
  "agent_name": "synthesizer",

  "final_recommendation": "<option_a | option_b | hybrid | defer | insufficient_information>",
  "confidence": 0.0,

  "rationale": [
    "...",
    "..."
  ],

  "tradeoffs": [
    "...",
    "..."
  ],

  "unresolved_risks": [
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

Specialist agent outputs:
{agent_outputs}

Detector output:
{detector_output}

Critic output:
{critic_output}
"""
        ),
    ]
)


# =========================
# Runner
# =========================

def run_synthesizer_agent(state: DecisionState) -> dict:
    """
    Synthesizer agent.

    Produces the final decision by synthesizing:
    - specialist outputs,
    - detector conflicts,
    - critic issues.

    Produces SynthesizerOutput only.
    """

    # Collect specialist outputs
    specialist_outputs = {}
    for agent_name in ["systems", "ml_ai", "cost", "product"]:
        output = getattr(state.current, agent_name, None)
        if output is not None:
            specialist_outputs[agent_name] = output.model_dump()

    # Detector output
    detector_output = (
        state.current.detector.model_dump()
        if state.current.detector
        else {}
    )

    # Critic output
    critic_output = (
        state.current.critic.model_dump()
        if state.current.critic
        else {}
    )

    # Options
    options = state.input.options
    if hasattr(options, "model_dump"):
        options = options.model_dump()

    # Constraints
    constraints = state.input.constraints
    if hasattr(constraints, "model_dump"):
        constraints = constraints.model_dump()

    # Format prompt
    messages = SYNTHESIZER_AGENT_PROMPT.format_messages(
        decision_question=state.input.decision_question,
        options=options,
        constraints=constraints,
        agent_outputs=specialist_outputs,
        detector_output=detector_output,
        critic_output=critic_output,
    )

    client = OpenAI()

    response = client.responses.create(
        model=SYNTHESIZER_MODEL,
        input="\n".join(m.content for m in messages),
        reasoning={"effort": "minimal"},
    )

    response_text = response.output_text

    try:
        agent_output = SynthesizerOutput.model_validate_json(response_text)
    except ValidationError as e:
        raise RuntimeError(f"Synthesizer Agent output invalid: {e}")

    return {
        "synthesizer": agent_output
    }