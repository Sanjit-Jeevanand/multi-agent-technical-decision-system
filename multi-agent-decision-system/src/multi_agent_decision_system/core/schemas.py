from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


# -----------------------------
# INPUT SCHEMAS
# -----------------------------

class DecisionConstraints(BaseModel):
    budget_sensitivity: Optional[Literal["low", "medium", "high"]] = None
    latency_sensitivity: Optional[Literal["low", "medium", "high"]] = None
    team_size: Optional[Literal["small", "medium", "large"]] = None
    risk_tolerance: Optional[Literal["low", "medium", "high"]] = None


class DecisionInput(BaseModel):
    decision_question: str
    decision_type: Optional[str] = None
    decision_template: Optional[Literal[
        "architecture",
        "build_vs_buy",
        "operational_mode",
        "tooling"
    ]] = None
    constraints: Optional[DecisionConstraints] = None


# -----------------------------
# INPUT LOGGING (AUDIT / UI)
# -----------------------------

class IterationInputLog(BaseModel):

    # Input provided to the planner agent
    planner_input: Optional[Dict] = None

    # Inputs provided to each specialist agent (keyed by agent name)
    agent_inputs: Dict[str, Dict] = Field(default_factory=dict)

    # Input provided to the critic agent
    critic_input: Optional[Dict] = None

    # Input provided to the synthesizer agent
    synthesizer_input: Optional[Dict] = None


class InputLog(BaseModel):

    # Raw user-provided input at run start (immutable)
    initial_input: Dict

    # Agent-scoped inputs logged directly by each agent for audit and UI inspection.
    agent_inputs: Dict[str, Dict] = Field(default_factory=dict)

    # Inputs recorded per iteration, keyed by iteration index
    iterations: Dict[int, IterationInputLog] = Field(default_factory=dict)


# -----------------------------
# OUTPUT LOGGING (AUDIT / UI)
# -----------------------------

class OutputLogEntry(BaseModel):
    stage: str
    payload: Dict


class OutputLog(BaseModel):

    # Planner output (single, immutable)
    planner_output: Optional[Dict] = None

    # Outputs from specialist agents, keyed by agent name,
    # each value is a list to support multiple iterations
    agent_outputs: Dict[str, list[Dict]] = Field(default_factory=dict)

    # Disagreement detector output per iteration
    disagreement_outputs: list[Dict] = Field(default_factory=list)

    # Critic agent outputs per iteration
    critic_outputs: list[Dict] = Field(default_factory=list)

    # Synthesizer outputs per iteration
    synthesizer_outputs: list[Dict] = Field(default_factory=list)

    # Confidence and termination outputs per iteration
    termination_outputs: list[Dict] = Field(default_factory=list)


# -----------------------------
# PLANNER OUTPUT
# -----------------------------

class PlannerDimension(BaseModel):
    question: str
    why_it_matters: str
    key_unknowns: List[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):

    # Exactly one dimension per agent (optional if not relevant)
    systems: Optional[PlannerDimension] = None
    ml: Optional[PlannerDimension] = None
    cost: Optional[PlannerDimension] = None
    product_risk: Optional[PlannerDimension] = None

    # Meta framing
    assumptions: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)

    # Cost awareness (reported, not optimized)
    model_used: str
    estimated_tokens_in: int
    estimated_tokens_out: int


# -----------------------------
# AGENT OUTPUTS
# -----------------------------

# Shared output contract for all specialist agents (Phase 3).
RecommendationLabel = Literal[
    "option_a",
    "option_b",
    "hybrid",
    "defer",
    "insufficient_information"
]

class AgentOutput(BaseModel):
    agent_name: str
    recommendation: RecommendationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    benefits: List[str]
    risks: List[str]


# -----------------------------
# DISAGREEMENTS
# -----------------------------

class Disagreement(BaseModel):
    decision_dimension: str
    options: List[str]
    agents_in_favor: Dict[str, List[str]]
    summary: Optional[str] = None


# -----------------------------
# CRITIC FEEDBACK
# -----------------------------

class CriticFeedback(BaseModel):
    challenged_assumptions: List[str]
    missing_considerations: List[str]
    overconfidence_flags: List[str]


# -----------------------------
# SYNTHESIS OUTPUT
# -----------------------------

class RejectedAlternative(BaseModel):
    option: str
    reason: str


class SynthesisOutput(BaseModel):
    final_decision: str
    rationale: List[str]
    rejected_alternatives: List[RejectedAlternative]
    accepted_risks: List[str]  # Accepted risks are referenced by description only; severity is resolved earlier by the critic.


# -----------------------------
# CONFIDENCE & TERMINATION
# -----------------------------

class ConfidenceMetrics(BaseModel):
    agent_confidences: Dict[str, float]
    aggregate_confidence: float = Field(ge=0.0, le=1.0)


class TerminationState(BaseModel):
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    high_severity_risks_remaining: int
    iteration_count: int
    max_iterations: int = 3
    can_terminate: bool


# -----------------------------
# HUMAN FEEDBACK (OPTIONAL)
# -----------------------------

class HumanFeedback(BaseModel):
    decision: Literal["approve", "reject", "revise"]
    comments: Optional[str] = None


# -----------------------------
# METADATA
# -----------------------------

class ExecutionMetadata(BaseModel):
    run_id: str
    timestamps: Dict[str, str]
    model_versions: Dict[str, str]