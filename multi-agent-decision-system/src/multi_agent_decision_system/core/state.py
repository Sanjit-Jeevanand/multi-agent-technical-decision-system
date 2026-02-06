from typing import Dict, Optional, Annotated
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime

from multi_agent_decision_system.core.schemas import (
    DecisionInput,
    PlannerOutput,
    AgentOutput,
    Disagreement,
    CriticFeedback,
    SynthesisOutput,
    ConfidenceMetrics,
    TerminationState,
    HumanFeedback,
    ExecutionMetadata,
)


# =====================================================
# GLOBAL DECISION STATE
# =====================================================

class DecisionState(BaseModel):

    # -----------------
    # IMMUTABLE INPUT
    # -----------------
    input: Annotated[DecisionInput, "immutable"]

    # -----------------
    # PLANNER OUTPUT
    # -----------------
    plan: Optional[PlannerOutput] = None

    # -----------------
    # SPECIALIST AGENT OUTPUTS
    # (Overwritten per iteration)
    # -----------------
    agent_outputs: Dict[str, AgentOutput] = Field(default_factory=dict)

    # -----------------
    # DISAGREEMENTS
    # -----------------
    disagreements: Optional[list[Disagreement]] = None

    # -----------------
    # CRITIC FEEDBACK
    # -----------------
    critic_feedback: Optional[CriticFeedback] = None

    # -----------------
    # SYNTHESIS RESULT
    # -----------------
    synthesis: Optional[SynthesisOutput] = None

    # -----------------
    # CONFIDENCE METRICS
    # -----------------
    confidence: Optional[ConfidenceMetrics] = None

    # -----------------
    # TERMINATION STATE
    # -----------------
    termination: TerminationState

    # -----------------
    # OPTIONAL HUMAN FEEDBACK
    # -----------------
    human_feedback: Optional[HumanFeedback] = None

    # -----------------
    # EXECUTION METADATA
    # -----------------
    metadata: ExecutionMetadata


# =====================================================
# STATE INITIALIZATION
# =====================================================

def create_initial_state(
    decision_input: DecisionInput,
    confidence_threshold: float = 0.75,
    max_iterations: int = 3,
) -> DecisionState:


    run_id = str(uuid4())
    timestamp = datetime.utcnow().isoformat()

    termination_state = TerminationState(
        confidence_threshold=confidence_threshold,
        high_severity_risks_remaining=0,
        iteration_count=0,
        max_iterations=max_iterations,
        can_terminate=False,
    )

    metadata = ExecutionMetadata(
        run_id=run_id,
        timestamps={"created_at": timestamp},
        model_versions={},
    )

    return DecisionState(
        input=decision_input,
        plan=None,
        agent_outputs={},
        disagreements=None,
        critic_feedback=None,
        synthesis=None,
        confidence=None,
        termination=termination_state,
        human_feedback=None,
        metadata=metadata,
    )


# =====================================================
# ITERATION HELPERS
# =====================================================

def increment_iteration(state: DecisionState) -> DecisionState:
    state.termination.iteration_count += 1
    return state


def record_timestamp(state: DecisionState, key: str) -> DecisionState:
    state.metadata.timestamps[key] = datetime.utcnow().isoformat()
    return state