from typing import Dict, Optional
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
    InputLog,
    OutputLog,
    IterationInputLog,
)


# =====================================================
# GLOBAL DECISION STATE
# =====================================================

class DecisionState(BaseModel):

    # -----------------
    # IMMUTABLE INPUT
    # -----------------
    input: DecisionInput

    # -----------------
    # INPUT LOGGING
    # -----------------
    input_log: InputLog

    # -----------------
    # OUTPUT LOGGING
    # -----------------
    output_log: OutputLog

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

    input_log = InputLog(
        initial_input=decision_input.dict()
    )

    output_log = OutputLog()

    return DecisionState(
        input=decision_input,
        input_log=input_log,
        output_log=output_log,
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
    initialize_iteration_log(state)
    return state


def record_timestamp(state: DecisionState, key: str) -> DecisionState:
    state.metadata.timestamps[key] = datetime.utcnow().isoformat()
    return state


def initialize_iteration_log(state: DecisionState) -> DecisionState:
    i = state.termination.iteration_count
    if i not in state.input_log.iterations:
        state.input_log.iterations[i] = IterationInputLog()
    return state