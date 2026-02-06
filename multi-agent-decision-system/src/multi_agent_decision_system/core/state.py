# src/multi_agent_decision_system/core/state.py

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import uuid4
import time

from multi_agent_decision_system.core.schema import (
    AgentName,
    DecisionConstraints,
    PlannerOutput,
    SpecialistOutput,
    DetectorOutput,
    CriticOutput,
    SynthesizerOutput,
    GateOutput,
)


# =========================
# Decision Input
# =========================

class DecisionInput(BaseModel):
    decision_question: str
    constraints: DecisionConstraints


# =========================
# Run Metadata
# =========================

class RunMetadata(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: float = Field(default_factory=lambda: time.time())

    current_iteration: int = 0
    max_iterations: int = 3

    active_agent: Optional[AgentName] = None
    terminated: bool = False
    termination_reason: Optional[str] = None


# =========================
# Iteration State
# =========================

class IterationState(BaseModel):
    iteration: int

    # Planner
    planner: Optional[PlannerOutput] = None

    # Specialist agents
    systems: Optional[SpecialistOutput] = None
    ml_ai: Optional[SpecialistOutput] = None
    cost: Optional[SpecialistOutput] = None
    product: Optional[SpecialistOutput] = None

    # Reasoning agents
    detector: Optional[DetectorOutput] = None
    critic: Optional[CriticOutput] = None
    synthesizer: Optional[SynthesizerOutput] = None
    gate: Optional[GateOutput] = None


# =========================
# Global Decision State
# =========================

class DecisionState(BaseModel):
    """
    Single LangGraph state object.
    Passed between every node.
    """

    input: DecisionInput
    metadata: RunMetadata

    iterations: List[IterationState] = Field(default_factory=list)

    # Always points to iterations[-1]
    current: Optional[IterationState] = None

    # Terminal outputs
    final_decision: Optional[SynthesizerOutput] = None
    approved: bool = False


# =========================
# State Helpers
# =========================

def create_initial_state(
    decision_question: str,
    constraints: DecisionConstraints | Dict,
    max_iterations: int = 3,
) -> DecisionState:
    """
    Canonical state constructor.
    Guarantees constraints are always a Pydantic model.
    """

    normalized_constraints = (
        constraints
        if isinstance(constraints, DecisionConstraints)
        else DecisionConstraints(**constraints)
    )

    metadata = RunMetadata(max_iterations=max_iterations)

    return DecisionState(
        input=DecisionInput(
            decision_question=decision_question,
            constraints=normalized_constraints,
        ),
        metadata=metadata,
        iterations=[],
        current=None,
    )


def start_new_iteration(state: DecisionState) -> DecisionState:
    """
    Starts a new iteration and sets it as current.
    """

    iteration = IterationState(
        iteration=state.metadata.current_iteration
    )

    state.iterations.append(iteration)
    state.current = iteration

    return state


def advance_iteration(state: DecisionState) -> DecisionState:
    """
    Advances iteration counter and enforces termination.
    """

    state.metadata.current_iteration += 1

    if state.metadata.current_iteration >= state.metadata.max_iterations:
        state.metadata.terminated = True
        state.metadata.termination_reason = "max_iterations_reached"

    return state