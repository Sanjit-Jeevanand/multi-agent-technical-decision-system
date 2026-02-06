from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from multi_agent_decision_system.core.schema import (
    DecisionInput,
    PlannerOutput,
    SpecialistOutput,
    DetectorOutput,
    CriticOutput,
    SynthesizerOutput,
    GateOutput,
    AgentName,
)


# =========================
# Iteration Delta (Human input)
# =========================

class DecisionDelta(BaseModel):
    """
    User-provided changes between iterations.
    If this is empty, no new iteration should run.
    """
    updated_constraints: Optional[Dict] = None
    accepted_risks: List[str] = []
    rejected_recommendations: List[str] = []
    notes: Optional[str] = None


# =========================
# Iteration Snapshot
# =========================

class IterationSnapshot(BaseModel):
    iteration: int

    planner: Optional[PlannerOutput] = None
    systems: Optional[SpecialistOutput] = None
    ml_ai: Optional[SpecialistOutput] = None
    cost: Optional[SpecialistOutput] = None
    product: Optional[SpecialistOutput] = None

    detector: Optional[DetectorOutput] = None
    critic: Optional[CriticOutput] = None
    synthesizer: Optional[SynthesizerOutput] = None
    gate: Optional[GateOutput] = None

    delta_applied: Optional[DecisionDelta] = None


# =========================
# Current Iteration State
# =========================

class CurrentIteration(BaseModel):
    iteration: int

    planner: Optional[PlannerOutput] = None
    systems: Optional[SpecialistOutput] = None
    ml_ai: Optional[SpecialistOutput] = None
    cost: Optional[SpecialistOutput] = None
    product: Optional[SpecialistOutput] = None

    detector: Optional[DetectorOutput] = None
    critic: Optional[CriticOutput] = None
    synthesizer: Optional[SynthesizerOutput] = None
    gate: Optional[GateOutput] = None


# =========================
# Decision State v2
# =========================

class DecisionState(BaseModel):
    """
    Runtime state for a decision.
    This is NOT an LLM schema.
    """

    input: DecisionInput

    iteration: int = 1
    max_iterations: int = 1

    current: CurrentIteration = Field(default_factory=lambda: CurrentIteration(iteration=1))
    history: List[IterationSnapshot] = []

    approved: bool = False
    final_decision: Optional[SynthesizerOutput] = None

    def snapshot_current(self, delta: Optional[DecisionDelta] = None):
        """Freeze the current iteration into history."""
        self.history.append(
            IterationSnapshot(
                iteration=self.current.iteration,
                planner=self.current.planner,
                systems=self.current.systems,
                ml_ai=self.current.ml_ai,
                cost=self.current.cost,
                product=self.current.product,
                detector=self.current.detector,
                critic=self.current.critic,
                synthesizer=self.current.synthesizer,
                gate=self.current.gate,
                delta_applied=delta,
            )
        )

    def can_iterate(self) -> bool:
        return self.iteration < self.max_iterations

    def start_next_iteration(self, delta: DecisionDelta):
        if not self.can_iterate():
            raise RuntimeError("Max iterations reached")

        self.snapshot_current(delta)

        self.iteration += 1
        self.current = CurrentIteration(iteration=self.iteration)

        # NOTE: input mutation (constraints updates etc.)
        if delta.updated_constraints:
            self.input.constraints = self.input.constraints.copy(
                update=delta.updated_constraints
            )