from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class GatePolicyTier(str, Enum):
    EXPLORATION = "exploration"   # Iteration 1, pre-user intent
    COMMITMENT = "commitment"     # After user input
    OVERRIDE = "override"         # Force approve

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
    input: DecisionInput
    iteration: int = 0

    current: CurrentIteration
    history: list[IterationSnapshot] = []

    # 🔐 NEW: explicit authority tier
    gate_tier: GatePolicyTier = GatePolicyTier.EXPLORATION

    # Optional but useful
    force_approve: bool = False

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

    def start_next_iteration(self, delta: DecisionDelta):
        self.snapshot_current(delta)

        self.iteration += 1
        self.current = CurrentIteration(iteration=self.iteration)

        # Shift authority after first user input
        self.gate_tier = GatePolicyTier.COMMITMENT

        # NOTE: input mutation (constraints updates etc.)
        if delta.updated_constraints:
            self.input.constraints = self.input.constraints.copy(
                update=delta.updated_constraints
            )