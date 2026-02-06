from __future__ import annotations

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


# =========================
# Shared Enums
# =========================

Recommendation = Literal[
    "option_a",
    "option_b",
    "hybrid",
    "defer",
    "insufficient_information",
]

AgentName = Literal[
    "planner",
    "systems",
    "ml_ai",
    "cost",
    "product",
    "detector",
    "critic",
    "synthesizer",
    "gate",
]


class DecisionOptions(BaseModel):
    option_a: str
    option_b: str


class DecisionConstraints(BaseModel):
    latency_sensitivity: Optional[str] = None
    budget_sensitivity: Optional[str] = None
    team_size: Optional[str] = None
    risk_tolerance: Optional[str] = None


# New input model for full decision input
class DecisionInput(BaseModel):
    decision_question: str
    constraints: DecisionConstraints
    options: DecisionOptions

    
# =========================
# Base Output (for typing)
# =========================

class BaseAgentOutput(BaseModel):
    agent_name: AgentName


# =========================
# Specialist Agent Output
# (Systems / ML-AI / Cost / Product)
# =========================

class SpecialistOutput(BaseAgentOutput):
    recommendation: Recommendation
    confidence: float = Field(..., ge=0.0, le=1.0)

    benefits: List[str] = Field(..., min_length=2, max_length=4)
    risks: List[str] = Field(..., min_length=2, max_length=4)


# =========================
# Planner
# =========================

class PlannerSubQuestion(BaseModel):
    question: str
    why_it_matters: str
    key_unknowns: List[str]


class PlannerOutput(BaseAgentOutput):
    systems: Optional[PlannerSubQuestion] = None
    ml_ai: Optional[PlannerSubQuestion] = None
    cost: Optional[PlannerSubQuestion] = None
    product: Optional[PlannerSubQuestion] = None

    assumptions: List[str]
    clarifying_questions: List[str]


# =========================
# Detector
# =========================

class DetectedConflict(BaseModel):
    agents_involved: List[AgentName]
    description: str
    severity: Literal["low", "medium", "high"]


class DetectorOutput(BaseAgentOutput):
    conflicts: List[DetectedConflict]
    has_blocking_conflicts: bool


# =========================
# Critic
# =========================

class CriticIssue(BaseModel):
    agent: AgentName
    issue: str
    impact: Literal["low", "medium", "high"]


class CriticOutput(BaseAgentOutput):
    issues: List[CriticIssue]
    requires_revision: bool


# =========================
# Synthesizer
# =========================

class SynthesizerOutput(BaseAgentOutput):
    final_recommendation: Recommendation
    confidence: float = Field(..., ge=0.0, le=1.0)

    rationale: List[str]
    tradeoffs: List[str]

    unresolved_risks: List[str]


# =========================
# Gate
# =========================

class GateDecision(BaseModel):
    approved: bool
    blocking_reasons: List[str] = []
    required_actions: List[str] = []


class GateOutput(BaseAgentOutput):
    decision: GateDecision


# =========================
# Agent Execution Event
# (Realtime UI + WebSocket payload)
# =========================

class AgentExecutionEvent(BaseModel):
    run_id: str
    iteration: int

    agent_name: AgentName
    event_type: Literal[
        "agent_started",
        "agent_completed",
        "agent_failed",
    ]

    timestamp: float

    # Model metadata
    model: str
    temperature: Optional[float] = None

    # UI-safe outputs
    recommendation: Optional[Recommendation] = None
    confidence: Optional[float] = None
    benefits: Optional[List[str]] = None
    risks: Optional[List[str]] = None

    # Sanitized schemas
    input_schema: Dict
    output_schema: Dict

    # Cost estimation
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float