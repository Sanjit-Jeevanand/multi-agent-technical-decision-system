import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from multi_agent_decision_system.main import router

# =============================================================================
# Test App
# =============================================================================

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# =============================================================================
# Constants
# =============================================================================

DECISION_QUESTION = "Should we use batch or online inference?"

OPTIONS = {
    "option_a": "batch inference",
    "option_b": "online inference",
}

CONSTRAINTS = {
    "latency_sensitivity": "medium",
    "team_size": "small",
    "risk_tolerance": "medium",
}


# =============================================================================
# Health Check
# =============================================================================

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# =============================================================================
# Iteration 1 — Full Trace (Exploration)
# =============================================================================

def test_full_trace_iteration_1(client):
    payload = {
        "decision_question": DECISION_QUESTION,
        "options": OPTIONS,
        "constraints": CONSTRAINTS,
        "iteration": 1,
    }

    resp = client.post("/decision/full-trace", json=payload)
    assert resp.status_code == 200

    data = resp.json()

    # ------------------
    # Basic structure
    # ------------------
    assert data["iteration"] == 1
    assert data["gate_tier"] == "exploration"
    assert "agents" in data
    assert isinstance(data["agents"], list)

    # ------------------
    # Required agents
    # ------------------
    agent_names = {a["agent"] for a in data["agents"]}

    for required in [
        "planner",
        "systems",
        "ml_ai",
        "cost",
        "product",
        "detector",
        "critic",
        "synthesizer",
    ]:
        assert required in agent_names

    # ------------------
    # Per-agent cost visibility
    # ------------------
    for agent in data["agents"]:
        assert agent["input_tokens"] > 0
        assert agent["cost_usd"] >= 0

    # ------------------
    # Gate output exists
    # ------------------
    assert "gate" in data
    assert "approved" in data
    assert data["approved"] in (True, False)

    # ------------------
    # Cost summary
    # ------------------
    assert data["total_tokens"] > 0
    assert data["total_cost_usd"] > 0


# =============================================================================
# Iteration 2 — Commitment (User Input, Specialists Frozen)
# =============================================================================

def test_full_trace_iteration_2_commitment(client):
    payload = {
        "decision_question": DECISION_QUESTION,
        "options": OPTIONS,
        "constraints": CONSTRAINTS,
        "iteration": 2,
        "accepted_risks": [
            "Model performance may degrade between batch runs due to drift."
        ],
        "rejected_recommendations": ["hybrid"],
        "notes": "We accept staleness risk for v1 and prioritize operational simplicity.",
    }

    resp = client.post("/decision/full-trace", json=payload)
    assert resp.status_code == 200

    data = resp.json()

    # ------------------
    # Authority tier
    # ------------------
    assert data["gate_tier"] == "commitment"

    # ------------------
    # Specialists must NOT rerun
    # ------------------
    agent_names = {a["agent"] for a in data["agents"]}

    for forbidden in [
        "planner",
        "systems",
        "ml_ai",
        "cost",
        "product",
    ]:
        assert forbidden not in agent_names

    # ------------------
    # Core agents still run
    # ------------------
    for required in ["detector", "critic", "synthesizer"]:
        assert required in agent_names

    # ------------------
    # Decision must exist
    # ------------------
    assert data["final_recommendation"] is not None
    assert data["approved"] is True

    # ------------------
    # Cost sanity check
    # (Iteration 2 should be much cheaper)
    # ------------------
    assert data["total_cost_usd"] < 0.02


# =============================================================================
# Force Approve — Override Tier
# =============================================================================

def test_force_approve_override(client):
    payload = {
        "decision_question": DECISION_QUESTION,
        "options": OPTIONS,
        "constraints": CONSTRAINTS,
        "iteration": 2,
        "force_approve": True,
    }

    resp = client.post("/decision/full-trace", json=payload)
    assert resp.status_code == 200

    data = resp.json()

    assert data["approved"] is True