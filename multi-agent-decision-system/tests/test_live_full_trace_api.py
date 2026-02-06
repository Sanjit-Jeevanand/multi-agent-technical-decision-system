import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from multi_agent_decision_system.api import router

# =============================================================================
# Test App Fixture
# =============================================================================

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

# =============================================================================
# Shared Constants
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

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

# =============================================================================
# Iteration 1 — Exploration
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

    assert data["iteration"] == 1
    assert data["gate_tier"] == "exploration"
    assert data["approved"] is False

    assert "agents" in data
    assert len(data["agents"]) >= 7  # planner + 4 specialists + detector + critic + synthesizer

    agent_names = {a["agent"] for a in data["agents"]}
    assert "planner" in agent_names
    assert "synthesizer" in agent_names
    assert "gate" not in agent_names  # gate is top-level

    assert data["total_tokens"] > 0
    assert data["total_cost_usd"] > 0

# =============================================================================
# Iteration 2 — Commitment with User Delta
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

    assert data["iteration"] == 2
    assert data["gate_tier"] == "commitment"
    assert "delta" in data
    assert data["delta"]["accepted_risks"]

    # Specialists should NOT rerun in iteration 2
    agent_names = {a["agent"] for a in data["agents"]}
    assert "systems" in agent_names  # from iteration 1
    assert "ml_ai" in agent_names
    assert "product" in agent_names

    # Core agents rerun
    assert "detector" in agent_names
    assert "critic" in agent_names
    assert "synthesizer" in agent_names

    assert data["total_tokens"] > 0
    assert data["total_cost_usd"] > 0

# =============================================================================
# Iteration 2 — Force Approve Override
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

    assert data["gate_tier"] == "override"
    assert data["approved"] is True