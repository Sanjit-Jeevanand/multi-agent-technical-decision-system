# Multi-Agent Technical Decision System

**Live Demo (UI):**  
https://storage.googleapis.com/decision-system-site/index.html  

**Backend (Cloud Run):**  
https://decision-system-521270728838.europe-west1.run.app  

**Repository:**  
https://github.com/Sanjit-Jeevanand/multi-agent-technical-decision-system

---

## Overview

The **Multi-Agent Technical Decision System** is a real-time, interactive system that models how senior engineering teams make complex technical decisions under uncertainty.

Instead of relying on a single LLM response, the system forces:
- Independent reasoning
- Explicit disagreement
- Structured critique
- Controlled synthesis
- Policy-based validation

Every agent runs with a clearly defined role, input schema, and output schema, producing a **traceable and auditable decision process**.

---

## Why This Exists

Most LLM systems fail silently by:
- Collapsing multiple perspectives into one answer
- Hiding uncertainty
- Over-optimizing for fluency instead of correctness

This system is designed to:
- Surface disagreements instead of smoothing them over
- Make risks explicit
- Require justification before approval
- Support human-in-the-loop iteration

---

## Architecture

## High-Level Flow

User submits decision question
        ↓
Planner Agent
(decomposes problem + creates specialist prompts)
        ↓
Parallel Specialist Agents (run concurrently)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Systems    │   ML / AI   │    Cost     │   Product   │
└─────────────┴─────────────┴─────────────┴─────────────┘
        ↓
Disagreement Detector
(identifies conflicts and blocking risks)
        ↓
Critic Agent
(challenges assumptions and surfaces hidden issues)
        ↓
Synthesizer Agent
(produces final recommendation + confidence)
        ↓
Confidence Gate
(policy-based approval / rejection / iteration)
        ↓
Decision Approved OR Iterate with Human Feedback

Execution Characteristics:
- Planner runs first (strict dependency)
- Specialist agents run in true parallelism
- Detector → Critic → Synthesizer run sequentially
- Gate enforces approval rules
- Full trace streamed live via WebSocket



---

## Agents

### Planner Agent
**Purpose:**  
Decomposes the decision into structured sub-questions.

**Why it matters:**  
Ensures all specialists reason over the same problem definition.

---

### Specialist Agents (Parallel Execution)

Each specialist reasons independently and **cannot see other specialist outputs**.

| Agent     | Responsibility |
|----------|----------------|
| Systems  | Infrastructure, scalability, reliability |
| ML/AI   | Model feasibility, latency, training complexity |
| Cost    | Infrastructure + inference cost tradeoffs |
| Product | UX, market fit, delivery risk |

---

### Disagreement Detector
**Purpose:**  
Identifies conflicts between specialist recommendations.

**Output:**
- Conflicting agents
- Severity (low / medium / high)
- Blocking vs non-blocking conflicts

---

### Critic Agent
**Purpose:**  
Challenges assumptions and surfaces hidden risks.

**Key Feature:**  
Produces *actionable issues* instead of generic criticism.

---

### Synthesizer Agent
**Purpose:**  
Integrates all perspectives into a final recommendation.

**Outputs:**
- Final recommendation
- Confidence score
- Rationale
- Tradeoffs
- Unresolved risks

---

### Confidence Gate
**Purpose:**  
Enforces approval policy.

**Gate Tiers:**
- `exploration`
- `commitment`
- `override` (manual)

The gate can:
- Approve
- Reject
- Require iteration

---

## LLM Choice

### Model Used
- **GPT-5-mini** for specialists, planner, critic, detector
- **GPT-5.1** for synthesis

### Why GPT-5-mini
- Strong structured reasoning
- Lower cost for parallel execution
- Fast enough for real-time UI

### Why GPT-5.1 for synthesis
- Better global reasoning
- Handles tradeoffs and conflict resolution more reliably

---

## Cost

### Typical Cost Per Iteration
- **~$0.02 USD per iteration**

This includes:
- Planner
- 4 parallel specialists
- Detector
- Critic
- Synthesizer
- Gate

Token usage and cost are tracked **per agent** and streamed live to the UI.

---

## Real-Time Execution

- WebSockets (`/ws/full-trace`)
- Agents emit:
  - `agent_started`
  - `agent_completed`
  - `iteration_complete`
  - `run_complete`
- UI renders agent states independently (no batching)

Parallel agents run using a `ThreadPoolExecutor` to avoid blocking the event loop.

---

## Frontend

- **Pure static HTML**
- Hosted on **Google Cloud Storage**
- No framework, no build step
- Uses WebSockets for real-time updates
- Tailwind CSS for styling

---

## Backend

- **FastAPI**
- Hosted on **Google Cloud Run**
- Dockerized
- WebSocket support
- Stateless execution

---

## Deployment

### Frontend (Static)
- Google Cloud Storage
- Public bucket
- Zero cold start
- Zero compute cost

### Backend (API)
- Google Cloud Run
- Region: `europe-west1`
- Autoscaling enabled
- Cold start optimized where possible

---

## Folder Structure

multi-agent-decision-system/
├── index.html
│   # Static frontend UI (GCS hosted)
│   # WebSocket-driven real-time execution view
│
├── Dockerfile
│   # Container image for Cloud Run
│
├── requirements.txt
│   # Python dependencies
│
├── pyproject.toml
│   # Package + tooling config
│
├── src/
│   └── multi_agent_decision_system/
│       ├── api.py
│       │   # FastAPI app
│       │   # REST + WebSocket endpoints
│       │
│       ├── agents/
│       │   ├── planner_agent.py
│       │   ├── systems_agent.py
│       │   ├── ml_ai_agent.py
│       │   ├── cost_agent.py
│       │   ├── product_agent.py
│       │   ├── detector_agent.py
│       │   ├── critic_agent.py
│       │   ├── synthesizer_agent.py
│       │   └── gate_agent.py
│       │
│       ├── core/
│       │   ├── state.py
│       │   ├── state_v2.py
│       │   ├── schema.py
│       │   ├── events.py
│       │   └── orchestrator.py
│       │
│       ├── graph/
│       │   ├── planner_graph.py
│       │   ├── specialist_graph.py
│       │   └── decision_graph.py
│       │
│       ├── logic/
│       │   ├── disagreement_detector.py
│       │   ├── confidence_gate.py
│       │   └── termination.py
│       │
│       └── utils/
│           ├── logging.py
│           └── ids.py
│
├── tests/
│   # Unit + integration tests
│
└── docs/
    # Design notes and architectural explanations

---

## Key Design Decisions

- **No agent sees another agent’s output unless explicitly allowed**
- **Disagreement is a first-class signal**
- **Human feedback is a required step for iteration**
- **UI reflects real execution order, not simulated timing**
- **Costs are visible and attributable**

---

## What This Is Not

- ❌ Not a prompt chain
- ❌ Not a chat UI
- ❌ Not a monolithic LLM call
- ❌ Not optimized for “niceness”

This system is optimized for **correctness, traceability, and decision quality**.

---

## License

MIT License

---

## Author

**Sanjit Jeevanand**  
CS Master’s Student @ UCL  
Focus: ML systems, agent architectures, decision intelligence

---