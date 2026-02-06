from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict
import asyncio
import json
from fastapi.responses import StreamingResponse

from multi_agent_decision_system.core.state import (
    create_initial_state,
    start_new_iteration,
)
from multi_agent_decision_system.agents.planner_agent import run_planner_agent
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_ai_agent import run_ml_ai_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_agent import run_product_agent


# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="Multi-Agent Technical Decision System",
    version="0.1.0",
)


# =========================
# Request / Response Models
# =========================

class DecisionRequest(BaseModel):
    decision_question: str
    constraints: Dict


class DecisionResponse(BaseModel):
    run_id: str
    iteration: int

    planner: dict
    systems: dict | None = None
    ml_ai: dict | None = None
    cost: dict | None = None
    product: dict | None = None


# =========================
# Helper async generator for streaming SSE events
# =========================

async def send_event(ws: WebSocket, event: Dict):
    await ws.send_text(json.dumps(event))


def apply_planner_patch(planner, slice_name, patch_dict):
    if not hasattr(planner, slice_name):
        raise ValueError(f"Invalid planner slice name: {slice_name}")

    existing_slice = getattr(planner, slice_name)

    # Normalize existing slice to a dict
    if existing_slice is None:
        existing_data = {}
    elif hasattr(existing_slice, "model_dump"):
        existing_data = existing_slice.model_dump()
    elif isinstance(existing_slice, dict):
        existing_data = existing_slice
    else:
        raise ValueError(f"Unsupported planner slice type: {type(existing_slice)}")

    # Apply patch
    merged_data = {**existing_data, **patch_dict}

    # Store back as plain dict to allow free UI edits
    setattr(planner, slice_name, merged_data)


# ---------------------------
# WebSocket endpoint
# ---------------------------

@app.websocket("/ws/decision")
async def decision_ws(ws: WebSocket):
    await ws.accept()

    try:
        # ---- Receive start message ----
        msg = await ws.receive_json()
        if msg.get("type") != "start":
            await send_event(ws, {
                "type": "error",
                "message": "First message must be type=start"
            })
            return

        decision_question = msg["decision_question"]
        constraints = msg.get("constraints", {})

        # ---- Init state ----
        state = create_initial_state(
            decision_question=decision_question,
            constraints=constraints,
            max_iterations=1,
        )
        state = start_new_iteration(state)

        await send_event(ws, {
            "type": "run_started",
            "run_id": state.metadata.run_id,
            "iteration": state.metadata.current_iteration
        })

        # ---- Planner ----
        planner = run_planner_agent(state)["planner"]
        state.current.planner = planner

        await send_event(ws, {
            "type": "planner_ready",
            "agent": "planner",
            "payload": planner.model_dump()
        })

        # ---- Wait for planner edits and approval ----
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "edit_planner_slice":
                slice_name = msg.get("slice_name")
                patch = msg.get("patch", {})
                try:
                    apply_planner_patch(state.current.planner, slice_name, patch)
                    await send_event(ws, {
                        "type": "planner_slice_updated",
                        "slice_name": slice_name,
                        "payload": getattr(state.current.planner, slice_name) if getattr(state.current.planner, slice_name) else None
                    })
                except Exception as e:
                    await send_event(ws, {
                        "type": "error",
                        "message": f"Failed to edit planner slice '{slice_name}': {str(e)}"
                    })

            elif msg_type == "delete_planner_slice":
                slice_name = msg.get("slice_name")
                if not hasattr(state.current.planner, slice_name):
                    await send_event(ws, {
                        "type": "error",
                        "message": f"Invalid planner slice name: {slice_name}"
                    })
                else:
                    setattr(state.current.planner, slice_name, None)
                    await send_event(ws, {
                        "type": "planner_slice_deleted",
                        "slice_name": slice_name,
                        "payload": None
                    })

            elif msg_type == "approve_planner":
                break

            else:
                await send_event(ws, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

        # ---- Specialists (parallel) ----
        async def run_agent(name, fn):
            try:
                result = fn(state)[name]
                await send_event(ws, {
                    "type": "agent_event",
                    "agent": name,
                    "status": "completed",
                    "payload": result.model_dump()
                })
            except Exception as e:
                await send_event(ws, {
                    "type": "agent_event",
                    "agent": name,
                    "status": "failed",
                    "error": str(e)
                })

        await asyncio.gather(
            run_agent("systems", run_systems_agent),
            run_agent("ml_ai", run_ml_ai_agent),
            run_agent("cost", run_cost_agent),
            run_agent("product", run_product_agent),
        )

        await send_event(ws, {
            "type": "run_completed",
            "run_id": state.metadata.run_id
        })

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        await send_event(ws, {
            "type": "error",
            "message": str(e)
        })

async def decision_event_stream(state):
    # Run planner first
    planner_result = run_planner_agent(state)
    planner = planner_result["planner"]
    state.current.planner = planner
    yield f"data: {json.dumps({'stage': 'planner', 'planner': planner.model_dump()})}\n\n"

    # Run specialists in parallel
    async def run_systems():
        return run_systems_agent(state).get("systems")

    async def run_ml_ai():
        return run_ml_ai_agent(state).get("ml_ai")

    async def run_cost():
        return run_cost_agent(state).get("cost")

    async def run_product():
        return run_product_agent(state).get("product")

    # Wrap synchronous calls in asyncio.to_thread for concurrency
    tasks = [
        asyncio.to_thread(run_systems),
        asyncio.to_thread(run_ml_ai),
        asyncio.to_thread(run_cost),
        asyncio.to_thread(run_product),
    ]

    # Map agent name to task index for identification
    agent_names = ["systems", "ml_ai", "cost", "product"]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        idx = tasks.index(coro)
        name = agent_names[idx]
        yield f"data: {json.dumps({ 'stage': name, name: result.model_dump() if result else None })}\n\n"


# =========================
# Routes
# =========================

@app.post("/decision/run", response_model=DecisionResponse)
def run_decision(request: DecisionRequest):
    """
    Executes:
    - Planner
    - Systems
    - ML/AI
    - Cost
    - Product

    Single iteration, synchronous.
    """

    try:
        # --- Initialize state ---
        state = create_initial_state(
            decision_question=request.decision_question,
            constraints=request.constraints,
            max_iterations=1,
        )
        state = start_new_iteration(state)

        # --- Planner ---
        planner_result = run_planner_agent(state)
        planner = planner_result["planner"]
        state.current.planner = planner

        # --- Specialists ---
        systems = run_systems_agent(state).get("systems")
        ml_ai = run_ml_ai_agent(state).get("ml_ai")
        cost = run_cost_agent(state).get("cost")
        product = run_product_agent(state).get("product")

        # --- Build response ---
        return DecisionResponse(
            run_id=state.metadata.run_id,
            iteration=state.metadata.current_iteration,
            planner=planner.model_dump(),
            systems=systems.model_dump() if systems else None,
            ml_ai=ml_ai.model_dump() if ml_ai else None,
            cost=cost.model_dump() if cost else None,
            product=product.model_dump() if product else None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/decision/stream")
async def run_decision_stream(request: DecisionRequest):
    try:
        # --- Initialize state ---
        state = create_initial_state(
            decision_question=request.decision_question,
            constraints=request.constraints,
            max_iterations=1,
        )
        state = start_new_iteration(state)

        return StreamingResponse(decision_event_stream(state), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/decision/planner-only")
async def planner_only(request: DecisionRequest):
    try:
        # --- Initialize state ---
        state = create_initial_state(
            decision_question=request.decision_question,
            constraints=request.constraints,
            max_iterations=1,
        )
        state = start_new_iteration(state)

        # --- Planner ---
        planner_result = run_planner_agent(state)
        planner = planner_result["planner"]
        state.current.planner = planner

        return {
            "run_id": state.metadata.run_id,
            "planner": planner.model_dump(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# =========================
# Health check
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}