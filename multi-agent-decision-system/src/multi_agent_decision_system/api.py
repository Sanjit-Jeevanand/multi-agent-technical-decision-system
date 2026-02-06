from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Awaitable, Callable
import tiktoken
import asyncio

from multi_agent_decision_system.core.schema import DecisionInput
from multi_agent_decision_system.core.state_v2 import (
    DecisionState,
    CurrentIteration,
    DecisionDelta,
    GatePolicyTier,
)

from multi_agent_decision_system.agents.planner_agent import run_planner_agent
from multi_agent_decision_system.agents.systems_agent import run_systems_agent
from multi_agent_decision_system.agents.ml_ai_agent import run_ml_ai_agent
from multi_agent_decision_system.agents.cost_agent import run_cost_agent
from multi_agent_decision_system.agents.product_agent import run_product_agent
from multi_agent_decision_system.agents.detector_agent import run_detector_agent
from multi_agent_decision_system.agents.critic_agent import run_critic_agent
from multi_agent_decision_system.agents.synthesizer_agent import run_synthesizer_agent
from multi_agent_decision_system.agents.gate_agent import run_gate

router = APIRouter()

# =========================
# Token/Cost Estimation
# =========================

encoder = tiktoken.encoding_for_model("gpt-4")
GPT5_INPUT_PER_M = 1.25
GPT5_OUTPUT_PER_M = 10.0


def estimate(text: str, *, is_output: bool):
    tokens = len(encoder.encode(text)) / 2
    rate = GPT5_OUTPUT_PER_M if is_output else GPT5_INPUT_PER_M
    cost = (tokens / 1_000_000) * rate
    return tokens, round(cost, 6)


# =========================
# Request/Response Models
# =========================

class FullDecisionRequest(BaseModel):
    decision_question: str
    options: Dict[str, str]
    constraints: Dict[str, Optional[str]]
    iteration: int = 1
    accepted_risks: List[str] = []
    rejected_recommendations: List[str] = []
    notes: Optional[str] = None
    force_approve: bool = False


class AgentTrace(BaseModel):
    agent: str
    input_context: Dict
    output: Optional[Dict]
    input_tokens: int
    output_tokens: int
    cost_usd: float


class FullDecisionResponse(BaseModel):
    run_id: str
    iteration: int
    gate_tier: str
    input: Dict
    delta: Optional[Dict]
    agents: List[AgentTrace]
    gate: Dict
    approved: bool
    final_recommendation: Optional[str]
    total_tokens: int
    total_cost_usd: float


# =========================
# Helper Functions
# =========================

def agent_event(
    *,
    agent: str,
    state: DecisionState,
    output_model=None,
):
    input_context = {
        "decision_input": state.input.model_dump(),
        "current_iteration": state.current.model_dump(),
        "gate_tier": state.gate_tier.value,
    }

    input_text = str(input_context)
    in_tokens, in_cost = estimate(input_text, is_output=False)

    if output_model:
        out_text = output_model.model_dump_json()
        out_tokens, out_cost = estimate(out_text, is_output=True)
        output = output_model.model_dump()
    else:
        out_tokens = out_cost = 0
        output = None

    return {
        "agent": agent,
        "input_context": input_context,
        "output": output,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_usd": round(in_cost + out_cost, 6),
    }


def run_and_trace(agent_name: str, fn, state: DecisionState) -> AgentTrace:
    input_context = {
        "input": state.input.model_dump(),
        "current": state.current.model_dump(),
    }

    input_text = str(input_context)
    in_tokens, in_cost = estimate(input_text, is_output=False)

    output = fn(state).get(agent_name)
    if output:
        out_text = output.model_dump_json()
        out_tokens, out_cost = estimate(out_text, is_output=True)
        output_data = output.model_dump()
        setattr(state.current, agent_name, output)
    else:
        out_tokens = out_cost = 0
        output_data = None

    return AgentTrace(
        agent=agent_name,
        input_context=input_context,
        output=output_data,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        cost_usd=round(in_cost + out_cost, 6),
    )


# =========================
# New Async Helper for WebSocket Endpoint
# =========================

async def run_agent_with_realtime_trace(
    agent_name: str,
    runner: Callable[[DecisionState], Dict[str, BaseModel]],
    state: DecisionState,
    emit: Optional[Callable[[Dict], Awaitable[None]]] = None,
) -> Dict:
    input_context = {
        "input": state.input.model_dump(),
        "current": state.current.model_dump(),
    }
    input_text = str(input_context)
    in_tokens, in_cost = estimate(input_text, is_output=False)

    result = runner(state)
    output_model = result.get(agent_name)

    if output_model:
        out_text = output_model.model_dump_json()
        out_tokens, out_cost = estimate(out_text, is_output=True)
        output_data = output_model.model_dump()
        setattr(state.current, agent_name, output_model)
    else:
        out_tokens = out_cost = 0
        output_data = None

    event = {
        "type": "agent_completed",
        "agent": agent_name,
        "input_context": input_context,
        "output": output_data,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_usd": round(in_cost + out_cost, 6),
    }

    if emit:
        await emit(event)

    return event


# =========================
# Endpoints
# =========================

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/decision/full-trace", response_model=FullDecisionResponse)
def run_full_decision_trace(req: FullDecisionRequest):
    try:
        # Initialize state
        state = DecisionState(
            input=DecisionInput(
                decision_question=req.decision_question,
                options=req.options,
                constraints=req.constraints,
            ),
            iteration=1,
            current=CurrentIteration(iteration=1),
        )

        agent_traces: List[AgentTrace] = []

        # ITERATION 1 - Run all agents
        for name, fn in [
            ("planner", run_planner_agent),
            ("systems", run_systems_agent),
            ("ml_ai", run_ml_ai_agent),
            ("cost", run_cost_agent),
            ("product", run_product_agent),
        ]:
            trace = run_and_trace(name, fn, state)
            agent_traces.append(trace)

        # Core agents for iteration 1
        for name, fn in [
            ("detector", run_detector_agent),
            ("critic", run_critic_agent),
            ("synthesizer", run_synthesizer_agent),
        ]:
            trace = run_and_trace(name, fn, state)
            agent_traces.append(trace)

        # ITERATION 2+ if requested
        if req.iteration >= 2:
            delta = DecisionDelta(
                accepted_risks=req.accepted_risks,
                rejected_recommendations=req.rejected_recommendations,
                notes=req.notes,
            )
            state.start_next_iteration(delta)

            # In iteration 2+, only core agents rerun (specialists are frozen)
            for name, fn in [
                ("detector", run_detector_agent),
                ("critic", run_critic_agent),
                ("synthesizer", run_synthesizer_agent),
            ]:
                trace = run_and_trace(name, fn, state)
                agent_traces.append(trace)

        # Gate evaluation
        state.force_approve = req.force_approve
        if state.force_approve:
            state.gate_tier = GatePolicyTier.OVERRIDE

        gate = run_gate(state)
        state.current.gate = gate

        total_tokens = sum(t.input_tokens + t.output_tokens for t in agent_traces)
        total_cost = round(sum(t.cost_usd for t in agent_traces), 6)

        return FullDecisionResponse(
            run_id="local",
            iteration=state.iteration,
            gate_tier=state.gate_tier.value,
            input=state.input.model_dump(),
            delta=(
                {
                    "accepted_risks": req.accepted_risks,
                    "rejected_recommendations": req.rejected_recommendations,
                    "notes": req.notes,
                }
                if req.iteration >= 2
                else None
            ),
            agents=agent_traces,
            gate=gate.model_dump(),
            approved=gate.decision.approved,
            final_recommendation=(
                state.current.synthesizer.final_recommendation
                if state.current.synthesizer
                else None
            ),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/full-trace")
async def decision_full_trace(ws: WebSocket):
    await ws.accept()

    async def ws_emit(evt: Dict):
        await ws.send_json(evt)

    try:
        # Wait for start message
        msg = await ws.receive_json()
        if msg.get("type") != "start":
            await ws.close(code=1003)
            return

        # Initialize state
        state = DecisionState(
            input=DecisionInput(
                decision_question=msg["decision_question"],
                options=msg["options"],
                constraints=msg["constraints"],
            ),
            iteration=1,
            current=CurrentIteration(iteration=1),
            gate_tier=GatePolicyTier.EXPLORATION,
        )

        await ws.send_json({
            "type": "state_initialized",
            "state": state.input.model_dump(),
        })

        total_cost = 0.0
        total_tokens = 0

        # ====================
        # ITERATION 1
        # ====================
        evt = await run_agent_with_realtime_trace("planner", run_planner_agent, state, emit=ws_emit)
        total_cost += evt["cost_usd"]
        total_tokens += evt["input_tokens"] + evt["output_tokens"]

        for name, fn in [
            ("systems", run_systems_agent),
            ("ml_ai", run_ml_ai_agent),
            ("cost", run_cost_agent),
            ("product", run_product_agent),
        ]:
            evt = await run_agent_with_realtime_trace(name, fn, state, emit=ws_emit)
            total_cost += evt["cost_usd"]
            total_tokens += evt["input_tokens"] + evt["output_tokens"]

        # Core agents
        for name, fn in [
            ("detector", run_detector_agent),
            ("critic", run_critic_agent),
            ("synthesizer", run_synthesizer_agent),
        ]:
            evt = await run_agent_with_realtime_trace(name, fn, state, emit=ws_emit)
            total_cost += evt["cost_usd"]
            total_tokens += evt["input_tokens"] + evt["output_tokens"]

        state.force_approve = msg.get("force_approve", False)
        state.current.gate = run_gate(state)
        await ws.send_json({
            "agent": "gate",
            "output": state.current.gate.model_dump(),
        })

        await ws.send_json({
            "type": "iteration_complete",
            "iteration": 1,
            "gate_tier": state.gate_tier.value,
            "approved": state.current.gate.decision.approved,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
        })

        # ====================
        # WAIT FOR USER INPUT
        # ====================
        msg = await ws.receive_json()
        if msg.get("type") != "user_delta":
            await ws.close(code=1003)
            return

        delta = DecisionDelta(
            accepted_risks=msg.get("accepted_risks", []),
            rejected_recommendations=msg.get("rejected_recommendations", []),
            notes=msg.get("notes"),
        )

        state.start_next_iteration(delta)

        await ws.send_json({
            "type": "authority_shift",
            "gate_tier": state.gate_tier.value,
        })

        # ====================
        # ITERATION 2+
        # ====================
        for name, fn in [
            ("detector", run_detector_agent),
            ("critic", run_critic_agent),
            ("synthesizer", run_synthesizer_agent),
        ]:
            evt = await run_agent_with_realtime_trace(name, fn, state, emit=ws_emit)
            total_cost += evt["cost_usd"]
            total_tokens += evt["input_tokens"] + evt["output_tokens"]

        state.force_approve = msg.get("force_approve", False)
        if state.force_approve:
            state.gate_tier = GatePolicyTier.OVERRIDE

        state.current.gate = run_gate(state)
        await ws.send_json({
            "agent": "gate",
            "output": state.current.gate.model_dump(),
        })

        await ws.send_json({
            "type": "run_complete",
            "approved": state.current.gate.decision.approved,
            "final_recommendation": (
                state.current.synthesizer.final_recommendation
                if state.current.synthesizer
                else None
            ),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
        })

    except WebSocketDisconnect:
        return

    except Exception as e:
        await ws.send_json({
            "type": "error",
            "message": str(e),
        })