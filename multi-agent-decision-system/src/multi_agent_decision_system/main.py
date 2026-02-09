from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Awaitable, Callable
import tiktoken
import asyncio
import time
from math import floor
from concurrent.futures import ThreadPoolExecutor

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
app = FastAPI()

# =========================
# Token/Cost Estimation
# =========================

encoder = tiktoken.encoding_for_model("gpt-4")
GPT5_INPUT_PER_M = 1.25
GPT5_OUTPUT_PER_M = 10.0


def estimate(text: str, *, is_output: bool):
    tokens = floor(len(encoder.encode(text)) / 2)
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
# Enhanced Async Helper for Real-time Updates
# =========================

async def run_agent_with_realtime_trace(
    agent_name: str,
    runner: Callable[[DecisionState], Dict[str, BaseModel]],
    state: DecisionState,
    emit: Optional[Callable[[Dict], Awaitable[None]]] = None,
) -> Dict:
    
    # Prepare input context
    input_context = {
        "input": state.input.model_dump(),
        "current": state.current.model_dump(),
    }
    input_text = str(input_context)
    in_tokens, in_cost = estimate(input_text, is_output=False)

    # Emit start event
    if emit:
        await emit({
            "type": "agent_started",
            "agent": agent_name,
            "timestamp": time.time(),
        })
        # CRITICAL: allow browser to render executing state
        await asyncio.sleep(0.1)

    try:
        # Run the agent (sync)
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
            "timestamp": time.time(),
        }

        if emit:
            # Small delay before completion so UI transition is visible
            await asyncio.sleep(0.05)
            await emit(event)

        return event

    except Exception as e:
        if emit:
            await emit({
                "type": "agent_failed",
                "agent": agent_name,
                "error": str(e),
                "timestamp": time.time(),
            })
        raise


# =========================
# Endpoints
# =========================

@router.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return FileResponse("index.html")


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
async def decision_full_trace_websocket(ws: WebSocket):

    await ws.accept()

    async def ws_emit(evt: Dict):
        """Helper to send JSON events over WebSocket"""
        await ws.send_json(evt)
        # Force flush so UI receives event immediately
        await asyncio.sleep(0)

    # Create ThreadPoolExecutor for parallel specialist agent execution
    executor = ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_running_loop()

    # Helper for running a single specialist agent in parallel (sync runner in thread)
    # Returns the result for thread-safe state update after gather()
    async def run_single_agent_parallel(name, fn):
        # Emit agent_started event immediately (this will show in UI right away)
        await ws_emit({
            "type": "agent_started",
            "agent": name,
            "timestamp": time.time(),
        })
        # CRITICAL: Yield control so WebSocket message is sent immediately
        await asyncio.sleep(0)
        
        # Run sync agent runner in thread pool (non-blocking)
        result = await loop.run_in_executor(executor, fn, state)
        output_model = result.get(name)
        
        # Prepare input context (read-only, safe in parallel)
        input_context = {
            "input": state.input.model_dump(),
            "current": state.current.model_dump(),
        }
        input_text = str(input_context)
        in_tokens, in_cost = estimate(input_text, is_output=False)
        
        if output_model:
            out_text = output_model.model_dump_json()
            out_tokens, out_cost = estimate(out_text, is_output=True)
            output_data = output_model.model_dump()
        else:
            out_tokens = out_cost = 0
            output_data = None
        
        # Emit completion event
        event = {
            "type": "agent_completed",
            "agent": name,
            "input_context": input_context,
            "output": output_data,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(in_cost + out_cost, 6),
            "timestamp": time.time(),
        }
        await ws_emit(event)
        
        # Return data for sequential state update (thread-safe)
        return {
            "name": name,
            "model": output_model,
            "cost_usd": round(in_cost + out_cost, 6),
            "tokens": in_tokens + out_tokens
        }

    try:
        # Wait for start message
        msg = await ws.receive_json()
        if msg.get("type") != "start":
            await ws.close(code=1003, reason="Expected 'start' message")
            executor.shutdown(wait=False)
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
            "timestamp": time.time(),
        })

        total_cost = 0.0
        total_tokens = 0

        # ====================
        # ITERATION 1
        # ====================
        
        # Planner runs first (sequential - others depend on it)
        evt = await run_agent_with_realtime_trace("planner", run_planner_agent, state, emit=ws_emit)
        total_cost += evt["cost_usd"]
        total_tokens += evt["input_tokens"] + evt["output_tokens"]

        # Specialist agents run in TRUE PARALLEL using ThreadPoolExecutor
        specialist_tasks = [
            run_single_agent_parallel(name, fn)
            for name, fn in [
                ("systems", run_systems_agent),
                ("ml_ai", run_ml_ai_agent),
                ("cost", run_cost_agent),
                ("product", run_product_agent),
            ]
        ]
        
        # Wait for all specialists to complete
        specialist_results = await asyncio.gather(*specialist_tasks)
        
        # Thread-safe state update: merge results sequentially after parallel execution
        for res in specialist_results:
            if res["model"]:
                setattr(state.current, res["name"], res["model"])
            total_cost += res["cost_usd"]
            total_tokens += res["tokens"]

        # Core synthesis agents (sequential - they depend on specialist outputs)
        # Each will now show "executing" state individually due to asyncio.sleep(0)
        for name, fn in [
            ("detector", run_detector_agent),
            ("critic", run_critic_agent),
            ("synthesizer", run_synthesizer_agent),
        ]:
            evt = await run_agent_with_realtime_trace(name, fn, state, emit=ws_emit)
            total_cost += evt["cost_usd"]
            total_tokens += evt["input_tokens"] + evt["output_tokens"]

        # Run gate
        state.force_approve = msg.get("force_approve", False)
        state.current.gate = run_gate(state)
        
        await ws.send_json({
            "type": "agent_completed",
            "agent": "gate",
            "output": state.current.gate.model_dump(),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "timestamp": time.time(),
        })

        await ws.send_json({
            "type": "iteration_complete",
            "iteration": 1,
            "gate_tier": state.gate_tier.value,
            "approved": state.current.gate.decision.approved,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "timestamp": time.time(),
        })

        # If approved in iteration 1, we're done
        if state.current.gate.decision.approved:
            await ws.send_json({
                "type": "run_complete",
                "approved": True,
                "final_recommendation": (
                    state.current.synthesizer.final_recommendation
                    if state.current.synthesizer
                    else None
                ),
                "total_cost_usd": round(total_cost, 6),
                "total_tokens": total_tokens,
                "timestamp": time.time(),
            })
            executor.shutdown(wait=False)
            return

        # ====================
        # WAIT FOR USER INPUT
        # ====================
        msg = await ws.receive_json()
        if msg.get("type") != "user_delta":
            await ws.close(code=1003, reason="Expected 'user_delta' message")
            executor.shutdown(wait=False)
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
            "iteration": state.iteration,
            "timestamp": time.time(),
        })

        # ====================
        # ITERATION 2+
        # ====================
        # Only core agents re-run (specialists are frozen from iteration 1)
        for name, fn in [
            ("detector", run_detector_agent),
            ("critic", run_critic_agent),
            ("synthesizer", run_synthesizer_agent),
        ]:
            evt = await run_agent_with_realtime_trace(name, fn, state, emit=ws_emit)
            total_cost += evt["cost_usd"]
            total_tokens += evt["input_tokens"] + evt["output_tokens"]

        # Run gate again
        state.force_approve = msg.get("force_approve", False)
        if state.force_approve:
            state.gate_tier = GatePolicyTier.OVERRIDE

        state.current.gate = run_gate(state)
        
        await ws.send_json({
            "type": "agent_completed",
            "agent": "gate",
            "output": state.current.gate.model_dump(),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "timestamp": time.time(),
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
            "iteration": state.iteration,
            "timestamp": time.time(),
        })

    except WebSocketDisconnect:
        print("Client disconnected")
        executor.shutdown(wait=False)
        return

    except Exception as e:
        await ws.send_json({
            "type": "error",
            "message": str(e),
            "timestamp": time.time(),
        })
        executor.shutdown(wait=False)
        raise


@router.get("/agents/metadata")
def get_agents_metadata():
    """
    Returns metadata about all available agents.
    Useful for UI to display agent information before execution.
    """
    return {
        "agents": [
            {
                "name": "planner",
                "display_name": "Planner Agent",
                "role": "Decomposes complex decisions into sub-questions",
                "execution_order": 1,
                "runs_in_all_iterations": True,
            },
            {
                "name": "systems",
                "display_name": "Systems Agent",
                "role": "Evaluates infrastructure and operational concerns",
                "execution_order": 2,
                "runs_in_all_iterations": False,  # Only iteration 1
            },
            {
                "name": "ml_ai",
                "display_name": "ML/AI Agent",
                "role": "Assesses ML feasibility and requirements",
                "execution_order": 2,
                "runs_in_all_iterations": False,
            },
            {
                "name": "cost",
                "display_name": "Cost Agent",
                "role": "Analyzes financial implications",
                "execution_order": 2,
                "runs_in_all_iterations": False,
            },
            {
                "name": "product",
                "display_name": "Product Agent",
                "role": "Evaluates user experience and market fit",
                "execution_order": 2,
                "runs_in_all_iterations": False,
            },
            {
                "name": "detector",
                "display_name": "Disagreement Detector",
                "role": "Identifies conflicts between recommendations",
                "execution_order": 3,
                "runs_in_all_iterations": True,
            },
            {
                "name": "critic",
                "display_name": "Critic Agent",
                "role": "Challenges assumptions and identifies gaps",
                "execution_order": 4,
                "runs_in_all_iterations": True,
            },
            {
                "name": "synthesizer",
                "display_name": "Synthesizer Agent",
                "role": "Integrates all perspectives into final recommendation",
                "execution_order": 5,
                "runs_in_all_iterations": True,
            },
            {
                "name": "gate",
                "display_name": "Confidence Gate",
                "role": "Validates decision quality",
                "execution_order": 6,
                "runs_in_all_iterations": True,
            },
        ]
    }


@router.get("/agent/{agent_name}/details")
def get_agent_details(agent_name: str):
    """
    Returns detailed information about a specific agent including
    input/output schemas and example data.
    """
    
    agent_schemas = {
        "planner": {
            "name": "Planner Agent",
            "role": "Decomposes complex decisions into sub-questions for specialist agents",
            "model": "GPT-5-mini",
            "input_schema": {
                "decision_question": "str",
                "options": {
                    "option_a": "str",
                    "option_b": "str"
                },
                "constraints": {
                    "latency_sensitivity": "str",
                    "budget_sensitivity": "str",
                    "team_size": "str",
                    "risk_tolerance": "str"
                }
            },
            "output_schema": {
                "systems": {
                    "question": "str",
                    "why_it_matters": "str",
                    "key_unknowns": ["str"]
                },
                "ml_ai": "PlannerSubQuestion",
                "cost": "PlannerSubQuestion",
                "product": "PlannerSubQuestion",
                "assumptions": ["str"],
                "clarifying_questions": ["str"]
            },
            "code_reference": "src/agents/planner_agent.py"
        },
        "systems": {
            "name": "Systems Agent",
            "role": "Evaluates infrastructure, scalability, and operational concerns",
            "model": "GPT-5-mini",
            "input_schema": {
                "decision_question": "str",
                "planner_context": "PlannerSubQuestion",
                "constraints": "dict"
            },
            "output_schema": {
                "recommendation": "option_a | option_b | hybrid | defer",
                "confidence": "float (0.0-1.0)",
                "benefits": ["str (2-4 items)"],
                "risks": ["str (2-4 items)"]
            },
            "code_reference": "src/agents/systems_agent.py"
        },
        "ml_ai": {
            "name": "ML/AI Agent",
            "role": "Assesses ML feasibility, model complexity, and training requirements",
            "model": "GPT-5-mini",
            "input_schema": {
                "decision_question": "str",
                "planner_context": "PlannerSubQuestion",
                "constraints": "dict"
            },
            "output_schema": {
                "recommendation": "option_a | option_b | hybrid | defer",
                "confidence": "float (0.0-1.0)",
                "benefits": ["str (2-4 items)"],
                "risks": ["str (2-4 items)"]
            },
            "code_reference": "src/agents/ml_ai_agent.py"
        },
        "cost": {
            "name": "Cost Agent",
            "role": "Analyzes financial implications and ROI considerations",
            "model": "GPT-5-mini",
            "input_schema": {
                "decision_question": "str",
                "planner_context": "PlannerSubQuestion",
                "constraints": "dict"
            },
            "output_schema": {
                "recommendation": "option_a | option_b | hybrid | defer",
                "confidence": "float (0.0-1.0)",
                "benefits": ["str (2-4 items)"],
                "risks": ["str (2-4 items)"]
            },
            "code_reference": "src/agents/cost_agent.py"
        },
        "product": {
            "name": "Product Agent",
            "role": "Evaluates user experience, market fit, and feature value",
            "model": "GPT-5-mini",
            "input_schema": {
                "decision_question": "str",
                "planner_context": "PlannerSubQuestion",
                "constraints": "dict"
            },
            "output_schema": {
                "recommendation": "option_a | option_b | hybrid | defer",
                "confidence": "float (0.0-1.0)",
                "benefits": ["str (2-4 items)"],
                "risks": ["str (2-4 items)"]
            },
            "code_reference": "src/agents/product_agent.py"
        },
        "detector": {
            "name": "Disagreement Detector",
            "role": "Identifies conflicts and inconsistencies between specialist recommendations",
            "model": "GPT-5-mini",
            "input_schema": {
                "systems": "SpecialistOutput",
                "ml_ai": "SpecialistOutput",
                "cost": "SpecialistOutput",
                "product": "SpecialistOutput"
            },
            "output_schema": {
                "conflicts": [{
                    "agents_involved": ["str"],
                    "description": "str",
                    "severity": "low | medium | high"
                }],
                "has_blocking_conflicts": "bool"
            },
            "code_reference": "src/agents/detector_agent.py"
        },
        "critic": {
            "name": "Critic Agent",
            "role": "Challenges assumptions and identifies gaps in reasoning",
            "model": "GPT-5-mini",
            "input_schema": {
                "all_agent_outputs": "dict",
                "detected_conflicts": "DetectorOutput"
            },
            "output_schema": {
                "issues": [{
                    "agent": "str",
                    "issue": "str",
                    "impact": "low | medium | high"
                }],
                "requires_revision": "bool"
            },
            "code_reference": "src/agents/critic_agent.py"
        },
        "synthesizer": {
            "name": "Synthesizer Agent",
            "role": "Integrates all perspectives into a final recommendation",
            "model": "GPT-5.1",
            "input_schema": {
                "all_specialist_outputs": "dict",
                "conflicts": "DetectorOutput",
                "critic_feedback": "CriticOutput"
            },
            "output_schema": {
                "final_recommendation": "option_a | option_b | hybrid | defer",
                "confidence": "float (0.0-1.0)",
                "rationale": ["str"],
                "tradeoffs": ["str"],
                "unresolved_risks": ["str"]
            },
            "code_reference": "src/agents/synthesizer_agent.py"
        },
        "gate": {
            "name": "Confidence Gate",
            "role": "Validates decision quality and determines if approval criteria are met",
            "model": "Rule Based",
            "input_schema": {
                "synthesizer_output": "SynthesizerOutput",
                "gate_tier": "exploration | commitment | override",
                "force_approve": "bool"
            },
            "output_schema": {
                "decision": {
                    "approved": "bool",
                    "blocking_reasons": ["str"],
                    "required_actions": ["str"]
                }
            },
            "code_reference": "src/agents/gate_agent.py"
        }
    }
    
    if agent_name not in agent_schemas:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent '{agent_name}' not found. Available agents: {list(agent_schemas.keys())}"
        )
    
    return agent_schemas[agent_name]


@router.get("/execution/status")
def get_execution_status():
    """
    Returns the current execution status.
    Can be polled by UI for updates (alternative to WebSocket).
    """
    return {
        "status": "idle",
        "message": "Use WebSocket endpoint for real-time execution tracking"
    }


@router.post("/agent/{agent_name}/simulate")
def simulate_agent_execution(agent_name: str, input_data: Dict):
    """
    Simulates execution of a single agent for testing.
    Useful for UI development and debugging.
    """
    
    try:
        # Create minimal state for single agent testing
        state = DecisionState(
            input=DecisionInput(
                decision_question=input_data.get("decision_question", "Test decision"),
                options=input_data.get("options", {"option_a": "A", "option_b": "B"}),
                constraints=input_data.get("constraints", {}),
            ),
            iteration=1,
            current=CurrentIteration(iteration=1),
        )
        
        # Map agent names to their runner functions
        agent_runners = {
            "planner": run_planner_agent,
            "systems": run_systems_agent,
            "ml_ai": run_ml_ai_agent,
            "cost": run_cost_agent,
            "product": run_product_agent,
            "detector": run_detector_agent,
            "critic": run_critic_agent,
            "synthesizer": run_synthesizer_agent,
        }
        
        if agent_name not in agent_runners:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found"
            )
        
        # Run the agent
        result = agent_runners[agent_name](state)
        output = result.get(agent_name)
        
        if output:
            return {
                "agent": agent_name,
                "status": "success",
                "output": output.model_dump(),
                "input_context": {
                    "decision_question": state.input.decision_question,
                    "options": state.input.options,
                    "constraints": state.input.constraints
                }
            }
        else:
            return {
                "agent": agent_name,
                "status": "no_output",
                "message": "Agent executed but returned no output"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent simulation failed: {str(e)}"
        )
    
app.include_router(router)