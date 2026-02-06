from __future__ import annotations

import asyncio
from typing import Dict, List
from pydantic import BaseModel
from fastapi import WebSocket
from datetime import datetime, timezone

from multi_agent_decision_system.core.schema import AgentExecutionEvent


# =====================================================
# In-memory event store (per run)
# =====================================================

_EVENT_STORE: Dict[str, List[AgentExecutionEvent]] = {}

# Active WebSocket subscribers per run_id
_SUBSCRIBERS: Dict[str, List[WebSocket]] = {}

# Global event loop reference (set by FastAPI)
_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


# =====================================================
# Initialization
# =====================================================

def init_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Must be called once from FastAPI startup.
    """
    global _EVENT_LOOP
    _EVENT_LOOP = loop


# =====================================================
# Event Emission (Core API)
# =====================================================

def emit_event(event: AgentExecutionEvent) -> None:
    """
    Emits an event safely from sync or async code.

    - Stores event in memory
    - Broadcasts to WebSocket clients
    """

    if event.timestamp is None:
        event.timestamp = datetime.now(timezone.utc)

    # Store event
    _EVENT_STORE.setdefault(event.run_id, []).append(event)

    # Broadcast (thread-safe)
    if _EVENT_LOOP:
        asyncio.run_coroutine_threadsafe(
            _broadcast_event(event),
            _EVENT_LOOP,
        )


# =====================================================
# WebSocket Management
# =====================================================

async def register_websocket(run_id: str, websocket: WebSocket) -> None:
    """
    Register a WebSocket client for a run_id.
    Sends historical events immediately.
    """
    await websocket.accept()

    _SUBSCRIBERS.setdefault(run_id, []).append(websocket)

    # Replay history for late joiners
    for event in _EVENT_STORE.get(run_id, []):
        await websocket.send_json(event.model_dump())


async def unregister_websocket(run_id: str, websocket: WebSocket) -> None:
    """
    Remove a WebSocket client.
    """
    if run_id in _SUBSCRIBERS:
        _SUBSCRIBERS[run_id] = [
            ws for ws in _SUBSCRIBERS[run_id] if ws is not websocket
        ]


# =====================================================
# Internal Broadcaster
# =====================================================

async def _broadcast_event(event: AgentExecutionEvent) -> None:
    """
    Send event to all active subscribers.
    """
    subscribers = _SUBSCRIBERS.get(event.run_id, [])

    dead_sockets = []

    for ws in subscribers:
        try:
            await ws.send_json(event.model_dump())
        except Exception:
            dead_sockets.append(ws)

    # Cleanup broken connections
    for ws in dead_sockets:
        await unregister_websocket(event.run_id, ws)


# =====================================================
# Optional Utilities
# =====================================================

def get_events_for_run(run_id: str) -> List[AgentExecutionEvent]:
    """
    Used for debugging, testing, or REST fallback.
    """
    return _EVENT_STORE.get(run_id, [])