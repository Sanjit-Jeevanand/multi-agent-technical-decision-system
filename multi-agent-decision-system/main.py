from fastapi import FastAPI, WebSocket
import asyncio

from multi_agent_decision_system.core.events import (
    init_event_loop,
    register_websocket,
    unregister_websocket,
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    init_event_loop(asyncio.get_running_loop())


@app.websocket("/ws/run/{run_id}")
async def run_events(ws: WebSocket, run_id: str):
    await register_websocket(run_id, ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        await unregister_websocket(run_id, ws)