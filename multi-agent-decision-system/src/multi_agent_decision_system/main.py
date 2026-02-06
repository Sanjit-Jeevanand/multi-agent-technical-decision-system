from fastapi import FastAPI
from multi_agent_decision_system.api import router
# main.py
import os

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

app = FastAPI(
    title="Multi-Agent Technical Decision System",
    version="0.1.0",
)

app.include_router(router)