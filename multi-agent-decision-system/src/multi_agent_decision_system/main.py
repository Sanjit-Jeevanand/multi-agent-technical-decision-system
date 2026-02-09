from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from multi_agent_decision_system.api import router as api_router

app = FastAPI()

@app.get("/")
def root():
    return FileResponse("index.html")

app.include_router(api_router)