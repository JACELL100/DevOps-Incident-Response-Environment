"""
FastAPI Server for DevOps Incident Response Environment.

This server exposes the OpenEnv API endpoints for the environment,
allowing AI agents to interact with the incident response simulation
via HTTP requests.

Endpoints:
- POST /reset - Reset the environment
- POST /step - Take an action
- GET /state - Get current state
- GET /tasks - List available tasks
- POST /grade - Grade a completed episode
- GET /health - Health check
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import (
    Action,
    EnvironmentState,
    GradeResult,
    IncidentResponseEnv,
    Observation,
    StepResult,
    grade_task,
    list_tasks,
)


# ============================================================================
# Session Management
# ============================================================================

class SessionManager:
    """Manages environment sessions."""

    def __init__(self):
        self.sessions: dict[str, IncidentResponseEnv] = {}
        self.max_sessions = 100

    def create_session(self, task_id: str) -> str:
        """Create a new session."""
        # Clean up if too many sessions
        if len(self.sessions) >= self.max_sessions:
            oldest = list(self.sessions.keys())[0]
            del self.sessions[oldest]

        session_id = str(uuid.uuid4())
        self.sessions[session_id] = IncidentResponseEnv(task_id=task_id)
        return session_id

    def get_session(self, session_id: str) -> IncidentResponseEnv | None:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


session_manager = SessionManager()


# ============================================================================
# Request/Response Models
# ============================================================================

class ResetRequest(BaseModel):
    """Request to reset the environment."""
    task_id: str = "task_easy_oom"
    session_id: str | None = None


class ResetResponse(BaseModel):
    """Response from reset."""
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    """Request to take a step."""
    session_id: str
    action: Action | None = None
    action_str: str | None = None


class StepResponse(BaseModel):
    """Response from step."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class StateRequest(BaseModel):
    """Request for state."""
    session_id: str


class StateResponse(BaseModel):
    """Response with full state."""
    state: EnvironmentState


class GradeRequest(BaseModel):
    """Request to grade an episode."""
    session_id: str


class GradeResponse(BaseModel):
    """Response with grade."""
    score: float
    diagnosis_score: float
    remediation_score: float
    efficiency_score: float
    details: dict[str, Any]
    feedback: str


class TaskInfo(BaseModel):
    """Information about a task."""
    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int


class TasksResponse(BaseModel):
    """Response with list of tasks."""
    tasks: list[TaskInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    sessions_active: int


# ============================================================================
# Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("DevOps Incident Response Environment starting...")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="DevOps Incident Response Environment",
    description="An OpenEnv-compliant environment for training AI agents in production incident response",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - returns health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        sessions_active=len(session_manager.sessions),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        sessions_active=len(session_manager.sessions),
    )


@app.get("/tasks", response_model=TasksResponse)
async def get_tasks():
    """List all available tasks."""
    tasks = list_tasks()
    return TasksResponse(
        tasks=[TaskInfo(**t) for t in tasks]
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset the environment to initial state.

    Creates a new session or resets an existing one.
    """
    try:
        # Create or get session
        if request.session_id and request.session_id in session_manager.sessions:
            session_id = request.session_id
            env = session_manager.sessions[session_id]
            env.task_id = request.task_id
            env.task = env.task  # Reload task
        else:
            session_id = session_manager.create_session(request.task_id)
            env = session_manager.get_session(session_id)

        # Reset environment
        observation = env.reset()

        return ResetResponse(
            session_id=session_id,
            observation=observation,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Take a step in the environment.

    Executes the specified action and returns the result.
    """
    env = session_manager.get_session(request.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Parse action
        if request.action:
            action = request.action
        elif request.action_str:
            action = Action(action_str=request.action_str)
        else:
            raise HTTPException(status_code=400, detail="No action provided")

        # Execute step
        result = env.step(action)

        return StepResponse(
            observation=result.observation,
            reward=result.reward.value,
            done=result.done,
            info=result.info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state/{session_id}", response_model=StateResponse)
async def get_state(session_id: str):
    """Get the full state of an environment session."""
    env = session_manager.get_session(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        state = env.state()
        return StateResponse(state=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


@app.post("/state", response_model=StateResponse)
async def post_state(request: StateRequest):
    """Get the full state of an environment session (POST version)."""
    return await get_state(request.session_id)


@app.post("/grade", response_model=GradeResponse)
async def grade(request: GradeRequest):
    """
    Grade a completed episode.

    Returns a score from 0.0 to 1.0 based on agent performance.
    """
    env = session_manager.get_session(request.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = grade_task(env.task_id, env)

        return GradeResponse(
            score=result.score,
            diagnosis_score=result.diagnosis_score,
            remediation_score=result.remediation_score,
            efficiency_score=result.efficiency_score,
            details=result.details,
            feedback=result.feedback,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
