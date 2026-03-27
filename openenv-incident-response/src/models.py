"""
Pydantic models for the DevOps Incident Response Environment.

This module defines typed models for observations, actions, and rewards
following the OpenEnv specification.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class ServiceStatus(str, Enum):
    """Status of a microservice."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class LogLevel(str, Enum):
    """Log levels."""
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    QUERY_SERVICE = "query_service"
    READ_LOGS = "read_logs"
    GET_METRICS = "get_metrics"
    GET_ALERTS = "get_alerts"
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    ROLLBACK_SERVICE = "rollback_service"
    UPDATE_CONFIG = "update_config"
    RUN_DIAGNOSTIC = "run_diagnostic"
    RESOLVE_INCIDENT = "resolve_incident"


# ============================================================================
# Sub-models for Observations
# ============================================================================

class LogEntry(BaseModel):
    """A single log entry from a service."""
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    trace_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricData(BaseModel):
    """Metrics for a service."""
    service: str
    timestamp: datetime
    cpu_percent: float = Field(ge=0, le=100)
    memory_percent: float = Field(ge=0, le=100)
    memory_mb: float = Field(ge=0)
    memory_limit_mb: float = Field(ge=0)
    request_rate: float = Field(ge=0, description="Requests per second")
    error_rate: float = Field(ge=0, le=100, description="Percentage of requests erroring")
    latency_p50_ms: float = Field(ge=0)
    latency_p99_ms: float = Field(ge=0)
    active_connections: int = Field(ge=0)
    healthy_replicas: int = Field(ge=0)
    total_replicas: int = Field(ge=0)


class Alert(BaseModel):
    """An alert from the monitoring system."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    service: str
    title: str
    description: str
    firing: bool = True
    labels: dict[str, str] = Field(default_factory=dict)


class ServiceInfo(BaseModel):
    """Information about a service."""
    name: str
    status: ServiceStatus
    version: str
    replicas: int
    healthy_replicas: int
    last_deploy: datetime
    dependencies: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    endpoints: list[str] = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    """Result of running diagnostics on a service."""
    service: str
    timestamp: datetime
    checks: dict[str, bool]
    details: dict[str, str]
    recommendations: list[str] = Field(default_factory=list)


class IncidentInfo(BaseModel):
    """Information about the current incident."""
    id: str
    title: str
    severity: AlertSeverity
    started_at: datetime
    description: str
    affected_services: list[str]
    customer_impact: str


# ============================================================================
# Main Observation Model
# ============================================================================

class Observation(BaseModel):
    """
    The observation returned by the environment after each step.

    Contains all information the agent can observe about the current
    state of the incident and infrastructure.
    """
    # Incident context
    incident: IncidentInfo

    # Current step number
    step: int = 0

    # Results from the last action
    last_action: Optional[str] = None
    last_action_success: bool = True
    last_action_error: Optional[str] = None
    last_action_result: Optional[dict[str, Any]] = None

    # Currently visible data (populated by query actions)
    visible_services: dict[str, ServiceInfo] = Field(default_factory=dict)
    visible_logs: list[LogEntry] = Field(default_factory=list)
    visible_metrics: dict[str, MetricData] = Field(default_factory=dict)
    visible_alerts: list[Alert] = Field(default_factory=list)
    visible_diagnostics: dict[str, DiagnosticResult] = Field(default_factory=dict)

    # Action history
    action_history: list[str] = Field(default_factory=list)

    # Available services in the system
    available_services: list[str] = Field(default_factory=list)

    # Hint for the agent (optional, used in easier tasks)
    hint: Optional[str] = None


# ============================================================================
# Action Model
# ============================================================================

class Action(BaseModel):
    """
    An action the agent can take in the environment.

    The agent must specify an action type and relevant parameters.
    """
    action_type: ActionType

    # Target service for service-specific actions
    service: Optional[str] = None

    # Parameters for specific actions
    log_lines: int = Field(default=20, ge=1, le=100, description="Number of log lines to fetch")
    log_level_filter: Optional[LogLevel] = None

    scale_replicas: int = Field(default=1, ge=0, le=10, description="Target replica count")

    config_key: Optional[str] = None
    config_value: Optional[Any] = None

    # For resolve_incident action
    root_cause: Optional[str] = None
    resolution_summary: Optional[str] = None

    # Raw action string (for compatibility)
    action_str: Optional[str] = None

    class Config:
        use_enum_values = True


# ============================================================================
# Reward Model
# ============================================================================

class Reward(BaseModel):
    """
    Reward signal from the environment.

    Provides both a scalar reward and detailed breakdown for interpretability.
    """
    # Main reward value (0.0 to 1.0 for final, can be negative during episode)
    value: float

    # Detailed breakdown
    diagnosis_progress: float = Field(default=0.0, description="Progress toward identifying root cause")
    remediation_progress: float = Field(default=0.0, description="Progress toward fixing the issue")
    efficiency_bonus: float = Field(default=0.0, description="Bonus for efficient resolution")

    # Penalties
    unnecessary_action_penalty: float = Field(default=0.0, description="Penalty for redundant actions")
    harmful_action_penalty: float = Field(default=0.0, description="Penalty for making things worse")
    time_penalty: float = Field(default=0.0, description="Penalty for taking too long")

    # Explanation
    reason: str = ""


# ============================================================================
# State Model (for state() endpoint)
# ============================================================================

class EnvironmentState(BaseModel):
    """
    Full environment state for serialization.

    Used by the state() endpoint for checkpointing/debugging.
    """
    # Task information
    task_id: str
    task_difficulty: str

    # Episode state
    episode_step: int
    max_steps: int
    done: bool

    # Current observation
    observation: Observation

    # Cumulative reward
    total_reward: float

    # Ground truth (for grading)
    root_cause: str
    required_actions: list[str]

    # Internal simulator state
    services_state: dict[str, dict[str, Any]]
    resolved: bool


# ============================================================================
# Step Result Model
# ============================================================================

class StepResult(BaseModel):
    """Result of calling step() on the environment."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Task Definition Model
# ============================================================================

class TaskDefinition(BaseModel):
    """Definition of a task/scenario."""
    id: str
    name: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    max_steps: int
    time_limit_seconds: int

    # Scenario configuration
    incident_title: str
    incident_description: str
    incident_severity: AlertSeverity
    affected_services: list[str]
    customer_impact: str

    # Ground truth
    root_cause: str
    required_remediation: list[str]

    # Scoring weights
    diagnosis_weight: float = 0.4
    remediation_weight: float = 0.5
    efficiency_weight: float = 0.1
