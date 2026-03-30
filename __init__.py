"""
DevOps Incident Response Environment for OpenEnv.

A reinforcement learning environment for training AI agents
to handle production incident response in microservices architectures.

This is the main package that exports all public interfaces
following the OpenEnv scaffold convention.
"""

__version__ = "1.0.0"

# Export models (required by OpenEnv spec)
from models import (
    Action,
    ActionType,
    Alert,
    AlertSeverity,
    DiagnosticResult,
    EnvironmentState,
    IncidentInfo,
    LogEntry,
    LogLevel,
    MetricData,
    Observation,
    Reward,
    ServiceInfo,
    ServiceStatus,
    StepResult,
    TaskDefinition,
)

# Export server components
from server.environment import IncidentResponseEnv
from server.graders import GradeResult, TaskGrader, grade_task, grade_all_tasks
from server.simulator import InfrastructureSimulator, ServiceSimulator
from server.tasks import TASKS, get_task, list_tasks

# Convenience alias for the main environment class
Env = IncidentResponseEnv

__all__ = [
    # Version
    "__version__",
    # Main environment
    "IncidentResponseEnv",
    "Env",
    # Models
    "Action",
    "ActionType",
    "Alert",
    "AlertSeverity",
    "DiagnosticResult",
    "EnvironmentState",
    "IncidentInfo",
    "LogEntry",
    "LogLevel",
    "MetricData",
    "Observation",
    "Reward",
    "ServiceInfo",
    "ServiceStatus",
    "StepResult",
    "TaskDefinition",
    # Simulator
    "InfrastructureSimulator",
    "ServiceSimulator",
    # Graders
    "GradeResult",
    "TaskGrader",
    "grade_task",
    "grade_all_tasks",
    # Tasks
    "TASKS",
    "get_task",
    "list_tasks",
]
