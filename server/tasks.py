"""
Task definitions for the DevOps Incident Response Environment.

Each task represents a realistic incident scenario with defined:
- Difficulty level
- Initial state
- Success criteria
- Grading rubric
"""

from __future__ import annotations

from typing import Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AlertSeverity, TaskDefinition


# ============================================================================
# Task 1: Easy - OOM Incident
# ============================================================================

TASK_EASY = TaskDefinition(
    id="task_easy_oom",
    name="Memory Crisis: OOM Kill Resolution",
    difficulty="easy",
    description="""
    The order-service has been OOM killed during a traffic spike.
    Customers are unable to place orders, causing revenue loss.

    Your objective:
    1. Identify the root cause of the OOM
    2. Fix the configuration to prevent recurrence
    3. Restore service health

    Hints:
    - Check the order-service logs and metrics
    - Look at the JAVA_OPTS configuration
    - Services need configuration changes AND restarts to take effect
    """.strip(),
    max_steps=15,
    time_limit_seconds=300,
    incident_title="CRITICAL: Order Service Down - Customers Cannot Checkout",
    incident_description="The order-service has been repeatedly OOM killed. All checkout operations are failing. Revenue impact estimated at $50k/hour.",
    incident_severity=AlertSeverity.CRITICAL,
    affected_services=["order-service", "api-gateway"],
    customer_impact="100% of checkout operations failing. Estimated revenue loss: $50k/hour",
    root_cause="order-service OOM due to insufficient heap size",
    required_remediation=["update_config:order-service:JAVA_OPTS", "restart_service:order-service"],
    diagnosis_weight=0.3,
    remediation_weight=0.6,
    efficiency_weight=0.1,
)


# ============================================================================
# Task 2: Medium - Cascading Database Failure
# ============================================================================

TASK_MEDIUM = TaskDefinition(
    id="task_medium_cascade",
    name="Database Overload: Cascading Connection Failure",
    difficulty="medium",
    description="""
    Multiple services are experiencing database connection timeouts.
    The API gateway is showing elevated error rates.

    Your objective:
    1. Identify the root cause of the cascading failures
    2. Determine why the database is overloaded
    3. Fix connection pool configurations
    4. Restore all affected services to healthy state

    This incident has multiple affected services - you need to fix them
    in the right order for the system to recover.
    """.strip(),
    max_steps=25,
    time_limit_seconds=600,
    incident_title="CRITICAL: Multiple Services Failing - Database Connection Issues",
    incident_description="Alert storm: user-service, order-service, and api-gateway all reporting errors. Database appears overloaded. Customer complaints about slow responses and failures.",
    incident_severity=AlertSeverity.CRITICAL,
    affected_services=["postgres-db", "user-service", "order-service", "api-gateway"],
    customer_impact="60% of requests failing or timing out. Customer complaints escalating.",
    root_cause="postgres-db overloaded, connection pools exhausted across services",
    required_remediation=[
        "update_config:postgres-db:MAX_CONNECTIONS",
        "restart_service:postgres-db",
        "update_config:user-service:MAX_POOL_SIZE",
        "restart_service:user-service",
        "update_config:order-service:MAX_POOL_SIZE",
        "restart_service:order-service",
    ],
    diagnosis_weight=0.35,
    remediation_weight=0.55,
    efficiency_weight=0.1,
)


# ============================================================================
# Task 3: Hard - Complex Multi-Factor Incident
# ============================================================================

TASK_HARD = TaskDefinition(
    id="task_hard_complex",
    name="Perfect Storm: Multi-Factor Cascading Incident",
    difficulty="hard",
    description="""
    Multiple alerts firing. The system appears to be in a degraded state
    with several services affected. Initial investigation suggests
    this might not be a single root cause situation.

    Your objective:
    1. Triage the alerts and identify ALL contributing factors
    2. Determine which issues are related vs coincidental
    3. Prioritize and fix issues in the optimal order
    4. Restore full system health

    Warning: This incident has multiple root causes. A simple restart
    won't fix everything. You need to understand the full picture.
    """.strip(),
    max_steps=40,
    time_limit_seconds=900,
    incident_title="CRITICAL: System-Wide Degradation - Multiple Root Causes Suspected",
    incident_description="Alert storm across infrastructure. Redis cluster issues, database slow, API gateway errors at 30%, multiple services degraded. Incident commander has been paged.",
    incident_severity=AlertSeverity.CRITICAL,
    affected_services=["redis-cache", "user-service", "postgres-db", "order-service", "product-service", "api-gateway"],
    customer_impact="30% of requests failing. Search degraded. Checkout slow. Customer satisfaction scores dropping.",
    root_cause="redis-cache network partition causing cascading DB overload, plus unrelated product-service memory leak",
    required_remediation=[
        "restart_service:redis-cache",
        "restart_service:product-service",
        "scale_service:product-service",
    ],
    diagnosis_weight=0.4,
    remediation_weight=0.45,
    efficiency_weight=0.15,
)


# ============================================================================
# Task 4: Expert - Security Breach
# ============================================================================

TASK_EXPERT = TaskDefinition(
    id="task_expert_security",
    name="Security Breach: Credential Stuffing Attack",
    difficulty="expert",
    description="""
    SECURITY INCIDENT: Suspicious activity detected across authentication systems.
    
    Initial reports indicate:
    - Unusual spike in failed authentication attempts
    - Multiple users reporting account lockouts
    - Traffic anomalies from certain IP ranges
    - Potential credential stuffing attack in progress
    
    Your objective:
    1. Investigate the security alerts and identify attack patterns
    2. Determine the attack vector and affected systems
    3. Block malicious traffic while preserving legitimate access
    4. Mitigate immediate impact (unlock affected users, scale for cleanup)
    5. Restore normal security posture
    
    CRITICAL: This is a live security incident. Every minute of delay
    increases the blast radius. Balance speed with accuracy.
    """.strip(),
    max_steps=50,
    time_limit_seconds=1200,
    incident_title="SECURITY ALERT: Active Credential Stuffing Attack Detected",
    incident_description="Security team has detected credential stuffing attack in progress. 35% authentication failure rate, 150+ legitimate users locked out. Attack originating from IP range 185.220.x.x. Immediate response required.",
    incident_severity=AlertSeverity.CRITICAL,
    affected_services=["waf-gateway", "auth-service", "user-service", "redis-cache"],
    customer_impact="Users unable to log in. Account lockouts affecting legitimate users. Potential data exposure risk.",
    root_cause="Credential stuffing attack from IP range 185.220.0.0/16 targeting auth-service",
    required_remediation=[
        "update_config:waf-gateway:BLOCK_LIST",
        "restart_service:waf-gateway",
        "update_config:auth-service:LOCKOUT_DURATION_MIN",
        "restart_service:auth-service",
        "scale_service:auth-service",
    ],
    diagnosis_weight=0.45,
    remediation_weight=0.40,
    efficiency_weight=0.15,
)


# ============================================================================
# Task Registry
# ============================================================================

TASKS: dict[str, TaskDefinition] = {
    "task_easy_oom": TASK_EASY,
    "task_medium_cascade": TASK_MEDIUM,
    "task_hard_complex": TASK_HARD,
    "task_expert_security": TASK_EXPERT,
}


def get_task(task_id: str) -> TaskDefinition:
    """Get a task definition by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> list[dict[str, Any]]:
    """List all available tasks."""
    return [
        {
            "id": task.id,
            "name": task.name,
            "difficulty": task.difficulty,
            "description": task.description[:200] + "...",
            "max_steps": task.max_steps,
        }
        for task in TASKS.values()
    ]
