"""
Runbooks for the DevOps Incident Response Environment.

This module provides runbooks that agents can consult during incident response.
Runbooks contain standard operating procedures for common incident types.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Runbook


# ============================================================================
# Runbook Definitions
# ============================================================================

RUNBOOKS: dict[str, Runbook] = {
    "oom-response": Runbook(
        id="oom-response",
        title="Out of Memory (OOM) Kill Response",
        description="Procedures for responding to OOM kill events in containerized services.",
        symptoms=[
            "Container exit code 137 (SIGKILL)",
            "OutOfMemoryError in application logs",
            "High memory usage preceding crash",
            "Service restart loops (CrashLoopBackOff)",
        ],
        diagnostic_steps=[
            "1. Check service logs for OutOfMemoryError or heap space errors",
            "2. Review memory metrics - check if usage approached limit before crash",
            "3. Check if this coincides with traffic spike or specific request patterns",
            "4. Review recent deployments for memory-related changes",
            "5. Check JAVA_OPTS or equivalent memory configuration",
        ],
        remediation_steps=[
            "1. Increase memory limits (update JAVA_OPTS for Java services)",
            "2. Restart the affected service",
            "3. Monitor memory usage after restart",
            "4. If recurring, investigate memory leaks or optimize code",
            "5. Consider horizontal scaling if memory usage is legitimate",
        ],
        escalation_criteria=[
            "OOM recurring after memory increase",
            "Multiple services affected simultaneously",
            "Customer impact exceeding 10 minutes",
        ],
        related_services=["order-service", "user-service", "product-service"],
    ),
    
    "database-connection-pool": Runbook(
        id="database-connection-pool",
        title="Database Connection Pool Exhaustion",
        description="Procedures for handling database connection pool exhaustion.",
        symptoms=[
            "Connection timeout errors in application logs",
            "HikariPool connection not available errors",
            "Unable to acquire JDBC Connection",
            "High active connection count at limit",
        ],
        diagnostic_steps=[
            "1. Check database active connections vs max connections",
            "2. Review connection pool settings in affected services",
            "3. Check for slow queries that may be holding connections",
            "4. Look for connection leaks (connections not being returned)",
            "5. Check if issue correlates with traffic increase",
        ],
        remediation_steps=[
            "1. Increase database MAX_CONNECTIONS if at limit",
            "2. Restart database service",
            "3. Increase MAX_POOL_SIZE in affected application services",
            "4. Restart application services",
            "5. Monitor connection usage after changes",
        ],
        escalation_criteria=[
            "Database completely unresponsive",
            "Data corruption suspected",
            "Recovery taking more than 15 minutes",
        ],
        related_services=["postgres-db", "user-service", "order-service"],
    ),
    
    "cascading-failure": Runbook(
        id="cascading-failure",
        title="Cascading Failure Response",
        description="Procedures for handling cascading failures across multiple services.",
        symptoms=[
            "Multiple services showing errors simultaneously",
            "Circuit breakers tripping across service mesh",
            "Dependency timeout errors propagating upstream",
            "Error rates increasing in wave pattern",
        ],
        diagnostic_steps=[
            "1. Identify the originating service (earliest alerts)",
            "2. Map the dependency chain to understand blast radius",
            "3. Check shared dependencies (database, cache, message queue)",
            "4. Look for root cause in originating service",
            "5. Assess customer impact at each layer",
        ],
        remediation_steps=[
            "1. Fix root cause in originating service first",
            "2. Wait for circuit breakers to reset OR manually reset",
            "3. Restart affected downstream services in dependency order",
            "4. Verify each layer recovers before proceeding",
            "5. Monitor for full system recovery",
        ],
        escalation_criteria=[
            "Root cause cannot be identified within 10 minutes",
            "More than 3 services critically affected",
            "Customer impact exceeding 50%",
        ],
        related_services=["api-gateway", "all-backend-services"],
    ),
    
    "security-incident": Runbook(
        id="security-incident",
        title="Security Incident Response - Credential Attack",
        description="Procedures for responding to credential stuffing or brute force attacks.",
        symptoms=[
            "Unusual spike in authentication failures",
            "High rate of requests from specific IP ranges",
            "Geographic anomalies in traffic sources",
            "Account lockouts affecting legitimate users",
            "Session enumeration patterns detected",
        ],
        diagnostic_steps=[
            "1. Identify attacking IP ranges from WAF/auth logs",
            "2. Assess scope - how many accounts targeted?",
            "3. Check for successful unauthorized access",
            "4. Review rate limiting effectiveness",
            "5. Identify any compromised credentials",
        ],
        remediation_steps=[
            "1. Block attacking IP ranges at WAF level",
            "2. Restart WAF to apply new rules immediately",
            "3. Increase lockout duration to slow attack",
            "4. Restart auth service with new config",
            "5. Scale auth service to handle cleanup load",
            "6. Reset sessions for potentially compromised accounts",
            "7. Notify security team for further investigation",
        ],
        escalation_criteria=[
            "Evidence of successful unauthorized access",
            "Attack persisting despite blocks",
            "Credential database may be compromised",
            "Attack sophistication suggests advanced threat",
        ],
        related_services=["waf-gateway", "auth-service", "user-service"],
    ),
    
    "cache-failure": Runbook(
        id="cache-failure",
        title="Cache Layer Failure Response",
        description="Procedures for handling Redis cache failures and degradation.",
        symptoms=[
            "Redis cluster unavailable or degraded",
            "Cache hit rate dropping significantly",
            "Increased latency across services using cache",
            "Database load increasing (cache miss thundering herd)",
        ],
        diagnostic_steps=[
            "1. Check Redis cluster health and quorum status",
            "2. Verify network connectivity between Redis nodes",
            "3. Check memory usage and eviction rates",
            "4. Review recent configuration changes",
            "5. Assess downstream impact on dependent services",
        ],
        remediation_steps=[
            "1. Restart Redis cluster to restore quorum",
            "2. Verify cluster rejoins and synchronizes",
            "3. Monitor cache hit rates recovering",
            "4. Scale database temporarily if needed",
            "5. Restart dependent services if circuit breakers stuck",
        ],
        escalation_criteria=[
            "Data loss in cache affecting consistency",
            "Cluster unable to form quorum after restart",
            "Database unable to handle cache-miss load",
        ],
        related_services=["redis-cache", "user-service", "product-service"],
    ),
    
    "bad-deployment": Runbook(
        id="bad-deployment",
        title="Bad Deployment Rollback",
        description="Procedures for identifying and rolling back problematic deployments.",
        symptoms=[
            "Service in CrashLoopBackOff after deployment",
            "Application startup failures",
            "New error types appearing after deploy",
            "Performance degradation correlating with deploy time",
        ],
        diagnostic_steps=[
            "1. Confirm timing correlation with recent deployment",
            "2. Check deployment logs for startup errors",
            "3. Review changes in the deployment (code, config, deps)",
            "4. Check if issue is consistent across all replicas",
            "5. Verify rollback target version is known-good",
        ],
        remediation_steps=[
            "1. Initiate rollback to previous known-good version",
            "2. Verify rollback completes successfully",
            "3. Monitor service health after rollback",
            "4. Document failure for post-incident review",
            "5. Block bad version from redeployment",
        ],
        escalation_criteria=[
            "Rollback also failing",
            "Database migrations were part of bad deploy",
            "Multiple services affected by same deploy",
        ],
        related_services=["all-services"],
    ),
}


def get_runbook(runbook_id: str) -> Runbook | None:
    """Get a runbook by ID."""
    return RUNBOOKS.get(runbook_id)


def search_runbooks(keyword: str) -> list[Runbook]:
    """Search runbooks by keyword in title, description, or symptoms."""
    keyword = keyword.lower()
    results = []
    
    for runbook in RUNBOOKS.values():
        if (keyword in runbook.title.lower() or
            keyword in runbook.description.lower() or
            any(keyword in s.lower() for s in runbook.symptoms) or
            any(keyword in s.lower() for s in runbook.related_services)):
            results.append(runbook)
    
    return results


def get_runbooks_for_service(service_name: str) -> list[Runbook]:
    """Get runbooks related to a specific service."""
    results = []
    service_lower = service_name.lower()
    
    for runbook in RUNBOOKS.values():
        if any(service_lower in s.lower() for s in runbook.related_services):
            results.append(runbook)
        if "all-services" in runbook.related_services or "all-backend-services" in runbook.related_services:
            results.append(runbook)
    
    return list(set(results))  # Remove duplicates


def list_runbooks() -> list[dict]:
    """List all available runbooks."""
    return [
        {
            "id": r.id,
            "title": r.title,
            "description": r.description[:100] + "..." if len(r.description) > 100 else r.description,
        }
        for r in RUNBOOKS.values()
    ]
