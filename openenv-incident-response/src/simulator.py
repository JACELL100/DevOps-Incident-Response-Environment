"""
Infrastructure Simulator for the DevOps Incident Response Environment.

This module simulates a microservices architecture with realistic:
- Service health states and dependencies
- Log generation with error patterns
- Metrics (CPU, memory, latency, error rates)
- Alerts from monitoring systems
- Service behaviors and failure modes
"""

from __future__ import annotations

import copy
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from .models import (
    Alert,
    AlertSeverity,
    DiagnosticResult,
    LogEntry,
    LogLevel,
    MetricData,
    ServiceInfo,
    ServiceStatus,
)


class ServiceSimulator:
    """Simulates a single microservice."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        replicas: int = 3,
        dependencies: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.name = name
        self.version = version
        self.replicas = replicas
        self.healthy_replicas = replicas
        self.dependencies = dependencies or []
        self.config = config or {}
        self.endpoints = [f"/{name}/health", f"/{name}/api/v1"]
        self.last_deploy = datetime.now() - timedelta(days=random.randint(1, 30))

        # Internal state
        self._status = ServiceStatus.HEALTHY
        self._cpu_base = random.uniform(10, 30)
        self._memory_base = random.uniform(20, 40)
        self._memory_limit = 512.0
        self._latency_base = random.uniform(10, 50)
        self._error_rate = 0.0
        self._request_rate = random.uniform(100, 500)

        # Failure injection
        self._failure_mode: Optional[str] = None
        self._failure_params: dict[str, Any] = {}

        # Log buffer
        self._logs: list[LogEntry] = []
        self._log_id = 0

    def inject_failure(self, mode: str, params: dict[str, Any] | None = None):
        """Inject a failure mode into the service."""
        self._failure_mode = mode
        self._failure_params = params or {}
        self._apply_failure_effects()

    def clear_failure(self):
        """Clear any injected failures."""
        self._failure_mode = None
        self._failure_params = {}
        self._status = ServiceStatus.HEALTHY
        self.healthy_replicas = self.replicas
        self._error_rate = 0.0

    def _apply_failure_effects(self):
        """Apply effects based on the current failure mode."""
        mode = self._failure_mode
        params = self._failure_params

        if mode == "oom":
            # Out of memory - service crashes and restarts
            self._status = ServiceStatus.UNHEALTHY
            self.healthy_replicas = 0
            self._memory_base = 95.0
            self._add_log(LogLevel.ERROR, "OutOfMemoryError: Java heap space")
            self._add_log(LogLevel.ERROR, "Container killed due to OOM (exit code 137)")
            self._add_log(LogLevel.WARN, "Service restarting...")
            self._error_rate = 100.0

        elif mode == "memory_leak":
            # Memory leak - gradual memory increase
            self._memory_base = params.get("memory_percent", 85.0)
            self._status = ServiceStatus.DEGRADED
            self.healthy_replicas = max(1, self.replicas - 1)
            self._add_log(LogLevel.WARN, f"High memory usage detected: {self._memory_base:.1f}%")
            self._add_log(LogLevel.WARN, "GC overhead limit exceeded")
            self._error_rate = 15.0

        elif mode == "cpu_spike":
            # CPU spike - high CPU usage
            self._cpu_base = params.get("cpu_percent", 95.0)
            self._status = ServiceStatus.DEGRADED
            self._latency_base = 500.0
            self._add_log(LogLevel.WARN, f"High CPU usage: {self._cpu_base:.1f}%")
            self._add_log(LogLevel.WARN, "Request processing delayed due to CPU contention")
            self._error_rate = 5.0

        elif mode == "dependency_timeout":
            # Dependency is timing out
            dep = params.get("dependency", "unknown")
            self._status = ServiceStatus.DEGRADED
            self._latency_base = 5000.0  # 5 second timeout
            self._error_rate = 50.0
            self._add_log(LogLevel.ERROR, f"Connection timeout to {dep}: Read timed out after 5000ms")
            self._add_log(LogLevel.ERROR, f"CircuitBreaker for {dep} is OPEN")
            self._add_log(LogLevel.WARN, "Fallback activated for degraded dependency")

        elif mode == "database_connection_pool":
            # Database connection pool exhausted
            self._status = ServiceStatus.DEGRADED
            self._error_rate = 60.0
            self._add_log(LogLevel.ERROR, "HikariPool-1 - Connection is not available, request timed out after 30000ms")
            self._add_log(LogLevel.ERROR, "Unable to acquire JDBC Connection")
            self._add_log(LogLevel.WARN, f"Active connections: {params.get('active', 50)}/{params.get('max', 50)}")

        elif mode == "bad_deploy":
            # Bad deployment - service is crashing on startup
            self._status = ServiceStatus.UNHEALTHY
            self.healthy_replicas = 0
            self._error_rate = 100.0
            self._add_log(LogLevel.ERROR, "Application failed to start")
            self._add_log(LogLevel.ERROR, f"NullPointerException at {self.name}.Application.main(Application.java:42)")
            self._add_log(LogLevel.ERROR, "CrashLoopBackOff: restarting failed container")

        elif mode == "config_error":
            # Configuration error
            key = params.get("config_key", "UNKNOWN_CONFIG")
            self._status = ServiceStatus.UNHEALTHY
            self.healthy_replicas = 0
            self._error_rate = 100.0
            self._add_log(LogLevel.ERROR, f"Failed to parse configuration: {key}")
            self._add_log(LogLevel.ERROR, "IllegalArgumentException: Invalid configuration value")

        elif mode == "network_partition":
            # Network partition - some replicas can't communicate
            self._status = ServiceStatus.DEGRADED
            self.healthy_replicas = 1
            self._error_rate = 30.0
            self._add_log(LogLevel.ERROR, "Failed to connect to peer nodes")
            self._add_log(LogLevel.WARN, "Cluster quorum lost, running in degraded mode")

        elif mode == "disk_full":
            # Disk full
            self._status = ServiceStatus.UNHEALTHY
            self._error_rate = 80.0
            self._add_log(LogLevel.ERROR, "No space left on device")
            self._add_log(LogLevel.ERROR, "Failed to write to /data/logs/application.log")

        elif mode == "cascading":
            # Cascading failure from dependency
            self._status = ServiceStatus.UNHEALTHY
            self.healthy_replicas = 0
            self._error_rate = 100.0
            self._add_log(LogLevel.ERROR, "Upstream service unavailable")
            self._add_log(LogLevel.ERROR, "Circuit breaker tripped for all dependencies")

    def _add_log(self, level: LogLevel, message: str, trace_id: str | None = None):
        """Add a log entry."""
        self._log_id += 1
        self._logs.append(
            LogEntry(
                timestamp=datetime.now(),
                level=level,
                service=self.name,
                message=message,
                trace_id=trace_id or f"trace-{uuid.uuid4().hex[:8]}",
            )
        )

    def get_status(self) -> ServiceStatus:
        """Get current service status."""
        return self._status

    def get_info(self) -> ServiceInfo:
        """Get service information."""
        return ServiceInfo(
            name=self.name,
            status=self._status,
            version=self.version,
            replicas=self.replicas,
            healthy_replicas=self.healthy_replicas,
            last_deploy=self.last_deploy,
            dependencies=self.dependencies,
            config=self.config,
            endpoints=self.endpoints,
        )

    def get_metrics(self) -> MetricData:
        """Get current metrics."""
        # Add some noise to metrics
        cpu = min(100, max(0, self._cpu_base + random.uniform(-5, 5)))
        memory = min(100, max(0, self._memory_base + random.uniform(-2, 2)))
        latency = max(1, self._latency_base + random.uniform(-10, 10))

        return MetricData(
            service=self.name,
            timestamp=datetime.now(),
            cpu_percent=cpu,
            memory_percent=memory,
            memory_mb=(memory / 100) * self._memory_limit,
            memory_limit_mb=self._memory_limit,
            request_rate=self._request_rate + random.uniform(-20, 20),
            error_rate=self._error_rate,
            latency_p50_ms=latency,
            latency_p99_ms=latency * 3,
            active_connections=random.randint(10, 100),
            healthy_replicas=self.healthy_replicas,
            total_replicas=self.replicas,
        )

    def get_logs(self, count: int = 20, level_filter: LogLevel | None = None) -> list[LogEntry]:
        """Get recent logs."""
        logs = self._logs
        if level_filter:
            logs = [log for log in logs if log.level == level_filter]
        return logs[-count:]

    def restart(self) -> bool:
        """Restart the service."""
        self._add_log(LogLevel.INFO, "Service restart initiated")

        # Restart clears some failures
        if self._failure_mode in ["oom", "bad_deploy"]:
            # These persist through restart
            self._add_log(LogLevel.ERROR, "Service failed to start after restart")
            return False

        if self._failure_mode == "memory_leak":
            # Restart temporarily fixes memory leak
            self._memory_base = random.uniform(20, 40)
            self._status = ServiceStatus.HEALTHY
            self.healthy_replicas = self.replicas
            self._error_rate = 0.0
            self._add_log(LogLevel.INFO, "Service restarted successfully")
            return True

        if self._failure_mode is None:
            self._add_log(LogLevel.INFO, "Service restarted successfully")
            return True

        return False

    def scale(self, replicas: int) -> bool:
        """Scale the service."""
        if replicas < 0 or replicas > 10:
            self._add_log(LogLevel.ERROR, f"Invalid replica count: {replicas}")
            return False

        old_replicas = self.replicas
        self.replicas = replicas
        self._add_log(LogLevel.INFO, f"Scaling from {old_replicas} to {replicas} replicas")

        # If failure mode affects replicas, recalculate healthy ones
        if self._failure_mode:
            self.healthy_replicas = min(replicas, self.healthy_replicas)
        else:
            self.healthy_replicas = replicas

        return True

    def rollback(self, target_version: str = "previous") -> bool:
        """Rollback the service."""
        self._add_log(LogLevel.INFO, f"Rolling back to {target_version}")

        if self._failure_mode == "bad_deploy":
            # Rollback fixes bad deploy
            self.clear_failure()
            self.version = "0.9.0"  # Previous version
            self._add_log(LogLevel.INFO, "Rollback successful, service is healthy")
            return True

        self._add_log(LogLevel.INFO, "Rollback completed")
        return True

    def update_config(self, key: str, value: Any) -> bool:
        """Update service configuration."""
        self._add_log(LogLevel.INFO, f"Updating config: {key}={value}")
        self.config[key] = value

        if self._failure_mode == "config_error":
            if key == self._failure_params.get("config_key"):
                # Correct config fix doesn't auto-resolve; needs restart
                self._add_log(LogLevel.INFO, "Configuration updated. Restart required.")
                return True

        if self._failure_mode == "oom" and key == "JAVA_OPTS" and "-Xmx" in str(value):
            # Fixing memory limit
            self._add_log(LogLevel.INFO, "Memory configuration updated. Restart required.")
            self._failure_params["memory_fixed"] = True
            return True

        if self._failure_mode == "database_connection_pool" and key == "MAX_POOL_SIZE":
            # Increasing connection pool
            self._add_log(LogLevel.INFO, "Connection pool size updated. Restart required.")
            return True

        return True

    def run_diagnostic(self) -> DiagnosticResult:
        """Run diagnostics on the service."""
        checks = {
            "health_endpoint": self._status != ServiceStatus.DOWN,
            "memory_ok": self._memory_base < 90,
            "cpu_ok": self._cpu_base < 90,
            "dependencies_ok": self._failure_mode != "dependency_timeout",
            "disk_ok": self._failure_mode != "disk_full",
            "config_ok": self._failure_mode != "config_error",
        }

        details = {}
        recommendations = []

        if not checks["memory_ok"]:
            details["memory"] = f"High memory usage: {self._memory_base:.1f}%"
            recommendations.append("Consider increasing memory limit or fixing memory leak")

        if not checks["cpu_ok"]:
            details["cpu"] = f"High CPU usage: {self._cpu_base:.1f}%"
            recommendations.append("Check for CPU-intensive operations or scale horizontally")

        if not checks["dependencies_ok"]:
            dep = self._failure_params.get("dependency", "unknown")
            details["dependencies"] = f"Dependency {dep} is unhealthy"
            recommendations.append(f"Check status of {dep} service")

        if not checks["disk_ok"]:
            details["disk"] = "Disk space critically low"
            recommendations.append("Clear old logs or increase disk size")

        if not checks["config_ok"]:
            key = self._failure_params.get("config_key", "unknown")
            details["config"] = f"Invalid configuration: {key}"
            recommendations.append(f"Fix configuration value for {key}")

        if self._failure_mode == "oom":
            details["memory"] = "OutOfMemoryError detected in logs"
            recommendations.append("Increase heap size via JAVA_OPTS or fix memory leak")

        if self._failure_mode == "bad_deploy":
            details["deployment"] = "Application failing to start"
            recommendations.append("Consider rolling back to previous version")

        return DiagnosticResult(
            service=self.name,
            timestamp=datetime.now(),
            checks=checks,
            details=details,
            recommendations=recommendations,
        )

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "name": self.name,
            "version": self.version,
            "replicas": self.replicas,
            "healthy_replicas": self.healthy_replicas,
            "status": self._status.value,
            "failure_mode": self._failure_mode,
            "failure_params": self._failure_params,
            "cpu_base": self._cpu_base,
            "memory_base": self._memory_base,
            "error_rate": self._error_rate,
        }

    def load_state(self, state: dict[str, Any]):
        """Load state from dict."""
        self.version = state["version"]
        self.replicas = state["replicas"]
        self.healthy_replicas = state["healthy_replicas"]
        self._status = ServiceStatus(state["status"])
        self._failure_mode = state.get("failure_mode")
        self._failure_params = state.get("failure_params", {})
        self._cpu_base = state.get("cpu_base", 20.0)
        self._memory_base = state.get("memory_base", 30.0)
        self._error_rate = state.get("error_rate", 0.0)


class InfrastructureSimulator:
    """
    Simulates a complete microservices infrastructure.

    Manages multiple services with dependencies, generates realistic
    telemetry data, and handles failure scenarios.
    """

    def __init__(self):
        self.services: dict[str, ServiceSimulator] = {}
        self.alerts: list[Alert] = []
        self._alert_id = 0
        self._global_time = datetime.now()

    def add_service(
        self,
        name: str,
        version: str = "1.0.0",
        replicas: int = 3,
        dependencies: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> ServiceSimulator:
        """Add a service to the infrastructure."""
        service = ServiceSimulator(name, version, replicas, dependencies, config)
        self.services[name] = service
        return service

    def create_standard_infrastructure(self):
        """Create a standard microservices setup."""
        # API Gateway
        self.add_service(
            "api-gateway",
            version="2.1.0",
            replicas=3,
            dependencies=["user-service", "order-service", "product-service"],
            config={"RATE_LIMIT": 1000, "TIMEOUT_MS": 5000},
        )

        # User Service
        self.add_service(
            "user-service",
            version="1.5.2",
            replicas=3,
            dependencies=["postgres-db", "redis-cache"],
            config={"MAX_POOL_SIZE": 50, "CACHE_TTL": 300},
        )

        # Order Service
        self.add_service(
            "order-service",
            version="3.0.1",
            replicas=4,
            dependencies=["postgres-db", "user-service", "payment-service", "kafka"],
            config={"JAVA_OPTS": "-Xmx512m -Xms256m", "MAX_POOL_SIZE": 50},
        )

        # Product Service
        self.add_service(
            "product-service",
            version="1.2.0",
            replicas=2,
            dependencies=["postgres-db", "elasticsearch"],
            config={"CACHE_ENABLED": True},
        )

        # Payment Service
        self.add_service(
            "payment-service",
            version="2.0.0",
            replicas=3,
            dependencies=["payment-gateway-external"],
            config={"RETRY_COUNT": 3, "TIMEOUT_MS": 30000},
        )

        # Database
        self.add_service(
            "postgres-db",
            version="14.2",
            replicas=1,
            config={"MAX_CONNECTIONS": 200, "SHARED_BUFFERS": "256MB"},
        )

        # Cache
        self.add_service(
            "redis-cache",
            version="6.2",
            replicas=3,
            config={"MAXMEMORY": "512mb", "EVICTION_POLICY": "allkeys-lru"},
        )

        # Message Queue
        self.add_service(
            "kafka",
            version="3.0",
            replicas=3,
            config={"REPLICATION_FACTOR": 3, "NUM_PARTITIONS": 10},
        )

        # Search
        self.add_service(
            "elasticsearch",
            version="8.0",
            replicas=3,
            config={"HEAP_SIZE": "1g"},
        )

    def inject_failure(self, service_name: str, mode: str, params: dict[str, Any] | None = None):
        """Inject a failure into a service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")

        service = self.services[service_name]
        service.inject_failure(mode, params)

        # Generate appropriate alerts
        self._generate_alerts_for_failure(service_name, mode, params)

    def _generate_alerts_for_failure(self, service_name: str, mode: str, params: dict[str, Any] | None = None):
        """Generate alerts based on failure mode."""
        params = params or {}

        if mode == "oom":
            self._add_alert(
                AlertSeverity.CRITICAL,
                service_name,
                f"{service_name} OOMKilled",
                f"Container {service_name} was killed due to out of memory. Exit code 137.",
            )

        elif mode == "memory_leak":
            self._add_alert(
                AlertSeverity.WARNING,
                service_name,
                f"{service_name} High Memory Usage",
                f"Memory usage is at {params.get('memory_percent', 85)}% of limit.",
            )

        elif mode == "cpu_spike":
            self._add_alert(
                AlertSeverity.WARNING,
                service_name,
                f"{service_name} High CPU",
                f"CPU usage is at {params.get('cpu_percent', 95)}%.",
            )

        elif mode == "dependency_timeout":
            dep = params.get("dependency", "unknown")
            self._add_alert(
                AlertSeverity.CRITICAL,
                service_name,
                f"{service_name} Dependency Timeout",
                f"Service {service_name} cannot reach {dep}. Connections timing out.",
            )

        elif mode == "database_connection_pool":
            self._add_alert(
                AlertSeverity.CRITICAL,
                service_name,
                f"{service_name} Connection Pool Exhausted",
                f"Database connection pool is exhausted. Active: {params.get('active', 50)}/{params.get('max', 50)}",
            )

        elif mode == "bad_deploy":
            self._add_alert(
                AlertSeverity.CRITICAL,
                service_name,
                f"{service_name} CrashLoopBackOff",
                f"Pod {service_name} is in CrashLoopBackOff. Application failing to start.",
            )

        elif mode == "cascading":
            self._add_alert(
                AlertSeverity.CRITICAL,
                service_name,
                f"{service_name} Service Down",
                f"Service {service_name} is completely unavailable due to cascading failure.",
            )

    def _add_alert(self, severity: AlertSeverity, service: str, title: str, description: str):
        """Add a new alert."""
        self._alert_id += 1
        self.alerts.append(
            Alert(
                id=f"alert-{self._alert_id}",
                timestamp=datetime.now(),
                severity=severity,
                service=service,
                title=title,
                description=description,
                firing=True,
                labels={"env": "production", "team": "platform"},
            )
        )

    def get_alerts(self, service_name: str | None = None) -> list[Alert]:
        """Get active alerts, optionally filtered by service."""
        alerts = [a for a in self.alerts if a.firing]
        if service_name:
            alerts = [a for a in alerts if a.service == service_name]
        return alerts

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.firing = False

    def get_service_names(self) -> list[str]:
        """Get list of all service names."""
        return list(self.services.keys())

    def get_state(self) -> dict[str, dict[str, Any]]:
        """Get full state for serialization."""
        return {name: svc.get_state() for name, svc in self.services.items()}

    def load_state(self, state: dict[str, dict[str, Any]]):
        """Load state from dict."""
        for name, svc_state in state.items():
            if name in self.services:
                self.services[name].load_state(svc_state)

    def reset(self):
        """Reset all services to healthy state."""
        for service in self.services.values():
            service.clear_failure()
            service._logs.clear()
        self.alerts.clear()
        self._alert_id = 0


# ============================================================================
# Scenario Presets
# ============================================================================

def create_oom_scenario(infra: InfrastructureSimulator) -> dict[str, Any]:
    """
    Scenario: Order service running out of memory during peak traffic.

    Root cause: Heap size too small for traffic volume.
    Solution: Increase JAVA_OPTS heap size and restart.
    """
    infra.create_standard_infrastructure()
    infra.inject_failure("order-service", "oom")

    return {
        "root_cause": "order-service OOM due to insufficient heap size",
        "required_remediation": [
            "update_config:order-service:JAVA_OPTS:-Xmx1024m -Xms512m",
            "restart_service:order-service",
        ],
        "affected_services": ["order-service", "api-gateway"],
    }


def create_cascading_failure_scenario(infra: InfrastructureSimulator) -> dict[str, Any]:
    """
    Scenario: Database connection pool exhausted, causing cascading failures.

    Root cause: postgres-db max connections too low for current load.
    Solution: Increase database connections and connection pool sizes.
    """
    infra.create_standard_infrastructure()

    # Database is overloaded
    infra.inject_failure("postgres-db", "cpu_spike", {"cpu_percent": 98})

    # Services can't get connections
    infra.inject_failure("user-service", "database_connection_pool", {"active": 50, "max": 50})
    infra.inject_failure("order-service", "database_connection_pool", {"active": 50, "max": 50})

    # API gateway sees failures
    infra.services["api-gateway"]._status = ServiceStatus.DEGRADED
    infra.services["api-gateway"]._error_rate = 40.0

    infra._add_alert(
        AlertSeverity.CRITICAL,
        "api-gateway",
        "High Error Rate",
        "API Gateway error rate at 40%. Multiple downstream services affected.",
    )

    return {
        "root_cause": "postgres-db overloaded, connection pools exhausted across services",
        "required_remediation": [
            "update_config:postgres-db:MAX_CONNECTIONS:500",
            "restart_service:postgres-db",
            "update_config:user-service:MAX_POOL_SIZE:100",
            "restart_service:user-service",
            "update_config:order-service:MAX_POOL_SIZE:100",
            "restart_service:order-service",
        ],
        "affected_services": ["postgres-db", "user-service", "order-service", "api-gateway"],
    }


def create_bad_deploy_scenario(infra: InfrastructureSimulator) -> dict[str, Any]:
    """
    Scenario: Bad deployment of payment service causing checkout failures.

    Root cause: New version has a bug causing startup crash.
    Solution: Rollback to previous version.
    """
    infra.create_standard_infrastructure()

    # Payment service has bad deploy
    infra.services["payment-service"].version = "2.1.0"  # New bad version
    infra.inject_failure("payment-service", "bad_deploy")

    # Order service affected (can't process payments)
    infra.inject_failure("order-service", "dependency_timeout", {"dependency": "payment-service"})

    return {
        "root_cause": "payment-service v2.1.0 deployment has startup bug",
        "required_remediation": [
            "rollback_service:payment-service",
        ],
        "affected_services": ["payment-service", "order-service"],
    }


def create_memory_leak_scenario(infra: InfrastructureSimulator) -> dict[str, Any]:
    """
    Scenario: Slow memory leak in user-service causing gradual degradation.

    Root cause: Memory leak in user session handling.
    Solution: Restart service (temporary) and scale up to handle load.
    """
    infra.create_standard_infrastructure()
    infra.inject_failure("user-service", "memory_leak", {"memory_percent": 88})

    return {
        "root_cause": "user-service memory leak in session handling",
        "required_remediation": [
            "restart_service:user-service",
            "scale_service:user-service:5",
        ],
        "affected_services": ["user-service", "api-gateway"],
    }


def create_complex_incident_scenario(infra: InfrastructureSimulator) -> dict[str, Any]:
    """
    Scenario: Complex multi-factor incident with multiple root causes.

    This is the hard scenario combining multiple issues:
    1. Redis cache is down (network issue)
    2. This causes user-service to hit DB directly
    3. DB gets overloaded
    4. Order service starts timing out
    5. Plus there's a memory leak in product-service (unrelated)

    Solution requires identifying and fixing multiple issues.
    """
    infra.create_standard_infrastructure()

    # Redis is down
    infra.inject_failure("redis-cache", "network_partition")
    infra._add_alert(
        AlertSeverity.CRITICAL,
        "redis-cache",
        "Redis Cluster Unavailable",
        "Redis cluster has lost quorum. Cache reads failing.",
    )

    # User service hitting DB hard without cache
    infra.inject_failure("user-service", "dependency_timeout", {"dependency": "redis-cache"})
    infra.services["user-service"]._latency_base = 200.0
    infra.services["user-service"]._error_rate = 10.0

    # DB is overloaded from cache misses
    infra.inject_failure("postgres-db", "cpu_spike", {"cpu_percent": 92})

    # Order service slow due to user-service issues
    infra.services["order-service"]._status = ServiceStatus.DEGRADED
    infra.services["order-service"]._latency_base = 3000.0
    infra.services["order-service"]._error_rate = 25.0
    infra._add_alert(
        AlertSeverity.WARNING,
        "order-service",
        "High Latency",
        "Order service P99 latency exceeds 3s threshold.",
    )

    # Unrelated: product-service has memory leak
    infra.inject_failure("product-service", "memory_leak", {"memory_percent": 82})

    # API gateway showing errors
    infra.services["api-gateway"]._status = ServiceStatus.DEGRADED
    infra.services["api-gateway"]._error_rate = 30.0
    infra._add_alert(
        AlertSeverity.CRITICAL,
        "api-gateway",
        "Elevated Error Rate",
        "Multiple backend services degraded. Error rate: 30%",
    )

    return {
        "root_cause": "redis-cache network partition causing cascading DB overload, plus unrelated product-service memory leak",
        "required_remediation": [
            "restart_service:redis-cache",
            "restart_service:product-service",
            "scale_service:product-service:4",
        ],
        "affected_services": ["redis-cache", "user-service", "postgres-db", "order-service", "product-service", "api-gateway"],
    }
