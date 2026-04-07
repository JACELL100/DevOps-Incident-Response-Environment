"""
DevOps Incident Response Environment - Main Environment Implementation.

This module implements the OpenEnv specification for an AI agent
to learn incident response in a simulated microservices infrastructure.
"""

from __future__ import annotations

import json
import re
import sys
import os
from datetime import datetime, timedelta
from typing import Any, Optional

# Add parent directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Action,
    ActionType,
    AlertSeverity,
    EnvironmentState,
    IncidentInfo,
    IncidentTimeline,
    LogLevel,
    Observation,
    Reward,
    ServiceStatus,
    SLODefinition,
    SLOStatus,
    StepResult,
    TaskDefinition,
    TimelineEvent,
)
from server.simulator import (
    InfrastructureSimulator,
    create_bad_deploy_scenario,
    create_cascading_failure_scenario,
    create_complex_incident_scenario,
    create_memory_leak_scenario,
    create_oom_scenario,
    create_security_breach_scenario,
)
from server.tasks import get_task
from server.runbooks import get_runbook, search_runbooks, list_runbooks


class IncidentResponseEnv:
    """
    An OpenEnv-compliant environment for training AI agents
    in DevOps incident response.

    The agent must diagnose and resolve production incidents in a
    simulated microservices infrastructure by:
    - Querying service status, logs, and metrics
    - Identifying root causes
    - Executing remediation actions
    - Validating resolution
    """

    def __init__(self, task_id: str = "task_easy_oom"):
        """
        Initialize the environment.

        Args:
            task_id: The task/scenario to run
        """
        self.task_id = task_id
        self.task: TaskDefinition = get_task(task_id)
        self.infra = InfrastructureSimulator()

        # Episode state
        self.current_step = 0
        self.done = False
        self.total_reward = 0.0
        self.incident_start_time = datetime.now()

        # Tracking for grading
        self.actions_taken: list[str] = []
        self.services_queried: set[str] = set()
        self.correct_diagnosis = False
        self.remediation_progress: dict[str, bool] = {}
        self.runbooks_consulted: set[str] = set()

        # Scenario-specific ground truth
        self.scenario_data: dict[str, Any] = {}

        # Observation state
        self._observation: Optional[Observation] = None
        
        # Enhanced features
        self.timeline: IncidentTimeline = IncidentTimeline(
            incident_id=f"INC-{task_id.upper()[:8]}",
            started_at=datetime.now(),
        )
        self.escalation_level = 1
        self.slo_statuses: dict[str, SLOStatus] = {}

    def reset(self) -> Observation:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.done = False
        self.total_reward = 0.0
        self.actions_taken = []
        self.services_queried = set()
        self.correct_diagnosis = False
        self.remediation_progress = {}
        self.runbooks_consulted = set()
        self.incident_start_time = datetime.now()
        self.escalation_level = 1

        # Reset timeline
        self.timeline = IncidentTimeline(
            incident_id=f"INC-{self.task_id.upper()[:8]}",
            started_at=datetime.now(),
        )
        self._add_timeline_event("alert", "Incident detected and page sent", "system")

        # Reset infrastructure
        self.infra = InfrastructureSimulator()

        # Set up scenario based on task
        if self.task_id == "task_easy_oom":
            self.scenario_data = create_oom_scenario(self.infra)
        elif self.task_id == "task_medium_cascade":
            self.scenario_data = create_cascading_failure_scenario(self.infra)
        elif self.task_id == "task_hard_complex":
            self.scenario_data = create_complex_incident_scenario(self.infra)
        elif self.task_id == "task_expert_security":
            self.scenario_data = create_security_breach_scenario(self.infra)
        else:
            # Default fallback
            self.scenario_data = create_oom_scenario(self.infra)

        # Initialize SLO tracking (after services are set up)
        self._initialize_slo_tracking()

        # Initialize remediation tracking
        for action in self.scenario_data.get("required_remediation", []):
            self.remediation_progress[action] = False

        # Create initial observation
        self._observation = self._create_observation()
        return self._observation

    def step(self, action: Action) -> StepResult:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            StepResult with observation, reward, done, info
        """
        if self.done:
            return StepResult(
                observation=self._observation,
                reward=Reward(value=0.0, reason="Episode already done"),
                done=True,
                info={"message": "Episode has ended. Call reset() to start a new episode."},
            )

        self.current_step += 1

        # Parse action if provided as string
        if action.action_str and not action.action_type:
            action = self._parse_action_string(action.action_str)

        # Execute the action
        result = self._execute_action(action)

        # Calculate reward
        reward = self._calculate_reward(action, result)
        self.total_reward += reward.value

        # Check termination
        if self.current_step >= self.task.max_steps:
            self.done = True
            reward.reason += " Max steps reached."

        if self._check_resolution():
            self.done = True
            reward.value += 0.3  # Bonus for successful resolution
            reward.reason += " Incident resolved successfully!"

        # Update observation
        self._observation = self._create_observation(
            last_action=self._action_to_string(action),
            last_action_success=result.get("success", True),
            last_action_error=result.get("error"),
            last_action_result=result.get("data"),
        )

        # Record action
        self.actions_taken.append(self._action_to_string(action))

        return StepResult(
            observation=self._observation,
            reward=reward,
            done=self.done,
            info={
                "step": self.current_step,
                "total_reward": self.total_reward,
                "remediation_progress": self.remediation_progress,
            },
        )

    def state(self) -> EnvironmentState:
        """
        Get the full environment state for checkpointing.

        Returns:
            Complete serializable state
        """
        return EnvironmentState(
            task_id=self.task_id,
            task_difficulty=self.task.difficulty,
            episode_step=self.current_step,
            max_steps=self.task.max_steps,
            done=self.done,
            observation=self._observation or self._create_observation(),
            total_reward=self.total_reward,
            root_cause=self.scenario_data.get("root_cause", ""),
            required_actions=self.scenario_data.get("required_remediation", []),
            services_state=self.infra.get_state(),
            resolved=self._check_resolution(),
        )

    def close(self):
        """Clean up resources."""
        pass  # No cleanup needed for this simulation

    def _create_observation(
        self,
        last_action: str | None = None,
        last_action_success: bool = True,
        last_action_error: str | None = None,
        last_action_result: dict | None = None,
    ) -> Observation:
        """Create an observation from current state."""
        # Get current alerts
        alerts = self.infra.get_alerts()

        # Determine hint based on difficulty
        hint = None
        if self.task.difficulty == "easy" and self.current_step == 0:
            hint = "Start by checking the alerts and querying the affected service's logs and metrics."
        
        # Calculate incident duration
        incident_duration = int((datetime.now() - self.incident_start_time).total_seconds() / 60)
        
        # Get SLO violations
        slo_violations = self._get_slo_violations()

        return Observation(
            incident=IncidentInfo(
                id=f"INC-{self.task_id.upper()[:8]}",
                title=self.task.incident_title,
                severity=self.task.incident_severity,
                started_at=self.incident_start_time,
                description=self.task.incident_description,
                affected_services=self.task.affected_services,
                customer_impact=self.task.customer_impact,
            ),
            step=self.current_step,
            last_action=last_action,
            last_action_success=last_action_success,
            last_action_error=last_action_error,
            last_action_result=last_action_result,
            visible_alerts=alerts,
            action_history=self.actions_taken.copy(),
            available_services=self.infra.get_service_names(),
            hint=hint,
            available_runbooks=[r["id"] for r in list_runbooks()],
            slo_violations=slo_violations,
            incident_duration_minutes=incident_duration,
            escalation_level=self.escalation_level,
        )

    def _parse_action_string(self, action_str: str) -> Action:
        """Parse a natural language or formatted action string."""
        action_str = action_str.strip().lower()

        # Parse structured format: action_type:service:param
        if ":" in action_str:
            parts = action_str.split(":")
            action_type = parts[0]
            service = parts[1] if len(parts) > 1 else None
            param = parts[2] if len(parts) > 2 else None
            extra = parts[3] if len(parts) > 3 else None

            type_map = {
                "query_service": ActionType.QUERY_SERVICE,
                "query": ActionType.QUERY_SERVICE,
                "read_logs": ActionType.READ_LOGS,
                "logs": ActionType.READ_LOGS,
                "get_metrics": ActionType.GET_METRICS,
                "metrics": ActionType.GET_METRICS,
                "get_alerts": ActionType.GET_ALERTS,
                "alerts": ActionType.GET_ALERTS,
                "restart_service": ActionType.RESTART_SERVICE,
                "restart": ActionType.RESTART_SERVICE,
                "scale_service": ActionType.SCALE_SERVICE,
                "scale": ActionType.SCALE_SERVICE,
                "rollback_service": ActionType.ROLLBACK_SERVICE,
                "rollback": ActionType.ROLLBACK_SERVICE,
                "update_config": ActionType.UPDATE_CONFIG,
                "config": ActionType.UPDATE_CONFIG,
                "run_diagnostic": ActionType.RUN_DIAGNOSTIC,
                "diagnostic": ActionType.RUN_DIAGNOSTIC,
                "diagnostics": ActionType.RUN_DIAGNOSTIC,
                "resolve_incident": ActionType.RESOLVE_INCIDENT,
                "resolve": ActionType.RESOLVE_INCIDENT,
                # New enhanced actions
                "get_runbook": ActionType.GET_RUNBOOK,
                "runbook": ActionType.GET_RUNBOOK,
                "get_slo_status": ActionType.GET_SLO_STATUS,
                "slo": ActionType.GET_SLO_STATUS,
                "get_timeline": ActionType.GET_TIMELINE,
                "timeline": ActionType.GET_TIMELINE,
                "acknowledge_alert": ActionType.ACKNOWLEDGE_ALERT,
                "ack": ActionType.ACKNOWLEDGE_ALERT,
                "escalate": ActionType.ESCALATE,
            }

            if action_type in type_map:
                action = Action(action_type=type_map[action_type], service=service)

                if action_type in ["scale", "scale_service"] and param:
                    try:
                        action.scale_replicas = int(param)
                    except ValueError:
                        pass

                if action_type in ["config", "update_config"] and param:
                    action.config_key = param
                    action.config_value = extra
                
                if action_type in ["runbook", "get_runbook"] and service:
                    action.runbook_id = service
                    action.service = None
                
                if action_type in ["ack", "acknowledge_alert"] and service:
                    action.alert_id = service
                    action.service = None

                return action

        # Parse natural language
        if "runbook" in action_str:
            # Extract runbook ID from action string
            runbook_id = action_str.replace("runbook", "").replace("get_runbook", "").strip().strip(":")
            return Action(action_type=ActionType.GET_RUNBOOK, runbook_id=runbook_id if runbook_id else None)
        
        if "slo" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.GET_SLO_STATUS, service=service)
        
        if "timeline" in action_str:
            return Action(action_type=ActionType.GET_TIMELINE)
        
        if "escalate" in action_str:
            return Action(action_type=ActionType.ESCALATE)

        if "restart" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.RESTART_SERVICE, service=service)

        if "rollback" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.ROLLBACK_SERVICE, service=service)

        if "scale" in action_str:
            service = self._extract_service_name(action_str)
            replicas = self._extract_number(action_str) or 3
            return Action(action_type=ActionType.SCALE_SERVICE, service=service, scale_replicas=replicas)

        if "log" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.READ_LOGS, service=service)

        if "metric" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.GET_METRICS, service=service)

        if "alert" in action_str:
            return Action(action_type=ActionType.GET_ALERTS)

        if "diagnostic" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.RUN_DIAGNOSTIC, service=service)

        if "query" in action_str or "status" in action_str or "check" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.QUERY_SERVICE, service=service)

        if "config" in action_str:
            service = self._extract_service_name(action_str)
            return Action(action_type=ActionType.UPDATE_CONFIG, service=service)

        # Default to query if service mentioned
        service = self._extract_service_name(action_str)
        if service:
            return Action(action_type=ActionType.QUERY_SERVICE, service=service)

        # Fallback
        return Action(action_type=ActionType.GET_ALERTS)

    def _extract_service_name(self, text: str) -> str | None:
        """Extract service name from text."""
        services = self.infra.get_service_names()
        text_lower = text.lower()

        for service in services:
            if service.lower() in text_lower:
                return service

            # Handle variations
            simple_name = service.replace("-", "").replace("_", "")
            if simple_name in text_lower.replace("-", "").replace("_", ""):
                return service

        return None

    def _extract_number(self, text: str) -> int | None:
        """Extract a number from text."""
        match = re.search(r"\d+", text)
        return int(match.group()) if match else None

    def _normalize_action_type(self, action_type) -> str:
        """Normalize action type to string value."""
        if hasattr(action_type, 'value'):
            return action_type.value
        return str(action_type)

    def _execute_action(self, action: Action) -> dict[str, Any]:
        """Execute an action and return the result."""
        action_type = self._normalize_action_type(action.action_type)
        service_name = action.service

        try:
            if action_type == ActionType.QUERY_SERVICE.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                info = service.get_info()
                self._observation.visible_services[service_name] = info
                return {"success": True, "data": info.model_dump()}

            elif action_type == ActionType.READ_LOGS.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                logs = service.get_logs(action.log_lines, action.log_level_filter)
                self._observation.visible_logs.extend(logs)
                return {"success": True, "data": [log.model_dump() for log in logs]}

            elif action_type == ActionType.GET_METRICS.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                metrics = service.get_metrics()
                self._observation.visible_metrics[service_name] = metrics
                return {"success": True, "data": metrics.model_dump()}

            elif action_type == ActionType.GET_ALERTS.value:
                alerts = self.infra.get_alerts(service_name)
                return {"success": True, "data": [a.model_dump() for a in alerts]}

            elif action_type == ActionType.RESTART_SERVICE.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                success = service.restart()

                # Track remediation
                self._track_remediation(f"restart_service:{service_name}")

                return {
                    "success": success,
                    "data": {"message": f"Service {service_name} restart {'successful' if success else 'failed'}"},
                    "error": None if success else "Service failed to restart",
                }

            elif action_type == ActionType.SCALE_SERVICE.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                success = service.scale(action.scale_replicas)

                # Track remediation
                self._track_remediation(f"scale_service:{service_name}")

                return {
                    "success": success,
                    "data": {"message": f"Scaled {service_name} to {action.scale_replicas} replicas"},
                }

            elif action_type == ActionType.ROLLBACK_SERVICE.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                success = service.rollback()

                # Track remediation
                self._track_remediation(f"rollback_service:{service_name}")

                return {
                    "success": success,
                    "data": {"message": f"Rollback of {service_name} {'successful' if success else 'failed'}"},
                }

            elif action_type == ActionType.UPDATE_CONFIG.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                if not action.config_key:
                    return {"success": False, "error": "No config key provided"}

                service = self.infra.services[service_name]
                success = service.update_config(action.config_key, action.config_value)

                # Track remediation with key
                self._track_remediation(f"update_config:{service_name}:{action.config_key}")

                return {
                    "success": success,
                    "data": {"message": f"Config {action.config_key} updated on {service_name}"},
                }

            elif action_type == ActionType.RUN_DIAGNOSTIC.value:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                result = service.run_diagnostic()
                self._observation.visible_diagnostics[service_name] = result
                return {"success": True, "data": result.model_dump()}

            elif action_type == ActionType.RESOLVE_INCIDENT.value:
                # Check if actually resolved
                if self._check_resolution():
                    self._add_timeline_event("status_change", "Incident marked as resolved", "agent")
                    return {"success": True, "data": {"message": "Incident resolved successfully"}}
                else:
                    return {
                        "success": False,
                        "error": "Incident not yet resolved. Some services still unhealthy.",
                        "data": {"unhealthy_services": self._get_unhealthy_services()},
                    }

            # =========================================================
            # New Enhanced Actions
            # =========================================================
            
            elif action_type == ActionType.GET_RUNBOOK.value:
                # Get a specific runbook or search for runbooks
                if action.runbook_id:
                    runbook = get_runbook(action.runbook_id)
                    if runbook:
                        self.runbooks_consulted.add(action.runbook_id)
                        self._add_timeline_event("action", f"Consulted runbook: {runbook.title}", "agent")
                        return {"success": True, "data": runbook.model_dump()}
                    return {"success": False, "error": f"Runbook not found: {action.runbook_id}"}
                elif action.search_query:
                    results = search_runbooks(action.search_query)
                    return {"success": True, "data": [r.model_dump() for r in results]}
                else:
                    # List all runbooks
                    return {"success": True, "data": list_runbooks()}

            elif action_type == ActionType.GET_SLO_STATUS.value:
                # Get SLO status
                slo_data = self.get_slo_status(action.service)
                return {"success": True, "data": slo_data}

            elif action_type == ActionType.GET_TIMELINE.value:
                # Get incident timeline
                return {"success": True, "data": self.timeline.model_dump()}

            elif action_type == ActionType.ACKNOWLEDGE_ALERT.value:
                # Acknowledge an alert
                if not action.alert_id:
                    return {"success": False, "error": "No alert_id provided"}
                self.infra.resolve_alert(action.alert_id)
                self._add_timeline_event("action", f"Alert acknowledged: {action.alert_id}", "agent")
                return {"success": True, "data": {"message": f"Alert {action.alert_id} acknowledged"}}

            elif action_type == ActionType.ESCALATE.value:
                # Escalate incident
                result = self.escalate_incident(action.escalation_reason)
                return result

            else:
                return {"success": False, "error": f"Unknown action type: {action_type}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _track_remediation(self, action: str):
        """Track remediation action for partial credit."""
        # Check against required actions
        for required in self.remediation_progress:
            # Flexible matching
            if action.startswith(required.split(":")[0]):
                if required.split(":")[1] in action:
                    self.remediation_progress[required] = True
                    break

    def _calculate_reward(self, action: Action, result: dict[str, Any]) -> Reward:
        """Calculate reward for an action."""
        reward_value = 0.0
        diagnosis_progress = 0.0
        remediation_progress = 0.0
        efficiency_bonus = 0.0
        unnecessary_penalty = 0.0
        harmful_penalty = 0.0
        time_penalty = 0.0
        reason = ""

        action_type = self._normalize_action_type(action.action_type)
        success = result.get("success", False)

        # Diagnostic actions (querying info)
        if action_type in [
            ActionType.QUERY_SERVICE.value,
            ActionType.READ_LOGS.value,
            ActionType.GET_METRICS.value,
            ActionType.GET_ALERTS.value,
            ActionType.RUN_DIAGNOSTIC.value,
        ]:
            if success:
                # Reward for investigating affected services
                if action.service in self.scenario_data.get("affected_services", []):
                    reward_value += 0.02
                    diagnosis_progress += 0.1
                    reason = f"Investigated affected service: {action.service}"
                else:
                    # Small reward for any investigation
                    reward_value += 0.005
                    diagnosis_progress += 0.02
                    reason = f"Investigated service: {action.service}"

                # Bonus for comprehensive investigation
                affected = set(self.scenario_data.get("affected_services", []))
                if affected and affected.issubset(self.services_queried):
                    diagnosis_progress += 0.2
                    reward_value += 0.05
                    reason += " All affected services investigated."

        # Remediation actions
        elif action_type in [
            ActionType.RESTART_SERVICE.value,
            ActionType.SCALE_SERVICE.value,
            ActionType.ROLLBACK_SERVICE.value,
            ActionType.UPDATE_CONFIG.value,
        ]:
            if success:
                # Check if this is a required remediation
                action_str = self._action_to_string(action)
                is_required = any(
                    action_str.startswith(req.split(":")[0]) and req.split(":")[1] in action_str
                    for req in self.scenario_data.get("required_remediation", [])
                )

                if is_required:
                    reward_value += 0.1
                    remediation_progress += 0.25
                    reason = f"Correct remediation action: {action_str}"
                else:
                    # Small penalty for unnecessary actions
                    reward_value -= 0.01
                    unnecessary_penalty = 0.01
                    reason = f"Unnecessary action: {action_str}"
            else:
                reward_value -= 0.02
                reason = f"Action failed: {result.get('error', 'unknown error')}"

        # Time penalty (encourage efficiency)
        if self.current_step > self.task.max_steps * 0.8:
            time_penalty = 0.01 * (self.current_step - self.task.max_steps * 0.8)
            reward_value -= time_penalty
            reason += f" Time pressure penalty: {time_penalty:.3f}"

        # Efficiency bonus for quick resolution
        if self._check_resolution():
            efficiency_factor = 1.0 - (self.current_step / self.task.max_steps)
            efficiency_bonus = efficiency_factor * 0.1
            reward_value += efficiency_bonus
            reason += f" Efficiency bonus: {efficiency_bonus:.3f}"

        return Reward(
            value=round(reward_value, 4),
            diagnosis_progress=diagnosis_progress,
            remediation_progress=remediation_progress,
            efficiency_bonus=efficiency_bonus,
            unnecessary_action_penalty=unnecessary_penalty,
            harmful_action_penalty=harmful_penalty,
            time_penalty=time_penalty,
            reason=reason,
        )

    def _check_resolution(self) -> bool:
        """Check if the incident is resolved."""
        # All affected services must be healthy (not just degraded)
        affected = self.scenario_data.get("affected_services", [])
        for service_name in affected:
            if service_name in self.infra.services:
                service = self.infra.services[service_name]
                if service.get_status() != ServiceStatus.HEALTHY:
                    return False

        return True

    def _get_unhealthy_services(self) -> list[str]:
        """Get list of unhealthy services."""
        unhealthy = []
        for name, service in self.infra.services.items():
            if service.get_status() not in [ServiceStatus.HEALTHY]:
                unhealthy.append(name)
        return unhealthy

    def _action_to_string(self, action: Action) -> str:
        """Convert action to string representation."""
        # Handle both enum and string values for action_type
        action_type = action.action_type
        if hasattr(action_type, 'value'):
            action_type_str = action_type.value
        else:
            action_type_str = str(action_type)

        parts = [action_type_str]
        if action.service:
            parts.append(action.service)
        if action.config_key:
            parts.append(action.config_key)
        if action.config_value:
            parts.append(str(action.config_value))
        if action.scale_replicas and action_type_str == ActionType.SCALE_SERVICE.value:
            parts.append(str(action.scale_replicas))
        return ":".join(parts)

    # =========================================================================
    # Enhanced Feature Methods
    # =========================================================================

    def _initialize_slo_tracking(self):
        """Initialize SLO tracking for affected services."""
        self.slo_statuses = {}
        
        # Create SLO status for each service
        for service_name in self.infra.get_service_names():
            # Simulate SLO degradation based on service health
            service = self.infra.services.get(service_name)
            if not service:
                continue
                
            metrics = service.get_metrics()
            
            # Availability SLO (based on error rate)
            availability_current = 100.0 - metrics.error_rate
            availability_target = 99.9
            availability_breached = availability_current < availability_target
            
            # Latency SLO
            latency_target = 200.0  # ms
            latency_current = metrics.latency_p99_ms
            latency_breached = latency_current > latency_target
            
            # Error rate SLO
            error_target = 1.0  # %
            error_current = metrics.error_rate
            error_breached = error_current > error_target
            
            self.slo_statuses[service_name] = SLOStatus(
                service=service_name,
                availability_slo=SLODefinition(
                    name="availability",
                    target=availability_target,
                    current=availability_current,
                    error_budget_remaining=max(0, (availability_current - availability_target) / (100 - availability_target) * 100),
                    breached=availability_breached,
                ),
                latency_p99_slo=SLODefinition(
                    name="latency_p99",
                    target=latency_target,
                    current=latency_current,
                    error_budget_remaining=max(0, (latency_target - latency_current) / latency_target * 100) if latency_current < latency_target else 0,
                    breached=latency_breached,
                ),
                error_rate_slo=SLODefinition(
                    name="error_rate",
                    target=error_target,
                    current=error_current,
                    error_budget_remaining=max(0, (error_target - error_current) / error_target * 100) if error_current < error_target else 0,
                    breached=error_breached,
                ),
            )

    def _get_slo_violations(self) -> list[str]:
        """Get list of services with SLO violations."""
        violations = []
        for service_name, slo_status in self.slo_statuses.items():
            if (slo_status.availability_slo.breached or 
                slo_status.latency_p99_slo.breached or 
                slo_status.error_rate_slo.breached):
                violations.append(service_name)
        return violations

    def _add_timeline_event(self, event_type: str, description: str, actor: str = "agent", metadata: dict = None):
        """Add an event to the incident timeline."""
        self.timeline.events.append(TimelineEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            actor=actor,
            metadata=metadata or {},
        ))

    def get_timeline(self) -> IncidentTimeline:
        """Get the incident timeline."""
        return self.timeline

    def get_slo_status(self, service_name: str = None) -> dict:
        """Get SLO status for a service or all services."""
        if service_name and service_name in self.slo_statuses:
            return self.slo_statuses[service_name].model_dump()
        return {name: status.model_dump() for name, status in self.slo_statuses.items()}

    def escalate_incident(self, reason: str = None) -> dict:
        """Escalate the incident to the next level."""
        if self.escalation_level >= 3:
            return {"success": False, "error": "Already at maximum escalation level"}
        
        self.escalation_level += 1
        self._add_timeline_event(
            "escalation",
            f"Incident escalated to level {self.escalation_level}: {reason or 'Manual escalation'}",
            "agent",
        )
        
        return {
            "success": True,
            "data": {
                "new_level": self.escalation_level,
                "message": f"Incident escalated to level {self.escalation_level}",
            }
        }

