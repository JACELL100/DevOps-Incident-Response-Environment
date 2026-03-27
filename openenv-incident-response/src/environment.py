"""
DevOps Incident Response Environment - Main Environment Implementation.

This module implements the OpenEnv specification for an AI agent
to learn incident response in a simulated microservices infrastructure.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

from .models import (
    Action,
    ActionType,
    AlertSeverity,
    EnvironmentState,
    IncidentInfo,
    LogLevel,
    Observation,
    Reward,
    ServiceStatus,
    StepResult,
    TaskDefinition,
)
from .simulator import (
    InfrastructureSimulator,
    create_bad_deploy_scenario,
    create_cascading_failure_scenario,
    create_complex_incident_scenario,
    create_memory_leak_scenario,
    create_oom_scenario,
)
from .tasks import get_task


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

        # Tracking for grading
        self.actions_taken: list[str] = []
        self.services_queried: set[str] = set()
        self.correct_diagnosis = False
        self.remediation_progress: dict[str, bool] = {}

        # Scenario-specific ground truth
        self.scenario_data: dict[str, Any] = {}

        # Observation state
        self._observation: Optional[Observation] = None

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

        # Reset infrastructure
        self.infra = InfrastructureSimulator()

        # Set up scenario based on task
        if self.task_id == "task_easy_oom":
            self.scenario_data = create_oom_scenario(self.infra)
        elif self.task_id == "task_medium_cascade":
            self.scenario_data = create_cascading_failure_scenario(self.infra)
        elif self.task_id == "task_hard_complex":
            self.scenario_data = create_complex_incident_scenario(self.infra)
        else:
            # Default fallback
            self.scenario_data = create_oom_scenario(self.infra)

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

        return Observation(
            incident=IncidentInfo(
                id=f"INC-{self.task_id.upper()[:8]}",
                title=self.task.incident_title,
                severity=self.task.incident_severity,
                started_at=datetime.now(),
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

                return action

        # Parse natural language
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

    def _execute_action(self, action: Action) -> dict[str, Any]:
        """Execute an action and return the result."""
        action_type = action.action_type
        service_name = action.service

        try:
            if action_type == ActionType.QUERY_SERVICE:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                info = service.get_info()
                self._observation.visible_services[service_name] = info
                return {"success": True, "data": info.model_dump()}

            elif action_type == ActionType.READ_LOGS:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                logs = service.get_logs(action.log_lines, action.log_level_filter)
                self._observation.visible_logs.extend(logs)
                return {"success": True, "data": [log.model_dump() for log in logs]}

            elif action_type == ActionType.GET_METRICS:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                metrics = service.get_metrics()
                self._observation.visible_metrics[service_name] = metrics
                return {"success": True, "data": metrics.model_dump()}

            elif action_type == ActionType.GET_ALERTS:
                alerts = self.infra.get_alerts(service_name)
                return {"success": True, "data": [a.model_dump() for a in alerts]}

            elif action_type == ActionType.RESTART_SERVICE:
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

            elif action_type == ActionType.SCALE_SERVICE:
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

            elif action_type == ActionType.ROLLBACK_SERVICE:
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

            elif action_type == ActionType.UPDATE_CONFIG:
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

            elif action_type == ActionType.RUN_DIAGNOSTIC:
                if not service_name or service_name not in self.infra.services:
                    return {"success": False, "error": f"Unknown service: {service_name}"}

                service = self.infra.services[service_name]
                self.services_queried.add(service_name)
                result = service.run_diagnostic()
                self._observation.visible_diagnostics[service_name] = result
                return {"success": True, "data": result.model_dump()}

            elif action_type == ActionType.RESOLVE_INCIDENT:
                # Check if actually resolved
                if self._check_resolution():
                    return {"success": True, "data": {"message": "Incident resolved successfully"}}
                else:
                    return {
                        "success": False,
                        "error": "Incident not yet resolved. Some services still unhealthy.",
                        "data": {"unhealthy_services": self._get_unhealthy_services()},
                    }

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

        action_type = action.action_type
        success = result.get("success", False)

        # Diagnostic actions (querying info)
        if action_type in [
            ActionType.QUERY_SERVICE,
            ActionType.READ_LOGS,
            ActionType.GET_METRICS,
            ActionType.GET_ALERTS,
            ActionType.RUN_DIAGNOSTIC,
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
            ActionType.RESTART_SERVICE,
            ActionType.SCALE_SERVICE,
            ActionType.ROLLBACK_SERVICE,
            ActionType.UPDATE_CONFIG,
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
        # All affected services must be healthy
        affected = self.scenario_data.get("affected_services", [])
        for service_name in affected:
            if service_name in self.infra.services:
                service = self.infra.services[service_name]
                if service.get_status() not in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
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
        parts = [action.action_type.value]
        if action.service:
            parts.append(action.service)
        if action.config_key:
            parts.append(action.config_key)
        if action.config_value:
            parts.append(str(action.config_value))
        if action.scale_replicas and action.action_type == ActionType.SCALE_SERVICE:
            parts.append(str(action.scale_replicas))
        return ":".join(parts)
