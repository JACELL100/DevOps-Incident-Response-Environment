"""
Task Graders for the DevOps Incident Response Environment.

Each grader evaluates agent performance on a specific task,
producing a score from 0.0 to 1.0 based on:
- Diagnosis accuracy (did the agent identify the root cause?)
- Remediation completeness (did the agent fix the issue?)
- Efficiency (how quickly was the issue resolved?)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .environment import IncidentResponseEnv
from .models import EnvironmentState, ServiceStatus
from .tasks import TASKS, TaskDefinition


@dataclass
class GradeResult:
    """Result of grading an episode."""

    score: float  # 0.0 to 1.0
    diagnosis_score: float
    remediation_score: float
    efficiency_score: float
    details: dict[str, Any]
    feedback: str


class TaskGrader:
    """
    Grades agent performance on incident response tasks.

    Scoring is deterministic and based on:
    1. Whether the agent investigated affected services
    2. Whether the agent took correct remediation actions
    3. Whether the incident was resolved
    4. How efficiently the agent resolved the incident
    """

    def __init__(self, task: TaskDefinition):
        self.task = task

    def grade(self, env: IncidentResponseEnv) -> GradeResult:
        """
        Grade the agent's performance on a completed episode.

        Args:
            env: The environment after the episode

        Returns:
            GradeResult with score and breakdown
        """
        state = env.state()

        # Calculate component scores
        diagnosis_score = self._score_diagnosis(env, state)
        remediation_score = self._score_remediation(env, state)
        efficiency_score = self._score_efficiency(env, state)

        # Weighted combination
        final_score = (
            diagnosis_score * self.task.diagnosis_weight
            + remediation_score * self.task.remediation_weight
            + efficiency_score * self.task.efficiency_weight
        )

        # Clamp to [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))

        # Generate feedback
        feedback = self._generate_feedback(
            diagnosis_score, remediation_score, efficiency_score, state
        )

        return GradeResult(
            score=round(final_score, 4),
            diagnosis_score=round(diagnosis_score, 4),
            remediation_score=round(remediation_score, 4),
            efficiency_score=round(efficiency_score, 4),
            details={
                "services_queried": list(env.services_queried),
                "affected_services": env.scenario_data.get("affected_services", []),
                "actions_taken": env.actions_taken,
                "required_remediation": env.scenario_data.get("required_remediation", []),
                "remediation_progress": env.remediation_progress,
                "resolved": state.resolved,
                "steps_used": state.episode_step,
                "max_steps": state.max_steps,
            },
            feedback=feedback,
        )

    def _score_diagnosis(self, env: IncidentResponseEnv, state: EnvironmentState) -> float:
        """
        Score the agent's diagnostic investigation.

        Full credit for investigating all affected services.
        Partial credit proportional to coverage.
        """
        affected = set(env.scenario_data.get("affected_services", []))
        queried = env.services_queried

        if not affected:
            return 1.0  # No affected services = automatic pass

        # Calculate coverage
        covered = affected.intersection(queried)
        coverage = len(covered) / len(affected)

        # Bonus for using diagnostic tools
        diagnostic_bonus = 0.0
        diagnostic_count = sum(
            1 for a in env.actions_taken
            if "diagnostic" in a.lower() or "logs" in a.lower() or "metrics" in a.lower()
        )
        if diagnostic_count > 0:
            diagnostic_bonus = min(0.1, diagnostic_count * 0.02)

        return min(1.0, coverage + diagnostic_bonus)

    def _score_remediation(self, env: IncidentResponseEnv, state: EnvironmentState) -> float:
        """
        Score the agent's remediation actions.

        Full credit for completing all required remediation actions.
        Partial credit for partial completion.
        Major bonus for actually resolving the incident.
        """
        required = env.scenario_data.get("required_remediation", [])
        progress = env.remediation_progress

        if not required:
            return 1.0 if state.resolved else 0.5

        # Count completed remediation actions
        completed = sum(1 for r in progress.values() if r)
        completion_rate = completed / len(required)

        # Resolution bonus - this is the main goal
        resolution_bonus = 0.4 if state.resolved else 0.0

        # Penalty for unnecessary actions
        unnecessary_actions = len(env.actions_taken) - len(
            [a for a in env.actions_taken if any(req.split(":")[0] in a for req in required)]
        )
        unnecessary_penalty = min(0.2, unnecessary_actions * 0.02)

        score = completion_rate * 0.6 + resolution_bonus - unnecessary_penalty
        return max(0.0, min(1.0, score))

    def _score_efficiency(self, env: IncidentResponseEnv, state: EnvironmentState) -> float:
        """
        Score the agent's efficiency.

        Based on how quickly the incident was resolved relative to max steps.
        """
        steps_used = state.episode_step
        max_steps = state.max_steps

        if not state.resolved:
            # No efficiency bonus if not resolved
            return 0.0

        # Linear scaling: fewer steps = higher score
        # Perfect efficiency at 30% of max steps
        # Zero efficiency at max steps
        optimal = max_steps * 0.3
        if steps_used <= optimal:
            return 1.0
        elif steps_used >= max_steps:
            return 0.0
        else:
            return 1.0 - (steps_used - optimal) / (max_steps - optimal)

    def _generate_feedback(
        self,
        diagnosis_score: float,
        remediation_score: float,
        efficiency_score: float,
        state: EnvironmentState,
    ) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []

        # Diagnosis feedback
        if diagnosis_score >= 0.8:
            feedback_parts.append("Excellent investigation - all affected services were analyzed.")
        elif diagnosis_score >= 0.5:
            feedback_parts.append("Partial investigation - consider checking all affected services.")
        else:
            feedback_parts.append("Limited investigation - more diagnostic work needed.")

        # Remediation feedback
        if remediation_score >= 0.8:
            feedback_parts.append("Strong remediation - correct actions taken.")
        elif remediation_score >= 0.5:
            feedback_parts.append("Partial remediation - some required fixes were missed.")
        else:
            feedback_parts.append("Incomplete remediation - review the required actions.")

        # Efficiency feedback
        if state.resolved:
            if efficiency_score >= 0.8:
                feedback_parts.append("Very efficient resolution!")
            elif efficiency_score >= 0.5:
                feedback_parts.append("Resolved in reasonable time.")
            else:
                feedback_parts.append("Resolved but could be faster.")
        else:
            feedback_parts.append("Incident was not fully resolved.")

        return " ".join(feedback_parts)


def grade_task(task_id: str, env: IncidentResponseEnv) -> GradeResult:
    """
    Grade a specific task.

    Args:
        task_id: The task identifier
        env: The environment after the episode

    Returns:
        GradeResult with score and details
    """
    task = TASKS.get(task_id)
    if not task:
        raise ValueError(f"Unknown task: {task_id}")

    grader = TaskGrader(task)
    return grader.grade(env)


def grade_all_tasks(results: dict[str, IncidentResponseEnv]) -> dict[str, GradeResult]:
    """
    Grade all tasks from a set of completed environments.

    Args:
        results: Dictionary mapping task_id to completed environment

    Returns:
        Dictionary mapping task_id to GradeResult
    """
    grades = {}
    for task_id, env in results.items():
        if task_id in TASKS:
            grades[task_id] = grade_task(task_id, env)
    return grades


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_grader(task_id: str) -> bool:
    """
    Validate that a grader produces scores in the correct range.

    This is used by the OpenEnv validator.
    """
    task = TASKS.get(task_id)
    if not task:
        return False

    grader = TaskGrader(task)

    # Test with a fresh environment
    env = IncidentResponseEnv(task_id=task_id)
    env.reset()

    # Grade with no actions (should be low but valid)
    result = grader.grade(env)

    # Validate score range
    if not (0.0 <= result.score <= 1.0):
        return False
    if not (0.0 <= result.diagnosis_score <= 1.0):
        return False
    if not (0.0 <= result.remediation_score <= 1.0):
        return False
    if not (0.0 <= result.efficiency_score <= 1.0):
        return False

    return True


def validate_all_graders() -> dict[str, bool]:
    """Validate all task graders."""
    return {task_id: validate_grader(task_id) for task_id in TASKS}
