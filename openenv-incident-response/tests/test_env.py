"""
Tests for the DevOps Incident Response Environment.
"""

import pytest
from fastapi.testclient import TestClient

from src import (
    Action,
    ActionType,
    IncidentResponseEnv,
    grade_task,
    list_tasks,
)
from app import app


# ============================================================================
# Environment Tests
# ============================================================================

class TestEnvironment:
    """Test the core environment functionality."""

    def test_list_tasks(self):
        """Test that tasks are properly listed."""
        tasks = list_tasks()
        assert len(tasks) == 3
        assert tasks[0]["id"] == "task_easy_oom"
        assert tasks[1]["id"] == "task_medium_cascade"
        assert tasks[2]["id"] == "task_hard_complex"

    def test_reset(self):
        """Test environment reset."""
        env = IncidentResponseEnv("task_easy_oom")
        obs = env.reset()

        assert obs.step == 0
        assert obs.incident is not None
        assert obs.incident.title is not None
        assert len(obs.available_services) > 0

    def test_step_query(self):
        """Test query action."""
        env = IncidentResponseEnv("task_easy_oom")
        env.reset()

        action = Action(action_type=ActionType.QUERY_SERVICE, service="order-service")
        result = env.step(action)

        assert result.observation is not None
        assert result.reward is not None
        assert result.done is False

    def test_step_get_alerts(self):
        """Test get alerts action."""
        env = IncidentResponseEnv("task_easy_oom")
        env.reset()

        action = Action(action_type=ActionType.GET_ALERTS)
        result = env.step(action)

        assert result.reward.value >= 0

    def test_action_string_parsing(self):
        """Test action string parsing."""
        env = IncidentResponseEnv("task_easy_oom")
        env.reset()

        # Test with action_str
        action = Action(action_str="get_alerts")
        result = env.step(action)
        assert result.observation is not None

        action = Action(action_str="query_service:order-service")
        result = env.step(action)
        assert "order-service" in env.services_queried

    def test_grading(self):
        """Test episode grading."""
        env = IncidentResponseEnv("task_easy_oom")
        env.reset()

        # Take some actions
        env.step(Action(action_type=ActionType.GET_ALERTS))
        env.step(Action(action_type=ActionType.QUERY_SERVICE, service="order-service"))

        # Grade
        grade = grade_task("task_easy_oom", env)

        assert 0.0 <= grade.score <= 1.0
        assert 0.0 <= grade.diagnosis_score <= 1.0
        assert 0.0 <= grade.remediation_score <= 1.0
        assert grade.feedback is not None


# ============================================================================
# API Tests
# ============================================================================

class TestAPI:
    """Test the FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_tasks(self, client):
        """Test tasks endpoint."""
        response = client.get("/tasks")
        assert response.status_code == 200
        assert len(response.json()["tasks"]) == 3

    def test_reset(self, client):
        """Test reset endpoint."""
        response = client.post("/reset", json={"task_id": "task_easy_oom"})
        assert response.status_code == 200
        assert "session_id" in response.json()
        assert "observation" in response.json()

    def test_step(self, client):
        """Test step endpoint."""
        # First reset
        reset_response = client.post("/reset", json={"task_id": "task_easy_oom"})
        session_id = reset_response.json()["session_id"]

        # Then step
        step_response = client.post(
            "/step",
            json={"session_id": session_id, "action_str": "get_alerts"},
        )
        assert step_response.status_code == 200
        assert "observation" in step_response.json()
        assert "reward" in step_response.json()

    def test_grade(self, client):
        """Test grade endpoint."""
        # Reset
        reset_response = client.post("/reset", json={"task_id": "task_easy_oom"})
        session_id = reset_response.json()["session_id"]

        # Take some actions
        client.post("/step", json={"session_id": session_id, "action_str": "get_alerts"})

        # Grade
        grade_response = client.post("/grade", json={"session_id": session_id})
        assert grade_response.status_code == 200
        assert 0.0 <= grade_response.json()["score"] <= 1.0

    def test_state(self, client):
        """Test state endpoint."""
        # Reset
        reset_response = client.post("/reset", json={"task_id": "task_easy_oom"})
        session_id = reset_response.json()["session_id"]

        # Get state
        state_response = client.get(f"/state/{session_id}")
        assert state_response.status_code == 200
        assert "state" in state_response.json()


# ============================================================================
# Grader Validation Tests
# ============================================================================

class TestGraders:
    """Test grader validity."""

    def test_grader_score_range(self):
        """Verify graders produce scores in valid range."""
        for task_id in ["task_easy_oom", "task_medium_cascade", "task_hard_complex"]:
            env = IncidentResponseEnv(task_id)
            env.reset()

            # No actions - should still produce valid score
            grade = grade_task(task_id, env)
            assert 0.0 <= grade.score <= 1.0
            assert 0.0 <= grade.diagnosis_score <= 1.0
            assert 0.0 <= grade.remediation_score <= 1.0
            assert 0.0 <= grade.efficiency_score <= 1.0

    def test_grader_determinism(self):
        """Verify graders are deterministic."""
        env = IncidentResponseEnv("task_easy_oom")
        env.reset()

        # Take same actions
        env.step(Action(action_type=ActionType.GET_ALERTS))
        env.step(Action(action_type=ActionType.QUERY_SERVICE, service="order-service"))

        # Grade multiple times
        grade1 = grade_task("task_easy_oom", env)
        grade2 = grade_task("task_easy_oom", env)

        assert grade1.score == grade2.score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
