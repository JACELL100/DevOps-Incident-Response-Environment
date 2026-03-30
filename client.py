"""
Client implementation for DevOps Incident Response Environment.

This module provides an EnvClient class that can connect to a
remote OpenEnv server via HTTP requests.
"""

from __future__ import annotations

from typing import Any, Optional

import requests
from pydantic import BaseModel

from models import Action, Observation, EnvironmentState


class EnvClientConfig(BaseModel):
    """Configuration for the environment client."""
    base_url: str = "http://localhost:7860"
    timeout: int = 30


class EnvClient:
    """
    Client for interacting with a remote DevOps Incident Response Environment.

    This client connects to the FastAPI server and provides a Python interface
    for the OpenEnv API endpoints.

    Example:
        ```python
        from client import EnvClient

        client = EnvClient("http://localhost:7860")
        session_id, observation = client.reset("task_easy_oom")

        while True:
            action = get_action(observation)  # Your agent logic
            observation, reward, done, info = client.step(session_id, action)
            if done:
                break

        score = client.grade(session_id)
        print(f"Final score: {score}")
        ```
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        """
        Initialize the client.

        Args:
            base_url: URL of the OpenEnv server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def reset(self, task_id: str = "task_easy_oom") -> tuple[str, Observation]:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: The task to run

        Returns:
            Tuple of (session_id, initial_observation)
        """
        response = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["session_id"], Observation(**data["observation"])

    def step(
        self,
        session_id: str,
        action: Action | str,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            session_id: The session ID from reset()
            action: The action to take (Action object or action string)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if isinstance(action, str):
            payload = {"session_id": session_id, "action_str": action}
        else:
            payload = {"session_id": session_id, "action": action.model_dump()}

        response = self._session.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return (
            Observation(**data["observation"]),
            data["reward"],
            data["done"],
            data.get("info", {}),
        )

    def state(self, session_id: str) -> EnvironmentState:
        """
        Get the current environment state.

        Args:
            session_id: The session ID

        Returns:
            Full environment state
        """
        response = self._session.get(
            f"{self.base_url}/state/{session_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return EnvironmentState(**data["state"])

    def grade(self, session_id: str) -> dict[str, Any]:
        """
        Grade the completed episode.

        Args:
            session_id: The session ID

        Returns:
            Grading result with score and breakdown
        """
        response = self._session.post(
            f"{self.base_url}/grade",
            json={"session_id": session_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_tasks(self) -> list[dict[str, Any]]:
        """
        List all available tasks.

        Returns:
            List of task information dictionaries
        """
        response = self._session.get(
            f"{self.base_url}/tasks",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["tasks"]

    def health(self) -> dict[str, Any]:
        """
        Check server health.

        Returns:
            Health status
        """
        response = self._session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the client session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
