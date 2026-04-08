#!/usr/bin/env python3
"""
Baseline Inference Script for DevOps Incident Response Environment.

This script runs an AI agent against the environment deployed on Hugging Face Spaces
and reports performance scores using the required structured logging format.

Environment Variables (REQUIRED):
    API_BASE_URL: The API endpoint for the LLM
    MODEL_NAME: The model identifier to use
    HF_TOKEN: Hugging Face / API key

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Environment server URL (Hugging Face Space or local)
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Benchmark/Environment name
BENCHMARK = "devops-incident-response"

# Task to run (can be overridden via environment)
TASK_NAME = os.environ.get("TASK_NAME", "task_easy_oom")

# Inference parameters
MAX_STEPS = 15  # Will be updated per task
MAX_TOKENS = 1024
TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 2.0
SUCCESS_SCORE_THRESHOLD = 0.6


# ============================================================================
# Structured Logging Functions (REQUIRED FORMAT)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log the start of an episode in required format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Log each step in required format."""
    error_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action} reward={reward} done={done}{error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log the end of an episode in required format."""
    rewards_str = json.dumps(rewards)
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards_str}", flush=True)


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

Your job is to diagnose the root cause of the incident and take appropriate remediation actions.

## Available Actions

You can take the following actions (use the exact format shown):

1. **Get Alerts**: `get_alerts`
   - Get all active monitoring alerts (ALWAYS START HERE)

2. **Query Service Status**: `query_service:<service_name>`
   - Get current status, replicas, version, and dependencies

3. **Read Logs**: `read_logs:<service_name>`
   - Get recent log entries from the service

4. **Get Metrics**: `get_metrics:<service_name>`
   - Get CPU, memory, latency, and error rate metrics

5. **Run Diagnostics**: `run_diagnostic:<service_name>`
   - Run diagnostic checks on a service

6. **Restart Service**: `restart_service:<service_name>`
   - Restart a service (may fix some issues)

7. **Scale Service**: `scale_service:<service_name>:<replicas>`
   - Scale a service to specified replica count

8. **Rollback Service**: `rollback_service:<service_name>`
   - Rollback to the previous deployment

9. **Update Config**: `update_config:<service_name>:<key>:<value>`
   - Update a configuration value (requires restart)

10. **Resolve Incident**: `resolve_incident`
    - Mark the incident as resolved (only when fully fixed)

## Response Format

Respond with your reasoning followed by your action in this format:

```
THOUGHT: [Your analysis of the current situation]
ACTION: [action_string]
```

Example:
```
THOUGHT: I need to first check what alerts are firing to understand the scope of the incident.
ACTION: get_alerts
```

## Critical Guidelines

1. **ALWAYS start with get_alerts** - understand what's broken before investigating
2. Read logs and metrics for affected services to identify root cause
3. Identify the root cause before attempting fixes
4. Configuration changes require a service restart to take effect
5. Only scale or rollback if specifically indicated by the diagnostics
6. Verify your fix worked (check metrics/logs again) before calling resolve_incident
7. Work methodically - don't spam random actions

## Common Patterns

- **OOM errors**: Increase heap size with update_config, then restart
- **Connection timeouts**: Check connection pool settings, may need to increase max_connections
- **High latency**: Check dependent services, look for cascading failures
- **Cache failures**: May need to restart cache service or check network partitions

Remember: You're dealing with a production system. Be methodical and verify your fixes."""


# ============================================================================
# Environment Client (HTTP-based)
# ============================================================================

class EnvClient:
    """HTTP client for interacting with the environment server."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.session_id: Optional[str] = None

    def reset(self, task_id: str) -> dict:
        """Reset the environment and start a new episode."""
        response = self.client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
        )
        response.raise_for_status()
        data = response.json()
        self.session_id = data["session_id"]
        return data["observation"]

    def step(self, action_str: str) -> dict:
        """Take an action in the environment."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before step()")

        response = self.client.post(
            f"{self.base_url}/step",
            json={"session_id": self.session_id, "action_str": action_str}
        )
        response.raise_for_status()
        return response.json()

    def grade(self) -> dict:
        """Get the final grade for the episode."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before grade()")

        response = self.client.post(
            f"{self.base_url}/grade",
            json={"session_id": self.session_id}
        )
        response.raise_for_status()
        return response.json()

    def get_tasks(self) -> list:
        """Get list of available tasks."""
        response = self.client.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()["tasks"]

    def health(self) -> bool:
        """Check if the environment is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


# ============================================================================
# Action Parser
# ============================================================================

def parse_model_response(response: str) -> str:
    """Extract action from model response."""
    # Look for ACTION: pattern
    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()
        # Clean up any backticks or quotes
        action = action.strip("`'\"")
        return action

    # Look for action-like patterns
    action_patterns = [
        r"(query_service:\S+)",
        r"(read_logs:\S+)",
        r"(get_metrics:\S+)",
        r"(get_alerts)",
        r"(restart_service:\S+)",
        r"(scale_service:\S+:\d+)",
        r"(rollback_service:\S+)",
        r"(update_config:\S+:\S+:\S+)",
        r"(run_diagnostic:\S+)",
        r"(resolve_incident)",
    ]

    for pattern in action_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)

    # Default fallback
    return "get_alerts"


# ============================================================================
# LLM Agent
# ============================================================================

def get_model_message(
    client: OpenAI,
    step: int,
    observation: dict,
    last_reward: float,
    history: List[str]
) -> str:
    """Get an action from the LLM based on current observation."""
    # Format observation for the model
    obs_text = format_observation_dict(observation)

    # Build conversation
    user_content = f"Step {step}:\n{obs_text}"
    if last_reward != 0:
        user_content += f"\n\nLast reward: {last_reward:+.2f}"
    if history:
        user_content += f"\n\nRecent history:\n" + "\n".join(history[-5:])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = parse_model_response(text) if text else "get_alerts"
        return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "get_alerts"


def format_observation_dict(obs: dict) -> str:
    """Format observation dictionary as readable text for the model."""
    lines = []

    # Incident info
    incident = obs.get("incident", {})
    lines.append("=" * 60)
    lines.append(f"INCIDENT: {incident.get('title', 'Unknown')}")
    lines.append(f"Severity: {incident.get('severity', 'unknown')}")
    lines.append(f"Description: {incident.get('description', '')}")
    affected = incident.get('affected_services', [])
    lines.append(f"Affected Services: {', '.join(affected)}")
    lines.append(f"Customer Impact: {incident.get('customer_impact', '')}")
    lines.append("=" * 60)

    # Step info
    lines.append(f"\nStep: {obs.get('step', 0)}")

    # Available services
    services = obs.get('available_services', [])
    lines.append(f"\nAvailable services: {', '.join(services)}")

    # Last action result
    last_action = obs.get('last_action')
    if last_action:
        lines.append(f"\nLast action: {last_action}")
        if obs.get('last_action_success', True):
            lines.append("Result: SUCCESS")
        else:
            lines.append(f"Result: FAILED - {obs.get('last_action_error', '')}")

        result = obs.get('last_action_result')
        if result:
            result_str = json.dumps(result, indent=2, default=str)
            lines.append(f"Data:\n{result_str}")

    # Active alerts
    alerts = obs.get('visible_alerts', [])
    if alerts:
        lines.append("\n--- ACTIVE ALERTS ---")
        for alert in alerts:
            lines.append(f"[{alert.get('severity', 'unknown')}] {alert.get('title', '')}")
            lines.append(f"  Service: {alert.get('service', '')}")
            lines.append(f"  {alert.get('description', '')}")

    # Hint
    hint = obs.get('hint')
    if hint:
        lines.append(f"\nHINT: {hint}")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run inference on the environment."""
    # Validate configuration
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is required", flush=True)
        sys.exit(1)
    if not API_BASE_URL:
        print("[ERROR] API_BASE_URL environment variable is required", flush=True)
        sys.exit(1)
    if not MODEL_NAME:
        print("[ERROR] MODEL_NAME environment variable is required", flush=True)
        sys.exit(1)

    # Initialize clients
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env_client = EnvClient(ENV_URL)

    # Check environment health
    if not env_client.health():
        print(f"[ERROR] Environment at {ENV_URL} is not healthy", flush=True)
        sys.exit(1)

    # Get tasks and find max_steps for our task
    tasks = env_client.get_tasks()
    task_info = next((t for t in tasks if t["id"] == TASK_NAME), None)
    max_steps = task_info["max_steps"] if task_info else MAX_STEPS

    # Initialize tracking
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Log start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        observation = env_client.reset(TASK_NAME)
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            # Get action from LLM
            action = get_model_message(llm_client, step, observation, last_reward, history)

            # Take step in environment
            result = env_client.step(action)

            observation = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Log step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

            # Track history
            history.append(f"Step {step}: {action!r} -> reward {reward:+.2f}")

            if done:
                break

        # Get final grade
        grade_result = env_client.grade()
        score = grade_result.get("score", 0.0)
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}", flush=True)
        score = 0.0
        success = False

    finally:
        env_client.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
