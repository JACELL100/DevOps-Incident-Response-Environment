#!/usr/bin/env python3
"""
Baseline Inference Script for DevOps Incident Response Environment.

This script runs an AI agent against the environment deployed on Hugging Face Spaces
and reports performance scores using the required structured logging format.

IMPORTANT: This script runs ALL available tasks (at least 3) as required by the
OpenEnv validator. Each task is graded independently.

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
import urllib.request
import urllib.error
from typing import Any, List, Optional

from openai import OpenAI

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required


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

# Tasks to run - MUST run at least 3 tasks for validator
TASKS_TO_RUN = [
    "task_easy_oom",
    "task_medium_cascade",
    "task_hard_complex",
]

# Inference parameters
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
# Environment Client (HTTP-based using stdlib)
# ============================================================================

class EnvClient:
    """HTTP client for interacting with the environment server using stdlib."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None

    def _request(self, method: str, path: str, data: dict = None) -> dict:
        """Make an HTTP request using urllib."""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
        else:
            req = urllib.request.Request(url, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"HTTP {e.code}: {error_body}")

    def reset(self, task_id: str) -> dict:
        """Reset the environment and start a new episode."""
        data = self._request("POST", "/reset", {"task_id": task_id})
        self.session_id = data["session_id"]
        return data["observation"]

    def step(self, action_str: str) -> dict:
        """Take an action in the environment."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before step()")
        return self._request("POST", "/step", {"session_id": self.session_id, "action_str": action_str})

    def grade(self) -> dict:
        """Get the final grade for the episode."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before grade()")
        return self._request("POST", "/grade", {"session_id": self.session_id})

    def get_tasks(self) -> list:
        """Get list of available tasks."""
        return self._request("GET", "/tasks")["tasks"]

    def health(self) -> bool:
        """Check if the environment is healthy."""
        try:
            self._request("GET", "/health")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client (no-op for urllib)."""
        pass


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
# Run Single Task
# ============================================================================

def run_single_task(
    llm_client: OpenAI,
    env_client: EnvClient,
    task_id: str,
    max_steps: int
) -> tuple[bool, int, float, List[float]]:
    """
    Run a single task and return results.
    
    Returns:
        (success, steps_taken, score, rewards)
    """
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Log start
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment for this task
        observation = env_client.reset(task_id)
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
        # Ensure score is strictly between 0 and 1
        score = max(0.01, min(0.99, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}", flush=True)
        score = 0.01  # Minimum valid score on error
        success = False

    # Log end for this task
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return success, steps_taken, score, rewards


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run inference on ALL required tasks (at least 3)."""
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

    # Get all available tasks from the server
    available_tasks = env_client.get_tasks()
    task_map = {t["id"]: t for t in available_tasks}
    
    # Determine which tasks to run (at least 3)
    tasks_to_run = []
    for task_id in TASKS_TO_RUN:
        if task_id in task_map:
            tasks_to_run.append(task_id)
    
    # Fallback: if we don't have 3 tasks, add more from available
    if len(tasks_to_run) < 3:
        for task in available_tasks:
            if task["id"] not in tasks_to_run:
                tasks_to_run.append(task["id"])
            if len(tasks_to_run) >= 3:
                break
    
    print(f"[INFO] Running {len(tasks_to_run)} tasks: {tasks_to_run}", flush=True)
    
    # Run each task
    all_results = []
    for task_id in tasks_to_run:
        task_info = task_map.get(task_id, {})
        max_steps = task_info.get("max_steps", 15)
        
        print(f"\n[INFO] Starting task: {task_id} (max_steps={max_steps})", flush=True)
        
        success, steps, score, rewards = run_single_task(
            llm_client, env_client, task_id, max_steps
        )
        
        all_results.append({
            "task_id": task_id,
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        })
        
        print(f"[INFO] Task {task_id} complete: score={score:.4f}, success={success}", flush=True)
    
    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("[SUMMARY] All tasks completed:", flush=True)
    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0
    print(f"  Tasks run: {len(all_results)}", flush=True)
    print(f"  Average score: {avg_score:.4f}", flush=True)
    for r in all_results:
        print(f"  - {r['task_id']}: score={r['score']:.4f}, success={r['success']}", flush=True)
    print("=" * 60, flush=True)
    
    env_client.close()


if __name__ == "__main__":
    main()
