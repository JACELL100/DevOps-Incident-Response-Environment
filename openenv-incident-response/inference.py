#!/usr/bin/env python3
"""
Baseline Inference Script for DevOps Incident Response Environment.

This script runs an AI agent against all tasks in the environment
and reports performance scores. It uses the OpenAI API client
as specified in the hackathon requirements.

Environment Variables:
    API_BASE_URL: The API endpoint for the LLM
    MODEL_NAME: The model identifier to use
    HF_TOKEN: Hugging Face / API key (used as OPENAI_API_KEY)
    OPENAI_API_KEY: Alternative API key variable

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from openai import OpenAI

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    Action,
    ActionType,
    IncidentResponseEnv,
    grade_task,
    list_tasks,
)


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

MAX_TOKENS = 1024
TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.

Your job is to diagnose the root cause of the incident and take appropriate remediation actions.

## Available Actions

You can take the following actions (use the exact format shown):

1. **Query Service Status**: `query_service:<service_name>`
   - Get current status, replicas, version, and dependencies

2. **Read Logs**: `read_logs:<service_name>`
   - Get recent log entries from the service

3. **Get Metrics**: `get_metrics:<service_name>`
   - Get CPU, memory, latency, and error rate metrics

4. **Get Alerts**: `get_alerts`
   - Get all active monitoring alerts

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
THOUGHT: The order-service is showing OOM errors in the logs. The current heap size is too small. I need to increase the Java heap size.
ACTION: update_config:order-service:JAVA_OPTS:-Xmx1024m -Xms512m
```

## Guidelines

1. Start by gathering information - check alerts, query affected services, read logs
2. Look for patterns in errors and metrics
3. Identify the root cause before attempting fixes
4. Configuration changes require a service restart to take effect
5. Work methodically through the affected services
6. Only call resolve_incident when you've verified the fix worked

Remember: You're dealing with a production system. Be careful and methodical."""


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
# Agent
# ============================================================================

class IncidentResponseAgent:
    """AI agent for incident response using OpenAI API."""

    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model
        self.conversation_history: list[dict[str, Any]] = []

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []

    def get_action(self, observation_str: str) -> str:
        """Get an action from the model based on current observation."""
        # Add observation to history
        self.conversation_history.append({
            "role": "user",
            "content": observation_str,
        })

        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history,
        ]

        # Call the model
        for attempt in range(MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response = completion.choices[0].message.content or ""

                # Add response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })

                # Parse action
                action = parse_model_response(response)
                return action

            except Exception as e:
                print(f"  API call failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        # Fallback
        return "get_alerts"


# ============================================================================
# Observation Formatter
# ============================================================================

def format_observation(obs) -> str:
    """Format observation as readable text for the model."""
    lines = []

    # Incident info
    lines.append("=" * 60)
    lines.append(f"INCIDENT: {obs.incident.title}")
    lines.append(f"Severity: {obs.incident.severity}")
    lines.append(f"Description: {obs.incident.description}")
    lines.append(f"Affected Services: {', '.join(obs.incident.affected_services)}")
    lines.append(f"Customer Impact: {obs.incident.customer_impact}")
    lines.append("=" * 60)

    # Step info
    lines.append(f"\nStep: {obs.step}")

    # Available services
    lines.append(f"\nAvailable services: {', '.join(obs.available_services)}")

    # Last action result
    if obs.last_action:
        lines.append(f"\nLast action: {obs.last_action}")
        if obs.last_action_success:
            lines.append("Result: SUCCESS")
        else:
            lines.append(f"Result: FAILED - {obs.last_action_error}")

        if obs.last_action_result:
            result_str = json.dumps(obs.last_action_result, indent=2, default=str)
            lines.append(f"Data:\n{result_str}")

    # Active alerts
    if obs.visible_alerts:
        lines.append("\n--- ACTIVE ALERTS ---")
        for alert in obs.visible_alerts:
            lines.append(f"[{alert.severity}] {alert.title}")
            lines.append(f"  Service: {alert.service}")
            lines.append(f"  {alert.description}")

    # Hint (for easy tasks)
    if obs.hint:
        lines.append(f"\nHINT: {obs.hint}")

    return "\n".join(lines)


# ============================================================================
# Run Episode
# ============================================================================

def run_episode(env: IncidentResponseEnv, agent: IncidentResponseAgent) -> dict[str, Any]:
    """Run a single episode and return results."""
    observation = env.reset()
    agent.reset()

    total_reward = 0.0
    steps = 0
    actions_taken = []

    print(f"\n{'='*60}")
    print(f"Starting episode: {env.task.name}")
    print(f"Difficulty: {env.task.difficulty}")
    print(f"Max steps: {env.task.max_steps}")
    print(f"{'='*60}\n")

    while not env.done and steps < env.task.max_steps:
        # Format observation for model
        obs_str = format_observation(observation)

        # Get action from agent
        action_str = agent.get_action(obs_str)
        print(f"Step {steps + 1}: {action_str}")

        # Execute action
        action = Action(action_str=action_str)
        result = env.step(action)

        observation = result.observation
        total_reward += result.reward.value
        steps += 1
        actions_taken.append(action_str)

        # Print progress
        if result.done:
            print(f"  -> Episode done. Total reward: {total_reward:.4f}")
            break

    # Grade the episode
    grade = grade_task(env.task_id, env)

    return {
        "task_id": env.task_id,
        "task_name": env.task.name,
        "difficulty": env.task.difficulty,
        "steps": steps,
        "total_reward": total_reward,
        "score": grade.score,
        "diagnosis_score": grade.diagnosis_score,
        "remediation_score": grade.remediation_score,
        "efficiency_score": grade.efficiency_score,
        "feedback": grade.feedback,
        "resolved": env.state().resolved,
        "actions": actions_taken,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run inference on all tasks."""
    print("DevOps Incident Response Environment - Baseline Inference")
    print("=" * 60)

    # Check API key
    if not API_KEY:
        print("ERROR: No API key found. Set OPENAI_API_KEY or HF_TOKEN environment variable.")
        sys.exit(1)

    # Initialize client
    print(f"API Base: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    # Create agent
    agent = IncidentResponseAgent(client, MODEL_NAME)

    # Get all tasks
    tasks = list_tasks()
    print(f"\nFound {len(tasks)} tasks:")
    for task in tasks:
        print(f"  - {task['id']}: {task['name']} ({task['difficulty']})")

    # Run each task
    results = []
    for task in tasks:
        task_id = task["id"]
        env = IncidentResponseEnv(task_id=task_id)

        result = run_episode(env, agent)
        results.append(result)

        print(f"\n--- {task_id} Results ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Resolved: {result['resolved']}")
        print(f"Feedback: {result['feedback']}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0.0

    for result in results:
        status = "RESOLVED" if result["resolved"] else "UNRESOLVED"
        print(f"{result['task_id']}: {result['score']:.4f} [{status}]")

    print(f"\nAverage Score: {avg_score:.4f}")
    print(f"Total Score: {total_score:.4f}")

    # Save results
    output_file = "inference_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": avg_score,
            "total_score": total_score,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    main()
