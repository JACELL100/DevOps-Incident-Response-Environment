---
title: DevOps Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# DevOps Incident Response Environment

An **OpenEnv-compliant** reinforcement learning environment for training AI agents in production incident response. This environment simulates a realistic microservices infrastructure where agents must diagnose and resolve production incidents.

## Overview

### What is this?

This environment simulates the work of a **Site Reliability Engineer (SRE)** responding to production incidents. Agents interact with a simulated microservices architecture, analyzing logs, metrics, and alerts to identify root causes and take remediation actions.

### Why is this useful?

- **Real-world applicability**: Incident response is a critical skill at every tech company
- **Training data scarcity**: Real incident data is sensitive and rare
- **Evaluation benchmark**: Standardized tasks to measure AI agent capabilities
- **Safe experimentation**: Learn from failures without impacting production

### Key Features

- Full **OpenEnv spec compliance** with typed Pydantic models
- **3 difficulty levels** from simple OOM fixes to complex multi-factor incidents
- **Realistic simulation** of microservices, logs, metrics, and alerts
- **Programmatic graders** with deterministic 0.0-1.0 scoring
- **Shaped rewards** with partial progress signals

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/openenv/devops-incident-response
cd devops-incident-response

# Install dependencies
pip install -e .
```

### Run the Server

```bash
# Start the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
# Set your API credentials (REQUIRED)
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="your-huggingface-token"
export ENV_URL="http://localhost:7860"  # or your HF Space URL

# Run the baseline agent
python inference.py
```

### Docker

```bash
# Build from project root
docker build -f server/Dockerfile -t devops-incident-response .

# Run the container
docker run -p 7860:7860 devops-incident-response
```

## Project Structure

```
devops-incident-response/
├── __init__.py              # Root exports (Action, Observation, Env)
├── models.py                # Pydantic models
├── client.py                # EnvClient for remote access
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv configuration
├── pyproject.toml           # Package configuration
├── README.md                # This file
├── outputs/                 # Runtime outputs
│   ├── logs/
│   └── evals/
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI server
    ├── environment.py       # Main environment
    ├── simulator.py         # Infrastructure simulator
    ├── graders.py           # Task graders
    ├── tasks.py             # Task definitions
    ├── requirements.txt     # Server dependencies
    └── Dockerfile           # Container definition
```

## Environment Details

### The Simulated Infrastructure

The environment simulates a typical e-commerce microservices architecture:

```
                     ┌─────────────────┐
                     │   API Gateway   │
                     └────────┬────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
   ┌─────┴─────┐       ┌──────┴──────┐      ┌─────┴─────┐
   │   User    │       │    Order    │      │  Product  │
   │  Service  │       │   Service   │      │  Service  │
   └─────┬─────┘       └──────┬──────┘      └─────┬─────┘
         │                    │                    │
         │             ┌──────┴──────┐            │
         │             │   Payment   │            │
         │             │   Service   │            │
         │             └─────────────┘            │
         │                                        │
   ┌─────┴─────┐                           ┌─────┴─────┐
   │   Redis   │                           │Elasticsearch│
   │   Cache   │                           └───────────┘
   └───────────┘
         │
   ┌─────┴─────┐       ┌─────────────┐
   │ PostgreSQL│       │    Kafka    │
   │    DB     │       │             │
   └───────────┘       └─────────────┘
```

### Action Space

| Action | Description | Example |
|--------|-------------|---------|
| `query_service` | Get service status and info | `query_service:order-service` |
| `read_logs` | Read recent log entries | `read_logs:user-service` |
| `get_metrics` | Get CPU, memory, latency metrics | `get_metrics:postgres-db` |
| `get_alerts` | Get active monitoring alerts | `get_alerts` |
| `run_diagnostic` | Run diagnostic checks | `run_diagnostic:redis-cache` |
| `restart_service` | Restart a service | `restart_service:order-service` |
| `scale_service` | Scale service replicas | `scale_service:api-gateway:5` |
| `rollback_service` | Rollback to previous version | `rollback_service:payment-service` |
| `update_config` | Update configuration | `update_config:order-service:JAVA_OPTS:-Xmx1024m` |
| `resolve_incident` | Mark incident resolved | `resolve_incident` |
| `get_runbook` | Consult a runbook | `get_runbook:oom-response` |
| `get_slo_status` | Check SLO violations | `get_slo_status:order-service` |
| `get_timeline` | View incident timeline | `get_timeline` |
| `escalate` | Escalate to next level | `escalate` |

### Observation Space

Each observation includes:

- **Incident Context**: Title, severity, description, affected services, customer impact
- **Visible Data**: Services queried, logs read, metrics fetched, alerts active
- **Last Action Result**: Success/failure, error messages, returned data
- **Available Services**: List of all services in the infrastructure
- **SLO Violations**: Services currently violating SLOs
- **Incident Duration**: Time elapsed since incident started
- **Escalation Level**: Current escalation level (1-3)

### Reward Structure

Rewards are shaped to guide learning:

| Component | Value | Description |
|-----------|-------|-------------|
| Diagnostic actions | +0.02 | Investigating affected services |
| Correct remediation | +0.10 | Taking required fix actions |
| Successful resolution | +0.30 | Fully resolving the incident |
| Efficiency bonus | up to +0.10 | Resolving quickly |
| Runbook consultation | +0.01 | Using runbooks for guidance |
| Unnecessary actions | -0.01 | Actions not needed |
| Time pressure | -0.01/step | Penalty after 80% of max steps |

## Tasks

### Task 1: Memory Crisis (Easy)

**Scenario**: The order-service has been OOM killed during a traffic spike.

**Root Cause**: Java heap size too small for traffic volume.

**Solution**: Increase JAVA_OPTS heap size and restart the service.

**Max Steps**: 15

### Task 2: Database Overload (Medium)

**Scenario**: Multiple services experiencing database connection timeouts with cascading failures.

**Root Cause**: PostgreSQL max connections too low, causing connection pool exhaustion across services.

**Solution**: Increase database connections and connection pool sizes for affected services.

**Max Steps**: 25

### Task 3: Perfect Storm (Hard)

**Scenario**: Multiple alerts firing across infrastructure with several services affected.

**Root Cause**: Multiple concurrent issues - Redis cluster network partition causing cache failures, leading to database overload, PLUS an unrelated memory leak in product-service.

**Solution**: Fix Redis cluster, restart services with memory leaks, scale as needed.

**Max Steps**: 40

### Task 4: Security Breach (Expert)

**Scenario**: Active credential stuffing attack detected. Suspicious login attempts, account lockouts, and traffic anomalies from specific IP ranges.

**Root Cause**: Credential stuffing attack from malicious IP range targeting authentication service.

**Solution**: Block malicious IPs at WAF, adjust lockout policies, scale auth service to handle cleanup.

**Max Steps**: 50

## Enhanced Features

### 🔖 Runbooks

Agents can consult runbooks for guidance on handling specific incident types:

```python
# Get a specific runbook
action = "get_runbook:oom-response"

# Search runbooks by keyword
action = "get_runbook"  # Lists all available runbooks
```

Available runbooks:
- `oom-response` - Out of Memory incidents
- `database-connection-pool` - Connection pool issues
- `cascading-failure` - Cascading failure response
- `security-incident` - Credential attack response
- `cache-failure` - Redis/cache issues
- `bad-deployment` - Deployment rollback procedures

### 📊 SLO Tracking

Monitor Service Level Objectives during incidents:

- **Availability SLO**: Target 99.9% uptime
- **Latency P99 SLO**: Target < 200ms
- **Error Rate SLO**: Target < 1%

```python
# Check SLO status
action = "get_slo_status:order-service"
```

### 📜 Incident Timeline

Track all events during incident response for post-incident review:

```python
# Get timeline
action = "get_timeline"
```

### 🚨 Escalation

Escalate incidents to higher severity levels when needed:

```python
# Escalate incident
action = "escalate"
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Reset environment, get initial observation |
| POST | `/step` | Take an action, get result |
| GET | `/state/{session_id}` | Get full environment state |
| POST | `/grade` | Grade completed episode |
| GET | `/tasks` | List available tasks |
| GET | `/health` | Health check |

### Using the Python Client

```python
from client import EnvClient

# Connect to server
client = EnvClient("http://localhost:7860")

# Start episode
session_id, observation = client.reset("task_easy_oom")

# Run episode
while True:
    action = "get_alerts"  # Your agent logic here
    observation, reward, done, info = client.step(session_id, action)
    if done:
        break

# Get final score
result = client.grade(session_id)
print(f"Score: {result['score']}")
```

## Grading

Each task is graded on three components:

1. **Diagnosis Score** (30-40%): Did the agent investigate the affected services?
2. **Remediation Score** (45-60%): Did the agent take the correct fix actions?
3. **Efficiency Score** (10-15%): How quickly was the incident resolved?

Final scores range from 0.0 to 1.0.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### OpenEnv CLI

```bash
# Validate environment
openenv validate .

# Deploy to Hugging Face
openenv push --repo-id your-username/devops-incident-response
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | **Yes** |
| `MODEL_NAME` | Model to use for inference | **Yes** |
| `HF_TOKEN` | Hugging Face / API key for LLM | **Yes** |
| `ENV_URL` | Environment server URL | No (default: `http://localhost:7860`) |
| `TASK_NAME` | Task to run | No (default: `task_easy_oom`) |
| `PORT` | Server port | No (default: `7860`) |

### Example .env file

```bash
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
ENV_URL=https://your-space.hf.space
```

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

Built for the OpenEnv Hackathon. Special thanks to Hugging Face and Meta for organizing.
