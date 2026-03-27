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
pip install -r requirements.txt
```

### Run the Server

```bash
# Start the FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
# Set your API credentials
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-api-key"

# Run the baseline agent
python inference.py
```

### Docker

```bash
# Build the image
docker build -t devops-incident-response .

# Run the container
docker run -p 7860:7860 devops-incident-response
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

### Observation Space

Each observation includes:

- **Incident Context**: Title, severity, description, affected services, customer impact
- **Visible Data**: Services queried, logs read, metrics fetched, alerts active
- **Last Action Result**: Success/failure, error messages, returned data
- **Available Services**: List of all services in the infrastructure

### Reward Structure

Rewards are shaped to guide learning:

| Component | Value | Description |
|-----------|-------|-------------|
| Diagnostic actions | +0.02 | Investigating affected services |
| Correct remediation | +0.10 | Taking required fix actions |
| Successful resolution | +0.30 | Fully resolving the incident |
| Efficiency bonus | up to +0.10 | Resolving quickly |
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

### Example: Running an Episode

```python
import requests

BASE_URL = "http://localhost:7860"

# Reset environment
response = requests.post(f"{BASE_URL}/reset", json={"task_id": "task_easy_oom"})
session_id = response.json()["session_id"]
observation = response.json()["observation"]

# Take actions
while True:
    action = {"session_id": session_id, "action_str": "get_alerts"}
    response = requests.post(f"{BASE_URL}/step", json=action)
    result = response.json()

    if result["done"]:
        break

# Grade the episode
response = requests.post(f"{BASE_URL}/grade", json={"session_id": session_id})
print(f"Score: {response.json()['score']}")
```

## Grading

Each task is graded on three components:

1. **Diagnosis Score** (30-40%): Did the agent investigate the affected services?
2. **Remediation Score** (45-60%): Did the agent take the correct fix actions?
3. **Efficiency Score** (10-15%): How quickly was the incident resolved?

Final scores range from 0.0 to 1.0.

## Baseline Scores

| Task | GPT-4o-mini | GPT-4o |
|------|-------------|--------|
| Easy (OOM) | ~0.65 | ~0.82 |
| Medium (Cascade) | ~0.45 | ~0.68 |
| Hard (Complex) | ~0.30 | ~0.52 |

## Development

### Project Structure

```
devops-incident-response/
├── app.py                 # FastAPI server
├── inference.py           # Baseline inference script
├── openenv.yaml           # OpenEnv configuration
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── src/
    ├── __init__.py        # Package exports
    ├── models.py          # Pydantic models
    ├── environment.py     # Main environment
    ├── simulator.py       # Infrastructure simulator
    ├── graders.py         # Task graders
    └── tasks/
        └── __init__.py    # Task definitions
```

### Running Tests

```bash
pytest tests/ -v
```

### Validation

```bash
# Validate OpenEnv compliance
openenv validate .
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model to use for inference | `gpt-4o-mini` |
| `OPENAI_API_KEY` | API key for LLM | Required |
| `HF_TOKEN` | Alternative API key | - |
| `PORT` | Server port | `7860` |

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

Built for the OpenEnv Hackathon. Special thanks to Hugging Face and Meta for organizing.
