# Edictum Demo Repository

Full scenario demos, adversarial tests, benchmarks, and observability setup for
[Edictum](https://github.com/acartag7/edictum) -- runtime contracts for AI agents.

**Docs:** [docs.edictum.dev](https://docs.edictum.dev)
**PyPI:** [pypi.org/project/edictum](https://pypi.org/project/edictum/)

## What's here

### Scenarios

Real-world governance scenarios demonstrating Edictum in different industries.

| Scenario | Directory | Description |
|----------|-----------|-------------|
| **Pharmacovigilance** | `scenarios/pharma/` | Clinical trial agent with patient data protection, change control, PII detection |
| **DevOps** | `scenarios/devops/` | Infrastructure agent with blast radius limits, secret protection |
| **Fintech** | `scenarios/fintech/` | Trading agent with trade limits, account access control, regulatory compliance |
| **Customer Support** | `scenarios/customer-support/` | Support agent with data minimization, refund limits, escalation control |

Each scenario includes a governed agent AND an unguarded baseline for comparison.

### Framework Adapters

Same governance contracts, 5 different agent frameworks. Proves Edictum is framework-agnostic.

| Framework | Demo | Adapter API |
|-----------|------|-------------|
| LangChain + LangGraph | `adapters/demo_langchain.py` | `adapter.as_tool_wrapper()` |
| OpenAI Agents SDK | `adapters/demo_openai_agents.py` | `adapter.as_guardrails()` |
| CrewAI | `adapters/demo_crewai.py` | `adapter.register()` |
| Agno | `adapters/demo_agno.py` | `adapter.as_tool_hook()` |
| Semantic Kernel | `adapters/demo_semantic_kernel.py` | `adapter.register(kernel)` |

### Adversarial Testing

Tests whether governance holds under adversarial conditions across multiple LLMs.

```bash
python adversarial/test_adversarial.py --model gpt-4.1
python adversarial/test_adversarial.py --model deepseek
python adversarial/test_adversarial.py --model qwen
```

### Benchmarks

```bash
python benchmark/benchmark_latency.py          # Governance overhead: ~55us
python benchmark/prompt_vs_contracts.py         # Prompt engineering vs contracts
```

### Observability

OTel -> Grafana Cloud pipeline with pre-built dashboard. See `observability/README.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy .env.example to .env and add your API keys
cp .env.example .env
```

Required API keys:
- `OPENAI_API_KEY` -- for GPT-4.1 agent demos
- `OPENROUTER_API_KEY` -- for DeepSeek/Qwen adversarial tests and DevOps demos
- `OTEL_EXPORTER_OTLP_ENDPOINT` + `OTEL_EXPORTER_OTLP_HEADERS` -- for Grafana Cloud (optional)

## Quick start

```bash
# Run a scenario
python scenarios/pharma/pharma_agent.py
python scenarios/fintech/fintech_agent.py
python scenarios/customer-support/support_agent.py

# Run with a different role (denied access)
python adapters/demo_langchain.py --role researcher

# Run in observe mode (log, don't block)
python adapters/demo_langchain.py --mode observe

# Run adversarial tests
python adversarial/test_adversarial.py

# Run benchmark
python benchmark/benchmark_latency.py
```

## Structure

```
edictum-demo/
  scenarios/
    pharma/                     # Clinical trial pharmacovigilance
    devops/                     # File organizer with blast radius limits
    fintech/                    # Trading compliance
    customer-support/           # Support agent with data minimization
  adapters/                     # 5 framework comparison demos
  adversarial/                  # Multi-model adversarial tests
  benchmark/                    # Latency + prompt-vs-contracts benchmarks
  observability/                # OTel config + Grafana dashboard
  docs/                         # Adapter development insights
  examples/                     # Claude Agent SDK demo
```

## Env vars

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (pharma, fintech, support demos) |
| `OPENROUTER_API_KEY` | OpenRouter API key (DevOps, adversarial tests) |
| `EDICTUM_MODEL` | Override LLM model for DevOps/SDK demos |
| `EDICTUM_OTEL_CONSOLE` | Set to `1` for console OTel output |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for Grafana Cloud |
| `OTEL_EXPORTER_OTLP_HEADERS` | URL-encoded auth headers |
