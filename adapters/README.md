# Framework Adapters

Same governance contracts, same pharma scenario, 5 different agent frameworks. Proves Edictum is framework-agnostic.

All demos use the pharmacovigilance scenario from `scenarios/pharma/` with identical YAML contracts, mock data, and task. The shared utilities are in `shared.py`.

## Demos

| Framework | File | Adapter API |
|-----------|------|-------------|
| LangChain + LangGraph | `demo_langchain.py` | `adapter.as_tool_wrapper()` |
| OpenAI Agents SDK | `demo_openai_agents.py` | `adapter.as_guardrails()` |
| CrewAI | `demo_crewai.py` | `adapter.register()` |
| Agno | `demo_agno.py` | `adapter.as_tool_hook()` |
| Semantic Kernel | `demo_semantic_kernel.py` | `adapter.register(kernel)` |

## Run

```bash
python adapters/demo_langchain.py
python adapters/demo_openai_agents.py
python adapters/demo_agno.py
python adapters/demo_semantic_kernel.py
python adapters/demo_crewai.py

# Different role (denied access)
python adapters/demo_langchain.py --role researcher

# Observe mode (log but don't block)
python adapters/demo_langchain.py --mode observe

# With tracking ticket (unlocks case report updates)
python adapters/demo_langchain.py --ticket CAPA-2025-042
```

All demos require `OPENAI_API_KEY` in `.env`.

## Results Summary

| Framework | Tool Calls | Denied | PII Warnings | Est. Cost |
|-----------|-----------|--------|-------------|-----------|
| LangChain | 9 | 1 | 1 | $0.025 |
| OpenAI Agents | 9 | 1 | 2 | $0.018 |
| Agno | 11 | 1 | 2 | N/A |
| Semantic Kernel | 11 | 1 | 2 | $0.008 |
| CrewAI | ~8 | 1 | ~2 | $0.040 |

## Key Findings

- **LangChain** is the gold standard -- zero workarounds, true PII interception
- **Agno** is the simplest integration -- one line (`tool_hooks=[hook]`), but no token metrics
- **Semantic Kernel** is cheapest -- batches tool calls, but requires careful chat history handling
- **OpenAI Agents SDK** output guardrail is side-effect only -- cannot redact PII before the LLM sees it
- **CrewAI** is most expensive (~3x token usage) due to verbose prompt construction

## Known Limitations

| Framework | Limitation |
|-----------|------------|
| **OpenAI Agents SDK** | Output guardrail is side-effect only -- it can detect PII in tool output and log a warning, but cannot redact/modify the response before the LLM sees it. PII reaches the model context. |
| **CrewAI** | ~3x token usage compared to other frameworks due to verbose internal prompt construction. The adapter works correctly, but costs scale accordingly. |
| **Agno** | No token usage metrics exposed by the framework. Cost tracking is unavailable. |
| **Semantic Kernel** | When building chat history for the LLM, filter out `AuthorRole.TOOL` messages to avoid duplicating tool results that SK already inlines. |

See `FINDINGS.md` for detailed per-adapter integration notes, bugs found, and workarounds.
