# Bugs Found

Issues discovered while building and testing edictum-demo scenarios.

## Edictum Library

### 1. No bugs found in v0.5.2

All 5 framework adapters, 4 scenarios, and adversarial tests across 3 LLMs worked
cleanly with the public API. No workarounds required.

Previous versions had issues with CrewAI name normalization and OpenAI Agents
guardrail registration -- both are fixed in 0.5.2.

## Framework-Specific Notes

These are not Edictum bugs, but relevant context for adapter users:

| Framework | Note |
|-----------|------|
| OpenAI Agents SDK | Output guardrail is side-effect only -- cannot redact PII before the LLM sees it |
| CrewAI | ~3x token usage due to verbose prompt construction |
| Agno | No token usage metrics exposed by framework |
| Semantic Kernel | Requires filtering `AuthorRole.TOOL` from chat history to avoid duplicate tool results |
| LangGraph | `create_react_agent` deprecated in v1.0 -- moving to `langchain.agents` |
