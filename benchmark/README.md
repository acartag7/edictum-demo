# Benchmarks

Performance measurement for Edictum governance overhead.

## benchmark_latency.py

End-to-end latency measurement with real OpenAI API calls. Measures 4 phases:

1. **Baseline** -- direct tool call (no LLM, no governance)
2. **Governance only** -- Edictum contract evaluation without LLM
3. **LLM only** -- OpenAI API call without governance
4. **End-to-end** -- full agent loop with LLM + governance

```bash
python benchmark/benchmark_latency.py
```

**Latest results:** 54.3us governance overhead = 0.01% of 764ms LLM round-trip.

## prompt_vs_contracts.py

A -> B -> C customer journey benchmark comparing three deployment stages:

- **A (Today)**: Bloated system prompt with governance rules in natural language. LLM self-polices.
- **B (Day-one)**: Clean prompt, Edictum in observe mode. Full visibility, zero behavior change.
- **C (Production)**: Clean prompt, Edictum in enforce mode. Deterministic contract enforcement.

```bash
python benchmark/prompt_vs_contracts.py               # all scenarios
python benchmark/prompt_vs_contracts.py --quick        # default scenario only
python benchmark/prompt_vs_contracts.py --runs 3       # repeat for non-determinism evidence
```

Requires `OPENAI_API_KEY` in `.env`.

## What the benchmarks prove

1. **Governance overhead is negligible** -- microseconds vs hundreds of milliseconds for LLM calls
2. **Contracts are deterministic** -- same input always produces the same governance decision, unlike prompt engineering
3. **Deploy observe mode tomorrow** -- zero risk, full audit trail, then flip to enforce when confident
