# Feature Ideas

Ideas for Edictum library improvements discovered while building demos.

## High Priority

### 1. Built-in PII detection postcondition

Every scenario needs a `pii-in-output` postcondition with essentially the same regex.
A built-in `pii_detection` postcondition type (like `sensitive_reads` for preconditions)
would eliminate boilerplate and provide better detection (NER, Presidio integration).

### 2. Session contract reset API

Session contracts (e.g., `max-tool-calls`) accumulate state across the guard lifetime.
There's no public API to reset session state between runs without creating a new guard.
Useful for long-running services handling multiple user sessions.

### 3. Cost/token budget contracts

A `token_budget` session contract that caps estimated API cost per session. The guard
already sees tool call counts -- if it could also see token usage, it could enforce
spend limits. Critical for fintech and enterprise deployments.

## Medium Priority

### 4. Contract composition / inheritance

Fintech and customer-support scenarios share ~60% of their contracts (PII detection,
session limits, ticket requirements). A way to compose contract bundles would reduce
duplication:

```yaml
apiVersion: edictum/v1
kind: ContractBundle
extends:
  - common/pii_contracts.yaml
  - common/session_limits.yaml
```

### 5. Deny reason templating

Contract denial messages are static strings. Supporting variable interpolation would
make them more actionable:

```yaml
message: "Trade of {args.quantity} shares exceeds limit of 1000 for role {principal.role}"
```

### 6. Adapter-level audit sink configuration

Currently OTel and audit sinks are configured at the guard level. Allowing per-adapter
sink configuration would let different frameworks route audit events differently
(e.g., CrewAI to console, LangChain to OTLP).

## Nice to Have

### 7. Contract testing CLI

A `edictum test` command that validates contracts against a set of test cases without
needing to spin up an agent:

```bash
edictum test contracts.yaml --cases test_cases.yaml
```

### 8. Observe-to-enforce migration report

When running in observe mode, a report showing "if you had been in enforce mode,
these N calls would have been denied" -- makes the B-to-C transition more confident.
