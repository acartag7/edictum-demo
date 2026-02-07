# Edictum Adapter Demos — Findings & Quirks

Discovered during implementation and runtime testing of full adapter demos for Edictum 0.5.1.
Upstream fixes shipped in **Edictum 0.5.2** (PyPI).
All demos use the same pharmacovigilance scenario with identical contracts, mock data, and task.

---

## Runtime Test Results

### Test Matrix

Each adapter was run in 3 configurations:
- **Default** (`--role pharmacovigilance`): Full access, PII postconditions trigger
- **Researcher** (`--role researcher`): Restricted access, denied from detailed AE data
- **Observe** (`--mode observe`): Log-only mode, would-deny events without blocking

### Governance Results

| Metric | LangChain | OpenAI Agents | Agno | Semantic Kernel | CrewAI |
|--------|-----------|---------------|------|-----------------|--------|
| **Default — Tool calls** | 11 | 11 | 11 | 10 | 5 |
| **Default — Allowed** | 10 | 10 | 10 | 8 | 4 |
| **Default — Denied** | 1 | 1 | 1 | 2 | 1 |
| **Default — PII warnings** | 2 | 2 | 2 | 2 | 1 |
| **Researcher — Denied** | 3 | 3 | 3 | 2 | 3 |
| **Observe — Would-deny** | 1 | — | 1 | 1 | — |

All frameworks enforce the same contracts consistently:
- `case-report-requires-ticket`: Denies case report updates without a CAPA/deviation ticket
- `restrict-patient-data`: Denies researcher role from `adverse_events_detailed`
- PII postcondition: Detects patient identifiers in `adverse_events_detailed` output

### Token Usage & Cost (GPT-4.1, default role)

| Metric | LangChain | OpenAI Agents | Agno | Semantic Kernel | CrewAI |
|--------|-----------|---------------|------|-----------------|--------|
| LLM round-trips | 6 | 6 | N/A | 1* | 6 |
| Prompt tokens | ~7,300 | 7,318 | N/A | 1,037* | 9,378 |
| Completion tokens | ~1,000 | 1,034 | N/A | 499* | 4,878 |
| Est. cost | ~$0.020 | $0.023 | N/A | $0.007* | $0.058 |

\* SK batches tool calls internally — reports fewer round-trips. Total was 2,136 tokens but only 1 round-trip reported.
Agno does not expose token metrics via `response.metrics` (underdocumented API).
CrewAI is the most expensive due to verbose prompt construction (agent role, backstory, task description repeated).

---

## Per-Adapter Findings

### LangChain + LangGraph

**Status: Gold standard. Cleanest integration.**

**Integration pattern:**
```python
adapter = LangChainAdapter(guard, principal=principal)
wrapper = adapter.as_tool_wrapper(on_postcondition_warn=redact_callback)
tool_node = ToolNode(tools=tools, wrap_tool_call=wrapper)
```

**What works well:**
- Wrap-around pattern is the most powerful: adapter executes the tool and can transform the result before the LLM sees it
- `on_postcondition_warn` callback receives `(result: ToolMessage, findings: list[Finding])` — can mutate `result.content` directly
- PII redaction is **true interception**: the LLM never sees unredacted content
- Token tracking via `usage_metadata` on AIMessage — must iterate ALL AI messages (intermediate tool-calling messages carry tokens too)

**Issues found:** None. This is the reference implementation.

**Workarounds needed:** None.

---

### OpenAI Agents SDK

**Status: Works after workaround. Has an adapter bug and a PII gap.**

**Integration pattern (workaround):**
```python
# adapter.as_guardrails() is BROKEN — creates 3-arg functions but SDK expects 1-arg.
# Must construct guardrails manually:
async def _input_guardrail_fn(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    result = await adapter._pre(tool_name, tool_arguments, call_id)
    if result is not None:
        return ToolGuardrailFunctionOutput.reject_content(result)
    return ToolGuardrailFunctionOutput.allow()

input_gr = ToolInputGuardrail(guardrail_function=_input_guardrail_fn)
# Guardrails go on TOOLS, not on Agent:
@function_tool(tool_input_guardrails=[input_gr], tool_output_guardrails=[output_gr])
def query_clinical_data(...): ...
```

**Bugs found:**

1. **EDICTUM BUG: `as_guardrails()` signature mismatch.** The adapter's `as_guardrails()` method uses `@tool_input_guardrail` / `@tool_output_guardrail` decorators which create functions with signature `(context, agent, data)`. But the SDK's `ToolInputGuardrail.run()` calls `guardrail_function(data)` with only 1 argument. This is a real bug in the edictum adapter — the decorator-based approach doesn't match the SDK's calling convention. **Severity: Blocker.** Users cannot use `as_guardrails()` at all.

2. **SDK limitation: Tool guardrails vs agent guardrails.** `ToolInputGuardrail` / `ToolOutputGuardrail` go on individual `@function_tool` decorators, NOT on `Agent(input_guardrails=...)`. The agent-level `input_guardrails` expects `InputGuardrail` objects (different class). Confusing API.

3. **PII detection is side-effect only.** The output guardrail detects PII via postconditions but **cannot redact** the tool result before the LLM sees it. The SDK's output guardrail returns allow/reject — reject would drop the entire result rather than redacting specific fields. This is a framework limitation.

4. **`Runner.run_sync()` fails in asyncio context.** `asyncio.run(main())` + `Runner.run_sync()` = `RuntimeError: cannot be called when an event loop is already running`. Must use `await Runner.run()` with `async def main()`.

5. **`openai` 2.x dependency.** The `agents` SDK 0.8.1 requires features from `openai>=2.0`. This conflicts with `semantic-kernel` which requires `openai<2`. Cannot run both in the same venv without version pinning gymnastics.

**Workarounds applied:**
- Bypass `as_guardrails()`, construct `ToolInputGuardrail`/`ToolOutputGuardrail` with 1-arg functions calling `adapter._pre()`/`_post()` directly
- Use `await Runner.run()` instead of `Runner.run_sync()`
- Catch guardrail exceptions by class name pattern (`"Tripwire" in type(e).__name__`)

---

### Agno

**Status: Works well. True PII interception.**

**Integration pattern:**
```python
adapter = AgnoAdapter(guard, principal=principal)
hook = adapter.as_tool_hook(on_postcondition_warn=redact_callback)
agent = Agent(model=OpenAIChat(id="gpt-4.1"), tools=[...], tool_hooks=[hook])
```

**What works well:**
- Wrap-around hook: adapter calls the tool and can transform the result via callback
- `redact_callback` receives `(result: str, findings: list[Finding])` — raw string, not framework wrapper
- PII redaction is true interception (same as LangChain)
- Simplest integration pattern — just pass `tool_hooks=[hook]`

**Issues found:**

1. **No token metrics.** `response.metrics` dict exists but does not contain token counts in our testing. The API for metrics is underdocumented — key names may vary between versions (`input_tokens` vs `prompt_tokens`). Fallback: iterate `response.messages` for per-message metrics, but this is also unreliable.

2. **Async-to-sync bridging.** Agno's `agent.run()` is synchronous. The adapter bridges async Edictum calls via `ThreadPoolExecutor`. Under concurrent tool calls in the same agent run, this bridge is untested and potentially fragile.

3. **Callback type differs from LangChain.** Must check `isinstance(result, str)` not `hasattr(result, 'content')`. The `on_postcondition_warn` callback API is `(result, findings) -> result` across all adapters, but the `result` type varies per framework.

**Workarounds needed:** None for core functionality. Token tracking requires alternative approach.

---

### Semantic Kernel

**Status: Works after significant fixes. Multiple integration hurdles.**

**Integration pattern:**
```python
adapter = SemanticKernelAdapter(guard, principal=principal)
adapter.register(kernel, on_postcondition_warn=redact_callback)
# Manual chat loop required:
while True:
    result = await chat_service.get_chat_message_content(chat_history, settings, kernel=kernel)
    if result.role != AuthorRole.TOOL:
        chat_history.add_message(result)
    if no_more_function_calls(result):
        break
```

**Bugs found:**

1. **Chat history corruption on deny.** When Edictum denies a tool call, the adapter sets `context.terminate = True`. SK's `invoke_function_call` still adds a `FunctionResultContent` (the denial message) to chat history. On the next loop iteration, the chat history has a `tool` role message without a matching `tool_calls` assistant message preceding it. OpenAI API rejects this: `messages with role 'tool' must be a response to a preceding message with 'tool_calls'`. **Fix:** Only add non-TOOL messages to chat_history in the demo's while loop (`if result.role != AuthorRole.TOOL`).

2. **FunctionResult wrapping (FIXED UPSTREAM).** The adapter set `context.function_result` to a raw string on denial and on postcondition remediation. In SK 1.39+, this field is typed as `FunctionResult | None` with pydantic validation, so raw strings fail. **Fix applied to edictum library** (`semantic_kernel.py`): added `_wrap_result` helper that wraps values in `FunctionResult(function=context.function.metadata, value=value)`.

3. **Tool name handling works correctly.** Despite initial concerns, the adapter correctly uses `context.function.name` (just `query_clinical_data`) not the fully-qualified `pharma-query_clinical_data`. No normalization needed in the demo.

4. **Token tracking under-reports.** SK batches tool call execution internally. `get_chat_message_content()` may process multiple tool calls and return a single result. This means `llm_calls` count is lower than actual API calls, and token counts are incomplete.

5. **Pydantic version sensitivity.** SK 1.8.0 imports `pydantic.networks.Url` which was removed in pydantic 2.12+. Required upgrading to SK 1.39.3 (which pins `pydantic<2.12,>=2.0`) and downgrading pydantic from 2.12.5 to 2.11.10. This creates pip dependency conflicts with `openai-agents` (wants `pydantic>=2.12.3`).

**Workarounds applied:**
- Filter `AuthorRole.TOOL` messages from manual `chat_history.add_message()` calls
- Proper `FunctionResult` wrapping in adapter filter

---

### CrewAI

**Status: Most quirky. Works but requires the most workarounds.**

**Integration pattern:**
```python
adapter = CrewAIAdapter(guard, principal=principal)
adapter._on_postcondition_warn = warn_callback

# Cannot use adapter.register() — decorators fail on bound methods.
# Must register plain functions directly:
register_before_tool_call_hook(before_hook_sync)
register_after_tool_call_hook(after_hook_sync)
```

**Issues found:**

1. **`adapter.register()` fails.** CrewAI's `@before_tool_call` / `@after_tool_call` decorators call `setattr(func, marker, True)` which fails on bound methods. **Fix:** Use `register_before_tool_call_hook()` / `register_after_tool_call_hook()` directly with plain functions.

2. **Global hooks.** Hooks are registered globally per process. Only one adapter can be active. Multiple adapters = last one wins. Unsuitable for multi-tenant or multi-adapter scenarios.

3. **Async bridging required.** CrewAI calls hooks synchronously but `CrewAIAdapter` hooks are async (Edictum's guard is async). **Fix:** Bridge with `asyncio.run()` or `ThreadPoolExecutor` for nested event loops.

4. **Tool name mismatch.** `@crewai_tool("Query Clinical Data")` creates tools with human-readable names, but contracts use `snake_case` (`query_clinical_data`). **Fix:** Normalize `context.tool_name` before passing to adapter, restore after.

5. **PII redaction partially works.** Unlike OpenAI Agents SDK, CrewAI's `after_tool_call` hook CAN return a string to replace the tool result. The demo exploits this for PII redaction, but the behavior is underdocumented and may change.

6. **Verbose output.** `verbose=True` on Agent and Crew produces extensive output mixed with governance display. Hard to separate framework logging from demo output.

7. **Most expensive.** CrewAI repeats the full agent role, backstory, and task description in every prompt, resulting in ~2-3x the token usage of other frameworks for the same task.

8. **Generic denial messages.** When `before_hook` returns `False`, CrewAI shows `"Tool execution blocked by hook. Tool: Update Case Report"` without the specific governance reason. The reason is only in the audit trail. `False` is the only deny signal — no message channel.

9. **Tracing prompt blocks exit.** CrewAI prompts "Would you like to view your execution traces?" with a 20s timeout at the end of every run. This delays script exit in non-interactive mode.

**Workarounds applied:**
- `register_*_hook()` instead of `adapter.register()`
- `_run_async()` helper for sync-to-async bridging
- `TOOL_NAME_MAP` dict + `_normalize_tool_name()` / `_restore_tool_name()` around every hook call

---

## Cross-Adapter Comparison

| Feature | LangChain | OpenAI Agents | Agno | Semantic Kernel | CrewAI |
|---------|-----------|---------------|------|-----------------|--------|
| **PII redaction** | True interception | Side-effect only | True interception | True interception | Partial (undocumented) |
| **Deny tool calls** | Yes | Yes (reject_content) | Yes | Yes (terminate) | Yes (returns False) |
| **Adapter API works as-is** | Yes | **No** (as_guardrails broken) | Yes | Yes* | **No** (register fails) |
| **Callback result type** | ToolMessage | raw string | str | str (FunctionResult) | raw string |
| **Multiple adapters/process** | Yes | Yes | Yes | Yes (per kernel) | **No** (global hooks) |
| **Async/sync** | Sync wrapper | Async native | Sync (bridges) | Async native | Sync (bridges needed) |
| **Tool name normalization** | Not needed | Not needed | Not needed | Not needed** | **Required** |
| **Token tracking** | Reliable | Reliable | **Broken** | Under-reports | Reliable |
| **Workarounds needed** | 0 | 3 | 0 | 2 | 4 |
| **Relative cost** | 1x | 1.1x | N/A | 0.3-0.5x | 3x |

\* SK adapter works but the demo's chat loop needs careful handling of TOOL role messages.
\** SK adapter correctly strips plugin prefix internally.

---

## Edictum Adapter Bugs (to fix upstream)

### 1. OpenAI Agents SDK: `as_guardrails()` signature mismatch
**Severity:** Blocker — adapter is unusable without workaround.
**Root cause:** The `@tool_input_guardrail` / `@tool_output_guardrail` decorators create functions with signature `(context, agent, data)`, but the SDK's `ToolInputGuardrail.run()` calls `guardrail_function(data)` with 1 arg.
**Fix:** Change `as_guardrails()` to return `ToolInputGuardrail` / `ToolOutputGuardrail` objects with 1-arg functions matching the SDK's `run()` calling convention.

### 2. CrewAI: `register()` fails on bound methods
**Severity:** High — users must bypass the public API.
**Root cause:** CrewAI decorators call `setattr(func, marker, True)` which fails on bound methods.
**Fix:** Use `register_before_tool_call_hook()` / `register_after_tool_call_hook()` internally instead of decorators, or register plain functions.

### 3. Semantic Kernel: FunctionResult wrapping (FIXED)
**Severity:** Blocker — adapter crashes on deny and on postcondition remediation.
**Root cause:** Adapter assigned raw strings to `context.function_result`, but SK 1.39+ validates this field as `FunctionResult | None` via pydantic.
**Fix applied:** Added `_wrap_result(context, value)` helper in `edictum/adapters/semantic_kernel.py` that wraps values in `FunctionResult(function=context.function.metadata, value=value)`.

### 4. Semantic Kernel: Chat history corruption on deny
**Severity:** Medium — affects the demo pattern, not the adapter itself.
**Root cause:** When `context.terminate = True`, SK still adds a tool result to chat history. The next API call sees a `tool` message without a preceding `tool_calls` message.
**Mitigation:** Document that demos must filter `AuthorRole.TOOL` messages when building chat history manually.

---

## Recommendations

1. **For regulated environments requiring PII interception**, use LangChain, Agno, or Semantic Kernel. OpenAI Agents SDK and CrewAI can only log PII, not block it.

2. **For simplest integration**, use LangChain (zero workarounds) or Agno (zero workarounds, but no token tracking).

3. **For cost-sensitive deployments**, Semantic Kernel is cheapest (batches calls) but has integration complexity. CrewAI is the most expensive (~3x).

4. **The `on_postcondition_warn` callback API is consistent** across all adapters — `(result, findings) -> result` — but whether the return value is respected depends on the framework.

5. **Tool name normalization** must be documented for CrewAI users. SK handles it internally.

6. **Fix the OpenAI Agents adapter** — `as_guardrails()` is broken. This is the highest-priority upstream fix.
