# Adapter Insights — Lessons from the Claude Agent SDK Integration

Findings from building the `ClaudeAgentSDKAdapter` and its hooks bridge.
Use these when scoping adapters for other frameworks.

---

## 1. Understand the SDK's Execution Model

Every SDK owns the agent loop differently. This determines *where* your adapter can intercept.

| Model | Example | Integration point |
|-------|---------|-------------------|
| **You own the loop** | OpenAI SDK, LangChain | You call `guard.run()` yourself — wraps each tool execution |
| **SDK owns the loop** | Claude Agent SDK | You provide hooks/callbacks — the SDK calls you |

**Why it matters**: When you own the loop, the adapter is trivial — `guard.run()` wraps whatever executor you already have. When the SDK owns the loop, you need to understand the hook lifecycle, callback signatures, and return value semantics to inject governance correctly.

**Scoping question**: *Who calls the tool executor — your code or the SDK?*

---

## 2. Map the Hook Points and Their Signatures

The adapter must translate between CallGuard's interface and the SDK's hook interface. These rarely match.

**Claude Agent SDK expects:**
```python
async def pre_tool_use(input_data: TypedDict, tool_use_id: str, context) -> HookOutput
async def post_tool_use(input_data: TypedDict, tool_use_id: str, context) -> HookOutput
```

**CallGuard adapter exposes:**
```python
async def pre(tool_name, tool_input, tool_use_id) -> dict
async def post(tool_use_id, tool_response) -> dict
```

The bridge is small but *must exist*. Key friction points:
- **Parameter names differ** (`input_data["tool_name"]` vs positional `tool_name`)
- **Return types differ** (SDK expects `HookOutput` or `dict` with specific keys)
- **Context objects** — SDKs pass context/state you may need to thread through

**Scoping question**: *What are the exact hook signatures? What do the return values mean — especially for "block this call"?*

---

## 3. Understand the SDK's Communication Lifecycle

This was the hardest bug. The Claude Agent SDK communicates with hook callbacks over the same bidirectional stream as the prompt. A plain string prompt closes the stream immediately after being consumed — hooks then fail with "Stream closed".

**Fix**: Use an `AsyncIterable` prompt that yields the user message then stays alive (via `asyncio.Event().wait()`) until the agent finishes.

```python
class StreamingPrompt:
    def __init__(self, text: str):
        self._text = text
        self._done = asyncio.Event()

    def finish(self):
        self._done.set()

    async def _generate(self):
        yield {"type": "user", "message": {...}, ...}
        await self._done.wait()  # keep stream alive
```

**Lesson**: Hooks don't just need the right *signature* — they need the right *transport*. If the SDK multiplexes hook communication over an existing channel, you must keep that channel alive.

**Scoping question**: *How does the SDK deliver hook invocations? Separate channel? Multiplexed? What keeps the channel alive?*

---

## 4. Account for the SDK's Own Security Boundaries

The Claude Agent SDK has its own sandboxing that operates **independently** of CallGuard:

- Restricts Bash to the session's working directory
- Blocks `mkdir`, `mv`, and other file operations even within the working dir
- Blocks access to `/tmp/` and other system paths
- Parallel tool calls: if one is denied, siblings get `"Sibling tool call errored"`

CallGuard allowed the call, but the SDK blocked it downstream. Two security layers, no coordination between them.

**Implications for adapters**:
- Your governance layer and the SDK's built-in safety may overlap or conflict
- A "denied" result might come from CallGuard *or* the SDK — users need to distinguish them
- Contracts referencing specific paths (like `require_target_dir`) must be parameterized for the SDK's constraints
- Error messages from the SDK's sandbox look different from CallGuard denials

**Scoping question**: *What security/sandboxing does the SDK already enforce? Where will it overlap with or shadow CallGuard's governance?*

---

## 5. Map Feature Coverage Per Integration Point

Not all integration points support all CallGuard features:

| Feature | `guard.run()` (own loop) | Hooks (SDK loop) | `can_use_tool` callback |
|---------|:---:|:---:|:---:|
| Preconditions | Y | Y | Y |
| Postconditions | Y | Y | N |
| Session contracts | Y | Y | Y |
| Audit trail | Y | Y | N |
| OTel spans | Y | Y | N |
| Observe mode | Y | Y | N |
| Operation limits | Y | Y | Partial |
| Redaction | Y | Y | N |

`can_use_tool` is simpler to integrate (one callback, no stream lifecycle issues) but loses postconditions, audit, OTel, and observe mode. Hooks give full feature parity but require deeper integration work.

**Scoping question**: *Which CallGuard features does the target SDK's integration point support? Is there a simpler fallback path with known trade-offs?*

---

## 6. Parallel Tool Calls Create Cascading Failures

When an SDK sends multiple tool calls in one turn (parallel execution), a denial on one can cascade:

```
cat .env          → DENIED by CallGuard (sensitive_reads)
cat app.py        → "Sibling tool call errored" (SDK killed it)
cat config.json   → "Sibling tool call errored" (SDK killed it)
...
```

The agent loses results for *all* sibling calls, not just the denied one. It must then retry them individually — burning turns and tool call budget.

**Implications**:
- Session contracts counting tool calls see inflated numbers
- Audit trails record the denial but not the sibling cascade
- The adapter should document this behavior so contract authors can account for it

**Scoping question**: *How does the SDK handle partial failures in parallel tool calls? Does it retry, cancel siblings, or surface errors individually?*

---

## 7. The Bridge Should Live in the Adapter

Currently, `make_sdk_hooks()` is a standalone function that translates between the adapter's hook format and the SDK's `HookMatcher` format. This should be absorbed into the adapter:

```python
# Current (two steps):
adapter = ClaudeAgentSDKAdapter(guard)
hooks = make_sdk_hooks(adapter)  # manual bridge

# Goal (one step):
adapter = ClaudeAgentSDKAdapter(guard)
hooks = adapter.to_sdk_hooks()  # returns SDK-ready HookMatchers
```

More broadly: the adapter's public API should return objects the SDK can consume directly. No external bridge functions, no signature translation in user code. The whole point of the adapter is to absorb this complexity.

**Scoping question**: *What is the minimum user-facing API? Can the adapter return SDK-native objects directly?*

---

## 8. Contracts Must Be Environment-Aware

The `require_target_dir` contract hardcoded `/tmp/` as the allowed base path. This broke when the SDK's sandbox forced us to use local paths. Fix: make contracts configurable via factory functions.

```python
# Factory: works for any environment
def make_require_target_dir(base="/tmp/"):
    @precondition("Bash")
    def require_target_dir(envelope):
        ...target.startswith(base)...
    return require_target_dir

# OpenAI SDK demos (subprocess, full /tmp/ access)
require_target_dir = make_require_target_dir("/tmp/")

# Claude Agent SDK (sandboxed to working dir)
require_target_dir = make_require_target_dir("./")
```

**Lesson**: The same governance logic applies everywhere, but the *parameters* change based on the runtime environment. Adapters should document what environmental constraints they impose so contract authors can adjust.

**Scoping question**: *What constraints does the target SDK impose that would affect existing contracts?*

---

## Summary Checklist for New Adapters

When scoping a new adapter, answer these questions:

1. **Execution model** — Who owns the agent loop? You or the SDK?
2. **Hook signatures** — What are the exact callback signatures and return types?
3. **Communication lifecycle** — How are hooks invoked? What keeps the channel alive?
4. **SDK security boundaries** — What does the SDK sandbox/block on its own?
5. **Feature coverage** — Which integration point gives full feature parity?
6. **Parallel call behavior** — How does the SDK handle partial failures?
7. **Adapter API surface** — Can we return SDK-native objects from the adapter?
8. **Contract compatibility** — What environment constraints affect existing contracts?
