"""
Edictum Agno Adapter Demo
==========================

Demonstrates Edictum governance using the Agno adapter with an Agno Agent.
The agent uses GPT-4.1 for pharmacovigilance tasks while Edictum governs
every tool call transparently via a wrap-around tool_hook.

Usage:
    python adapters/demo_agno.py
    python adapters/demo_agno.py --mode observe
    python adapters/demo_agno.py --role researcher
    python adapters/demo_agno.py --role researcher --ticket CAPA-2025-042
"""

from __future__ import annotations

from edictum import Edictum
from edictum.adapters.agno import AgnoAdapter
from agno.agent import Agent
from agno.models.openai import OpenAIChat

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from shared import (  # noqa: E402
    query_clinical_data,
    update_case_report,
    export_regulatory_document,
    search_medical_literature,
    redact_pii,
    CollectingAuditSink,
    CONTRACTS_PATH,
    SYSTEM_PROMPT,
    DEFAULT_TASK,
    parse_args,
    make_principal,
    print_banner,
    print_header,
    print_event,
    print_audit_summary,
    print_token_summary,
    setup_otel,
    teardown_otel,
)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args("Agno")
    principal = make_principal(args.role, args.ticket)
    task = args.task or DEFAULT_TASK
    setup_otel()

    # Audit sink
    sink = CollectingAuditSink()

    # Governance guard
    guard = Edictum.from_yaml(
        str(CONTRACTS_PATH),
        mode="observe" if args.mode == "observe" else None,
        audit_sink=sink,
    )

    # Agno adapter with postcondition-aware PII redaction
    adapter = AgnoAdapter(guard, principal=principal)

    def redact_callback(result, findings):
        if isinstance(result, str):
            result = redact_pii(result)
        return result

    hook = adapter.as_tool_hook(on_postcondition_warn=redact_callback)

    # Create Agno agent — plain functions are auto-wrapped by Agno
    agent = Agent(
        model=OpenAIChat(id="gpt-4.1"),
        tools=[query_clinical_data, update_case_report, export_regulatory_document, search_medical_literature],
        tool_hooks=[hook],
        instructions=[SYSTEM_PROMPT],
    )

    # Banner
    print_banner("Agno", principal, args.mode)
    print_header(f"TASK: {task}")

    # Run agent (Agno's agent.run() is sync)
    response = agent.run(task)

    # Display response
    print_header("AGENT RESPONSE")
    content = response.content if hasattr(response, "content") else str(response)
    print(f"  {content[:500]}")
    if len(content) > 500:
        print(f"  ... ({len(content)} chars total)")

    # Token tracking — extract from response metrics if available
    input_tokens = 0
    output_tokens = 0
    llm_calls = 0

    metrics = getattr(response, "metrics", None)
    if metrics and isinstance(metrics, dict):
        input_tokens = metrics.get("input_tokens", 0) or metrics.get("prompt_tokens", 0) or 0
        output_tokens = metrics.get("output_tokens", 0) or metrics.get("completion_tokens", 0) or 0
        llm_calls = metrics.get("llm_calls", 0) or 1

    # If metrics not on response directly, check messages
    if input_tokens == 0 and hasattr(response, "messages"):
        for msg in response.messages:
            msg_metrics = getattr(msg, "metrics", None)
            if msg_metrics and isinstance(msg_metrics, dict):
                llm_calls += 1
                input_tokens += msg_metrics.get("input_tokens", 0) or msg_metrics.get("prompt_tokens", 0) or 0
                output_tokens += msg_metrics.get("output_tokens", 0) or msg_metrics.get("completion_tokens", 0) or 0

    # Summaries
    print_audit_summary(sink)
    print_token_summary(input_tokens, output_tokens, llm_calls)

    print("  The agent was non-deterministic. The governance was not.")
    print("=" * 70)
    teardown_otel()


if __name__ == "__main__":
    main()
