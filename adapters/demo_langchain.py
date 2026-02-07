"""
Edictum LangChain + LangGraph Adapter Demo
============================================

Demonstrates Edictum governance using the LangChain adapter with a LangGraph
ReAct agent. The agent uses GPT-4.1 for pharmacovigilance tasks while Edictum
governs every tool call transparently.

Usage:
    python adapters/demo_langchain.py
    python adapters/demo_langchain.py --mode observe
    python adapters/demo_langchain.py --role researcher
    python adapters/demo_langchain.py --role researcher --ticket CAPA-2025-042
"""

from __future__ import annotations

import asyncio
import json

from edictum import Edictum
from edictum.adapters.langchain import LangChainAdapter
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, ToolNode

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from shared import (  # noqa: E402
    query_clinical_data as _query_clinical_data,
    update_case_report as _update_case_report,
    export_regulatory_document as _export_regulatory_document,
    search_medical_literature as _search_medical_literature,
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
    print_governance,
    print_audit_summary,
    print_token_summary,
    setup_otel,
    teardown_otel,
)


# ─── LangChain tool wrappers ─────────────────────────────────────────────────

@tool
def query_clinical_data(dataset: str, query: str = "") -> str:
    """Query clinical trial databases. Available datasets: trial_summary, adverse_events_summary, adverse_events_detailed, patient_records, lab_results."""
    return _query_clinical_data(dataset, query)


@tool
def update_case_report(event_id: str, section: str, content: str) -> str:
    """Update a section of an adverse event case report."""
    return _update_case_report(event_id, section, content)


@tool
def export_regulatory_document(document_type: str, trial_id: str, content: str) -> str:
    """Export a document for regulatory submission (e.g., safety narrative for IND/NDA)."""
    return _export_regulatory_document(document_type, trial_id, content)


@tool
def search_medical_literature(terms: str, max_results: int = 5) -> str:
    """Search medical literature for relevant publications."""
    return _search_medical_literature(terms, max_results)


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args("LangChain")
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

    # LangChain adapter with postcondition-aware PII redaction
    adapter = LangChainAdapter(guard, principal=principal)

    def redact_callback(result, findings):
        if hasattr(result, 'content') and isinstance(result.content, str):
            result.content = redact_pii(result.content)
        return result

    tools = [query_clinical_data, update_case_report, export_regulatory_document, search_medical_literature]
    tool_node = ToolNode(
        tools=tools,
        wrap_tool_call=adapter.as_tool_wrapper(on_postcondition_warn=redact_callback),
    )

    llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
    agent = create_react_agent(llm, tools=tool_node, prompt=SYSTEM_PROMPT)

    # Banner
    print_banner("LangChain + LangGraph", principal, args.mode)
    print_header(f"TASK: {task}")

    # Run agent
    result = agent.invoke({"messages": [("user", task)]})

    # Token tracking — count ALL AI messages (including tool-calling ones)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    llm_calls = 0

    # Display results
    for msg in result["messages"]:
        # Track tokens from every AI message
        if getattr(msg, 'type', None) == 'ai':
            usage = getattr(msg, 'usage_metadata', None)
            if usage:
                llm_calls += 1
                total_prompt_tokens += usage.get('input_tokens', 0)
                total_completion_tokens += usage.get('output_tokens', 0)

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n  Tool call: {tc['name']}({json.dumps(tc['args'], separators=(',', ':'))})")
        elif hasattr(msg, 'content') and hasattr(msg, 'tool_call_id'):
            # ToolMessage
            if msg.content.startswith("DENIED:"):
                print_governance("DENIED", msg.content[8:])
            elif '[REDACTED]' in msg.content:
                print_governance("WARNING", "PII detected — output redacted")
                if len(msg.content) > 200:
                    print_event("Result", f"{msg.content[:200]}...", "  ")
                else:
                    print_event("Result", msg.content, "  ")
            else:
                print_governance("ALLOWED", "executed successfully")
                if len(msg.content) > 200:
                    print_event("Result", f"{msg.content[:200]}...", "  ")
                else:
                    print_event("Result", msg.content, "  ")
        elif getattr(msg, 'type', None) == 'ai' and not getattr(msg, 'tool_calls', None) and msg.content:
            print_header("AGENT RESPONSE")
            print(f"  {msg.content[:500]}")
            if len(msg.content) > 500:
                print(f"  ... ({len(msg.content)} chars total)")

    # Summaries
    print_audit_summary(sink)
    print_token_summary(total_prompt_tokens, total_completion_tokens, llm_calls)

    print("  The agent was non-deterministic. The governance was not.")
    print("=" * 70)
    teardown_otel()


if __name__ == "__main__":
    asyncio.run(main())
