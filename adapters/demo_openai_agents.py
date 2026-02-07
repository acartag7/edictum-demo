"""
Edictum OpenAI Agents SDK Adapter Demo
=======================================

Demonstrates Edictum governance using the OpenAI Agents SDK adapter. The agent
uses GPT-4.1 for pharmacovigilance tasks while Edictum governs every tool call
via tool-level input/output guardrails.

Usage:
    python adapters/demo_openai_agents.py
    python adapters/demo_openai_agents.py --mode observe
    python adapters/demo_openai_agents.py --role researcher
    python adapters/demo_openai_agents.py --role researcher --ticket CAPA-2025-042
"""

from __future__ import annotations

import asyncio

from edictum import Edictum
from edictum.adapters.openai_agents import OpenAIAgentsAdapter
from agents import Agent, Runner, function_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from shared import (  # noqa: E402
    query_clinical_data as _query_clinical_data,
    update_case_report as _update_case_report,
    export_regulatory_document as _export_regulatory_document,
    search_medical_literature as _search_medical_literature,
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


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args("OpenAI Agents SDK")
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

    # OpenAI Agents SDK adapter
    adapter = OpenAIAgentsAdapter(guard, principal=principal)

    # Warn callback for PII detection (side-effect only — cannot redact)
    pii_warnings: list[dict] = []

    def warn_callback(result, findings):
        pii_warnings.append({
            "result": str(result)[:100],
            "findings": [f.message for f in findings],
        })
        return result

    # as_guardrails() returns (ToolInputGuardrail, ToolOutputGuardrail) ready
    # to attach to individual @function_tool definitions.
    input_gr, output_gr = adapter.as_guardrails(on_postcondition_warn=warn_callback)

    # Define tools with tool-level guardrails
    # Tool guardrails (ToolInputGuardrail/ToolOutputGuardrail) go on each tool,
    # not on Agent(input_guardrails=...) which expects InputGuardrail objects.

    @function_tool(tool_input_guardrails=[input_gr], tool_output_guardrails=[output_gr])
    def query_clinical_data(dataset: str, query: str = "") -> str:
        """Query clinical trial databases. Available datasets: trial_summary, adverse_events_summary, adverse_events_detailed, patient_records, lab_results."""
        return _query_clinical_data(dataset, query)

    @function_tool(tool_input_guardrails=[input_gr], tool_output_guardrails=[output_gr])
    def update_case_report(event_id: str, section: str, content: str) -> str:
        """Update a section of an adverse event case report."""
        return _update_case_report(event_id, section, content)

    @function_tool(tool_input_guardrails=[input_gr], tool_output_guardrails=[output_gr])
    def export_regulatory_document(document_type: str, trial_id: str, content: str) -> str:
        """Export a document for regulatory submission (e.g., safety narrative for IND/NDA)."""
        return _export_regulatory_document(document_type, trial_id, content)

    @function_tool(tool_input_guardrails=[input_gr], tool_output_guardrails=[output_gr])
    def search_medical_literature(terms: str, max_results: int = 5) -> str:
        """Search medical literature for relevant publications."""
        return _search_medical_literature(terms, max_results)

    # Create agent (no agent-level guardrails — guardrails are on tools)
    agent = Agent(
        name="pharma-agent",
        instructions=SYSTEM_PROMPT,
        tools=[query_clinical_data, update_case_report, export_regulatory_document, search_medical_literature],
    )

    # Banner
    print_banner("OpenAI Agents SDK", principal, args.mode)
    print_header(f"TASK: {task}")

    # Run agent
    try:
        result = await Runner.run(agent, task)
    except Exception as e:
        exc_name = type(e).__name__
        if "Tripwire" in exc_name or "Guardrail" in exc_name:
            print_governance("DENIED", f"Guardrail blocked execution: {e}")
            print_audit_summary(sink)
            print("=" * 70)
            return
        raise

    # Display final output
    print_header("AGENT RESPONSE")
    output = result.final_output
    print(f"  {output[:500]}")
    if len(output) > 500:
        print(f"  ... ({len(output)} chars total)")

    # PII warnings
    if pii_warnings:
        print_header("PII WARNINGS")
        for w in pii_warnings:
            print_governance("WARNING", "PII/PHI detected in tool output")
            for finding in w["findings"]:
                print_event("Finding", finding, "  ")
        print()
        print(
            "  NOTE: OpenAI Agents SDK output guardrail is side-effect only --\n"
            "  PII was detected but cannot be redacted from the tool result\n"
            "  before the LLM sees it."
        )

    # Token tracking via context_wrapper.usage
    usage = result.context_wrapper.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    llm_calls = usage.requests

    # Summaries
    print_audit_summary(sink)
    print_token_summary(input_tokens, output_tokens, llm_calls)

    print("  The agent was non-deterministic. The governance was not.")
    print("=" * 70)
    teardown_otel()


if __name__ == "__main__":
    asyncio.run(main())
