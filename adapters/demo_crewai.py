"""
Edictum CrewAI Adapter Demo
============================

Demonstrates Edictum governance using the CrewAI adapter with global
before/after tool-call hooks. A CrewAI agent uses GPT-4.1 for
pharmacovigilance tasks while Edictum governs every tool call via hooks.

Usage:
    python adapters/demo_crewai.py
    python adapters/demo_crewai.py --mode observe
    python adapters/demo_crewai.py --role researcher
    python adapters/demo_crewai.py --role researcher --ticket CAPA-2025-042
"""

from __future__ import annotations

from edictum import Edictum
from edictum.adapters.crewai import CrewAIAdapter
from crewai import Agent, Task, Crew
from crewai.tools import tool as crewai_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from shared import (  # noqa: E402
    query_clinical_data as shared_query_clinical_data,
    update_case_report as shared_update_case_report,
    export_regulatory_document as shared_export_regulatory_document,
    search_medical_literature as shared_search_medical_literature,
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


# --- CrewAI tool wrappers ---------------------------------------------------

@crewai_tool("Query Clinical Data")
def query_clinical_data(dataset: str, query: str = "") -> str:
    """Query clinical trial databases. Available datasets: trial_summary, adverse_events_summary, adverse_events_detailed, patient_records, lab_results."""
    return shared_query_clinical_data(dataset, query)


@crewai_tool("Update Case Report")
def update_case_report(event_id: str, section: str, content: str) -> str:
    """Update a section of an adverse event case report."""
    return shared_update_case_report(event_id, section, content)


@crewai_tool("Export Regulatory Document")
def export_regulatory_document(document_type: str, trial_id: str, content: str) -> str:
    """Export a document for regulatory submission (e.g., safety narrative for IND/NDA)."""
    return shared_export_regulatory_document(document_type, trial_id, content)


@crewai_tool("Search Medical Literature")
def search_medical_literature(terms: str, max_results: int = 5) -> str:
    """Search medical literature for relevant publications."""
    return shared_search_medical_literature(terms, max_results)


# --- Main --------------------------------------------------------------------

def main():
    args = parse_args("CrewAI")
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

    # CrewAI adapter with postcondition-aware PII warning callback
    adapter = CrewAIAdapter(guard, principal=principal)

    pii_warnings: list[dict] = []

    def warn_callback(result, findings):
        pii_warnings.append({"findings": [f.message for f in findings]})
        if isinstance(result, str):
            redacted = redact_pii(result)
            print_governance("WARNING", "PII detected in tool output")
            return redacted
        return result

    # register() handles tool name normalization, async bridging, and
    # hook registration internally (all fixed in edictum 0.5.2).
    adapter.register(on_postcondition_warn=warn_callback)

    # CrewAI agent
    tools = [query_clinical_data, update_case_report, export_regulatory_document, search_medical_literature]

    agent = Agent(
        role="Pharmacovigilance Specialist",
        goal="Analyze clinical trial safety data and prepare regulatory reports",
        backstory=(
            "You are an experienced pharmacovigilance specialist with expertise "
            "in clinical trial safety data analysis, adverse event reporting, "
            "and regulatory submissions."
        ),
        tools=tools,
        llm="gpt-4.1",
        verbose=True,
    )

    crew_task = Task(
        description=task,
        agent=agent,
        expected_output="A comprehensive safety review with clinical assessment and regulatory narrative",
    )

    crew = Crew(agents=[agent], tasks=[crew_task], verbose=True)

    # Banner
    print_banner("CrewAI", principal, args.mode)
    print_header(f"TASK: {task}")
    print_event("Adapter", "CrewAIAdapter -> register(before_tool_call, after_tool_call)")
    print()

    # Run crew
    result = crew.kickoff()

    # Display result
    print_header("CREW RESULT")
    raw = str(result.raw) if hasattr(result, "raw") else str(result)
    print(f"  {raw[:500]}")
    if len(raw) > 500:
        print(f"  ... ({len(raw)} chars total)")

    # PII warnings
    if pii_warnings:
        print_header("PII WARNINGS")
        for i, w in enumerate(pii_warnings, 1):
            for finding in w["findings"]:
                print(f"  {i}. {finding}")

    # Known limitations
    print_header("KNOWN LIMITATIONS")
    print("  1. Global hooks: hooks are registered globally. Multiple")
    print("     adapters in the same process would conflict.")
    print("  2. PII redaction: after_hook returns redacted string so CrewAI")
    print("     replaces the tool result (undocumented, may change).")
    print()

    # Audit + token summaries
    print_audit_summary(sink)

    # Token tracking from CrewAI result
    input_tokens = 0
    output_tokens = 0
    llm_calls = 0
    usage = getattr(result, "token_usage", None)
    if usage:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        llm_calls = getattr(usage, "successful_requests", 0) or 0

    print_token_summary(input_tokens, output_tokens, llm_calls)

    print("  The agent was non-deterministic. The governance was not.")
    print("=" * 70)
    teardown_otel()


if __name__ == "__main__":
    main()
