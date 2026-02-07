"""
Edictum Semantic Kernel Adapter Demo
======================================

Demonstrates Edictum governance using the Semantic Kernel adapter with
kernel filters. The agent uses GPT-4.1 for pharmacovigilance tasks while
Edictum governs every tool call transparently via AUTO_FUNCTION_INVOCATION
filters.

NOTE: Semantic Kernel uses `pluginName-functionName` format for tool names
(e.g., `pharma-query_clinical_data`). The adapter registers a kernel filter
that intercepts these calls before/after execution, stripping the plugin
prefix when creating Edictum envelopes.

Usage:
    python adapters/demo_semantic_kernel.py
    python adapters/demo_semantic_kernel.py --mode observe
    python adapters/demo_semantic_kernel.py --role researcher
    python adapters/demo_semantic_kernel.py --role researcher --ticket CAPA-2025-042
"""

from __future__ import annotations

import asyncio

from edictum import Edictum
from edictum.adapters.semantic_kernel import SemanticKernelAdapter
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

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
    print_audit_summary,
    print_token_summary,
    setup_otel,
    teardown_otel,
)


# ─── Semantic Kernel plugin ─────────────────────────────────────────────────

class PharmaPlugin:
    """Wraps shared pharma tools as Semantic Kernel kernel functions."""

    @kernel_function(
        name="query_clinical_data",
        description=(
            "Query clinical trial databases. Available datasets: "
            "trial_summary, adverse_events_summary, adverse_events_detailed, "
            "patient_records, lab_results."
        ),
    )
    def query_clinical_data(self, dataset: str, query: str = "") -> str:
        return shared_query_clinical_data(dataset, query)

    @kernel_function(
        name="update_case_report",
        description="Update a section of an adverse event case report.",
    )
    def update_case_report(self, event_id: str, section: str, content: str) -> str:
        return shared_update_case_report(event_id, section, content)

    @kernel_function(
        name="export_regulatory_document",
        description="Export a document for regulatory submission.",
    )
    def export_regulatory_document(self, document_type: str, trial_id: str, content: str) -> str:
        return shared_export_regulatory_document(document_type, trial_id, content)

    @kernel_function(
        name="search_medical_literature",
        description="Search medical literature for relevant publications.",
    )
    def search_medical_literature(self, terms: str, max_results: int = 5) -> str:
        return shared_search_medical_literature(terms, max_results)


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args("Semantic Kernel")
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

    # Semantic Kernel setup
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4.1"))
    kernel.add_plugin(PharmaPlugin(), "pharma")

    # Edictum adapter — registers AUTO_FUNCTION_INVOCATION filter on kernel
    def redact_callback(result, findings):
        if isinstance(result, str):
            return redact_pii(result)
        return result

    adapter = SemanticKernelAdapter(guard, principal=principal)
    adapter.register(kernel, on_postcondition_warn=redact_callback)

    # Banner
    print_banner("Semantic Kernel", principal, args.mode)
    print_header(f"TASK: {task}")

    # Chat loop — SK needs iterative calls for multi-turn function calling
    chat_service = kernel.get_service("chat")
    settings = kernel.get_prompt_execution_settings_from_service_id("chat")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    chat_history = ChatHistory(system_message=SYSTEM_PROMPT)
    chat_history.add_user_message(task)

    total_input_tokens = 0
    total_output_tokens = 0
    llm_calls = 0

    while True:
        result = await chat_service.get_chat_message_content(
            chat_history, settings, kernel=kernel
        )

        # SK manages chat_history internally for tool call rounds.
        # Only add the message when it's the final assistant response
        # (not a tool result, which SK already appended during invoke).
        from semantic_kernel.contents import AuthorRole
        if result.role != AuthorRole.TOOL:
            chat_history.add_message(result)

        # Track tokens from result metadata
        if hasattr(result, "metadata") and result.metadata:
            usage = result.metadata.get("usage", None)
            if usage:
                llm_calls += 1
                total_input_tokens += getattr(usage, "prompt_tokens", 0)
                total_output_tokens += getattr(usage, "completion_tokens", 0)

        # Check if done (no more function calls pending)
        if not result.items or not any(
            hasattr(item, "function_name")
            for item in result.items
            if hasattr(item, "function_name")
        ):
            break

    # Display final response
    print_header("AGENT RESPONSE")
    final_text = str(result)
    print(f"  {final_text[:500]}")
    if len(final_text) > 500:
        print(f"  ... ({len(final_text)} chars total)")

    # Note about SK tool naming
    print()
    print("  NOTE: Semantic Kernel uses 'pluginName-functionName' format")
    print("  (e.g., pharma-query_clinical_data). The Edictum adapter")
    print("  intercepts calls via AUTO_FUNCTION_INVOCATION kernel filter.")

    # Summaries
    print_audit_summary(sink)
    print_token_summary(total_input_tokens, total_output_tokens, llm_calls)

    print("  The agent was non-deterministic. The governance was not.")
    print("=" * 70)
    teardown_otel()


if __name__ == "__main__":
    asyncio.run(main())
