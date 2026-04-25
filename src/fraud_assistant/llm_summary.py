"""Guarded LLM investigation summaries over deterministic case packets."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypedDict

from fraud_assistant.guardrails import REQUIRED_SUMMARY_FIELDS, validate_investigation_summary


class InvestigationSummary(TypedDict):
    summary: str
    model_decision_explanation: str
    key_observed_facts: list[str]
    limitations: list[str]
    suggested_follow_up_questions: list[str]
    analyst_note_draft: str


class LLMClient(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> Mapping[str, Any]:
        ...


def generate_investigation_summary(
    case_packet: Mapping[str, Any],
    llm_client: LLMClient | Callable[[list[dict[str, str]]], Mapping[str, Any]],
) -> InvestigationSummary:
    """Generate and validate a structured investigation summary from a case packet."""
    messages = build_summary_messages(case_packet)
    raw_output = llm_client(messages)
    output = _coerce_summary_output(raw_output)
    validate_investigation_summary(output, case_packet)
    return output


def build_summary_messages(case_packet: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build LLM messages from only the supplied case packet."""
    return [
        {
            "role": "system",
            "content": (
                "You are a guarded fraud investigation assistant. The classifier, not you, made the "
                "fraud decision. Use only the supplied case packet. Do not invent merchant, customer, "
                "cardholder, account, IP address, device, location, chargeback, AML, or KYC facts. "
                "Refer to V1-V28 only as anonymized PCA features; do not assign business meanings to them. "
                "Always include limitations."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "case_packet": case_packet,
                    "required_output_fields": REQUIRED_SUMMARY_FIELDS,
                },
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _coerce_summary_output(raw_output: Mapping[str, Any]) -> InvestigationSummary:
    if not isinstance(raw_output, Mapping):
        raise TypeError("LLM client must return a mapping")

    output = {field: raw_output.get(field) for field in REQUIRED_SUMMARY_FIELDS}
    return output  # type: ignore[return-value]
