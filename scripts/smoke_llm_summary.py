"""Build and print a guarded fake-LLM investigation summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_assistant.case_packets import build_case_packet  # noqa: E402
from fraud_assistant.llm_summary import generate_investigation_summary  # noqa: E402


def fake_llm_client(_messages: list[dict[str, str]], case_packet: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic, factual response shaped like an LLM summary."""
    model = case_packet["model"]
    transaction = case_packet["transaction"]
    transaction_id = case_packet["transaction_id"]

    exact_missing_data_limitation = (
        "Merchant, customer, device, IP, location, and account-history data is not available."
    )

    return {
        "summary": (
            f"The classifier decision is {model['decision']} for transaction {transaction_id}. "
            "This summary uses only the retrieved case packet and deterministic model output."
        ),
        "model_decision_explanation": (
            f"The {model['model_name']} classifier produced fraud_probability "
            f"{model['fraud_probability']:.6f}. The frozen validation threshold is "
            f"{model['threshold']:.6f}, so the classifier decision is {model['decision']}."
        ),
        "key_observed_facts": [
            f"Transaction ID is {transaction_id}.",
            f"Transaction amount is {transaction['Amount']}.",
            f"Transaction time offset is {transaction['Time']}.",
            f"The model threshold source is {model['threshold_source']}.",
            "V1-V28 are anonymized PCA features with no business labels in this dataset.",
        ],
        "limitations": [
            "PCA features are anonymized.",
            exact_missing_data_limitation,
            "This is a portfolio prototype only.",
        ],
        "suggested_follow_up_questions": [
            "Are there external analyst notes available outside this dataset?",
            "Is there approved operational context that can be reviewed separately?",
        ],
        "analyst_note_draft": (
            f"The classifier made an {model['decision']} decision for transaction {transaction_id}. "
            f"The fraud probability was {model['fraud_probability']:.6f} against a frozen threshold "
            f"of {model['threshold']:.6f}. {exact_missing_data_limitation}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test guarded fake-LLM summary generation.")
    parser.add_argument("transaction_id", nargs="?", type=int, default=1)
    args = parser.parse_args()

    packet = build_case_packet(args.transaction_id)
    summary = generate_investigation_summary(
        packet,
        lambda messages: fake_llm_client(messages, packet),
    )

    print(f"summary: {summary['summary']}")
    print(f"model_decision_explanation: {summary['model_decision_explanation']}")

    print("key_observed_facts:")
    for fact in summary["key_observed_facts"]:
        print(f"- {fact}")

    print("limitations:")
    for limitation in summary["limitations"]:
        print(f"- {limitation}")

    print("suggested_follow_up_questions:")
    for question in summary["suggested_follow_up_questions"]:
        print(f"- {question}")

    print(f"analyst_note_draft: {summary['analyst_note_draft']}")


if __name__ == "__main__":
    main()
