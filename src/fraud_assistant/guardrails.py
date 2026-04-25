"""Rule-based guardrails for LLM investigation summaries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


REQUIRED_SUMMARY_FIELDS = [
    "summary",
    "model_decision_explanation",
    "key_observed_facts",
    "limitations",
    "suggested_follow_up_questions",
    "analyst_note_draft",
]

RESTRICTED_TERMS = {
    "merchant": ["merchant"],
    "customer": ["customer"],
    "ip": ["ip", "ip address"],
    "device": ["device"],
    "location": ["location"],
    "chargeback": ["chargeback", "chargebacks"],
    "aml": ["aml"],
    "kyc": ["kyc"],
    "cardholder": ["cardholder"],
    "account": ["account", "account history"],
}

ABSENCE_PATTERNS = [
    r"\bno\b",
    r"\bnot available\b",
    r"\bunavailable\b",
    r"\babsent\b",
    r"\bnot present\b",
    r"\bnot included\b",
    r"\bnot in (?:the )?(?:dataset|case packet)\b",
    r"\bnot part of (?:the )?(?:dataset|case packet)\b",
    r"\bwithout\b",
    r"\black(?:s|ing)?\b",
]

PCA_MEANING_PATTERN = re.compile(
    r"\bV(?:[1-9]|1[0-9]|2[0-8])\b.{0,40}\b(?:means|represents|indicates|corresponds to|maps to)\b",
    re.IGNORECASE,
)

CLASSIFIER_DECISION_PATTERN = re.compile(
    r"\b(?:classifier|model)\b.{0,80}\b(?:decision|decided|flag|approve|approved|score|probability)\b",
    re.IGNORECASE,
)


def validate_investigation_summary(output: Mapping[str, Any], case_packet: Mapping[str, Any]) -> None:
    """Validate an LLM summary against the factual case packet."""
    _validate_required_fields(output)
    _validate_limitations(output)
    _validate_classifier_decision_attribution(output)

    for sentence in _iter_sentences(_flatten_output_text(output)):
        _validate_pca_claim(sentence)
        _validate_restricted_terms(sentence, case_packet)


def _validate_required_fields(output: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if field not in output]
    if missing:
        raise ValueError(f"LLM summary missing required fields: {', '.join(missing)}")


def _validate_limitations(output: Mapping[str, Any]) -> None:
    limitations = output.get("limitations")
    if not limitations:
        raise ValueError("LLM summary must include limitations")
    if isinstance(limitations, str) and not limitations.strip():
        raise ValueError("LLM summary must include limitations")


def _validate_classifier_decision_attribution(output: Mapping[str, Any]) -> None:
    combined = " ".join(
        str(output.get(field, "")) for field in ("summary", "model_decision_explanation", "analyst_note_draft")
    )
    if not CLASSIFIER_DECISION_PATTERN.search(combined):
        raise ValueError("LLM summary must state that the classifier/model made the fraud decision")


def _validate_pca_claim(sentence: str) -> None:
    if PCA_MEANING_PATTERN.search(sentence):
        raise ValueError(f"Unsupported business meaning assigned to anonymized PCA feature: {sentence}")


def _validate_restricted_terms(sentence: str, case_packet: Mapping[str, Any]) -> None:
    lowered = sentence.lower()
    if _is_absence_statement(lowered):
        return

    for category, terms in RESTRICTED_TERMS.items():
        if any(_contains_term(lowered, term) for term in terms) and not _packet_has_fact(case_packet, terms):
            raise ValueError(f"Unsupported claim about {category}: {sentence}")


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None


def _is_absence_statement(sentence: str) -> bool:
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in ABSENCE_PATTERNS)


def _packet_has_fact(value: Any, terms: list[str]) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower().replace("_", " ")
            if any(_contains_term(key_text, term) for term in terms) and item not in (None, "", [], {}):
                return True
            if _packet_has_fact(item, terms):
                return True
    elif isinstance(value, list):
        return any(_packet_has_fact(item, terms) for item in value)
    return False


def _flatten_output_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_output_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_output_text(item) for item in value)
    return str(value)


def _iter_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
