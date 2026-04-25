"""Structured factual case packets for downstream assistant layers."""

from __future__ import annotations

from typing import Any, TypedDict

from fraud_assistant.db import get_transaction
from fraud_assistant.features import prepare_features
from fraud_assistant.model_service import score_transaction_features


DATASET_LIMITATIONS = [
    "PCA features are anonymized",
    "no customer identity",
    "no merchant data",
    "no location/device/IP data",
    "portfolio prototype only",
]


class CasePacket(TypedDict):
    transaction_id: int
    transaction: dict[str, Any]
    engineered_features: dict[str, float]
    model: dict[str, Any]
    dataset_limitations: list[str]


def build_case_packet(transaction_id: int) -> CasePacket:
    """Build a factual case packet from retrieved and computed fields only."""
    transaction = get_transaction(transaction_id)
    feature_frame = prepare_features(transaction)
    model_score = score_transaction_features(transaction)

    return {
        "transaction_id": transaction_id,
        "transaction": transaction,
        "engineered_features": {
            "Amount_log": float(feature_frame.loc[0, "Amount_log"]),
        },
        "model": model_score,
        "dataset_limitations": DATASET_LIMITATIONS.copy(),
    }
