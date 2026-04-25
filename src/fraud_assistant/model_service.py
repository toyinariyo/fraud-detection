"""Deterministic scoring with saved model artifacts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from fraud_assistant.config import MODEL_METADATA_PATH, MODEL_PATH, SCALER_PATH
from fraud_assistant.features import FEATURE_COLUMNS, prepare_features

THRESHOLD_SOURCE = "validation_frozen_metadata"


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(MODEL_METADATA_PATH)
    return model, scaler, metadata


def get_model_metadata() -> dict[str, Any]:
    """Return saved model metadata without recalculating any operating threshold."""
    _, _, metadata = _load_artifacts()
    return dict(metadata)


def score_transaction_features(transaction: dict[str, Any]) -> dict[str, Any]:
    """Score a transaction using the saved scaler, model, and frozen threshold."""
    model, scaler, metadata = _load_artifacts()
    feature_frame = prepare_features(transaction)
    feature_frame = feature_frame.loc[:, FEATURE_COLUMNS]

    scaled_features = scaler.transform(feature_frame)
    fraud_probability = float(model.predict_proba(scaled_features)[0, 1])
    threshold = float(metadata["threshold"])
    decision = "FLAG" if fraud_probability >= threshold else "APPROVE"

    return {
        "model_name": metadata.get("model_name", type(model).__name__),
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "decision": decision,
        "threshold_source": THRESHOLD_SOURCE,
    }
