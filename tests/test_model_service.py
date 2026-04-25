import joblib
import pytest

from fraud_assistant.config import MODEL_METADATA_PATH
from fraud_assistant.db import get_transaction
from fraud_assistant.model_service import get_model_metadata, score_transaction_features


def test_model_threshold_loaded_from_metadata():
    saved_metadata = joblib.load(MODEL_METADATA_PATH)

    assert get_model_metadata()["threshold"] == saved_metadata["threshold"]


def test_score_output_contains_probability_threshold_and_decision():
    transaction = get_transaction(1)
    score = score_transaction_features(transaction)

    assert score["model_name"]
    assert isinstance(score["fraud_probability"], float)
    assert 0.0 <= score["fraud_probability"] <= 1.0
    assert isinstance(score["threshold"], float)
    assert score["decision"] in {"FLAG", "APPROVE"}
    assert score["threshold_source"] == "validation_frozen_metadata"


def test_invalid_transaction_id_fails_cleanly():
    with pytest.raises(LookupError, match="Transaction ID -1 not found"):
        get_transaction(-1)
