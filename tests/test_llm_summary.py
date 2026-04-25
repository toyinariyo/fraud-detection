import json

import pytest

from fraud_assistant.llm_summary import build_summary_messages, generate_investigation_summary


def _case_packet():
    return {
        "transaction_id": 1,
        "transaction": {
            "transaction_id": 1,
            "Time": 0.0,
            "Amount": 10.0,
            "V14": -1.2,
        },
        "engineered_features": {"Amount_log": 2.3978952727983707},
        "model": {
            "model_name": "XGBoost",
            "fraud_probability": 0.1,
            "threshold": 0.9537375569343567,
            "decision": "APPROVE",
            "threshold_source": "validation_frozen_metadata",
        },
        "dataset_limitations": [
            "PCA features are anonymized",
            "no customer identity",
            "no merchant data",
            "no location/device/IP data",
            "portfolio prototype only",
        ],
    }


def _valid_llm_output():
    return {
        "summary": "The classifier decision is APPROVE for transaction 1.",
        "model_decision_explanation": (
            "The model decision used XGBoost probability 0.100000 against the frozen threshold."
        ),
        "key_observed_facts": ["Amount is 10.0.", "V14 is an anonymized PCA feature."],
        "limitations": [
            "No customer identity is available in this dataset.",
            "Merchant data is not available.",
            "Device, IP, and location data are not present in the case packet.",
        ],
        "suggested_follow_up_questions": ["Is there external analyst context available outside this dataset?"],
        "analyst_note_draft": "The classifier made the decision; this note summarizes the case packet only.",
    }


def test_build_summary_messages_use_only_case_packet_payload():
    packet = _case_packet()
    messages = build_summary_messages(packet)
    user_payload = json.loads(messages[1]["content"])

    assert user_payload["case_packet"] == packet
    assert "best_model.pkl" not in messages[1]["content"]
    assert "sqlite" not in messages[1]["content"].lower()


def test_generate_investigation_summary_with_fake_client_returns_required_fields():
    def fake_client(messages):
        assert "case_packet" in messages[1]["content"]
        return _valid_llm_output()

    output = generate_investigation_summary(_case_packet(), fake_client)

    assert output["summary"]
    assert output["model_decision_explanation"]
    assert output["key_observed_facts"]
    assert output["limitations"]
    assert output["suggested_follow_up_questions"]
    assert output["analyst_note_draft"]


def test_generate_investigation_summary_requires_classifier_decision_attribution():
    def fake_client(_messages):
        output = _valid_llm_output()
        output["summary"] = "The assistant decision is APPROVE for transaction 1."
        output["model_decision_explanation"] = "The assistant used the packet."
        output["analyst_note_draft"] = "Summary only."
        return output

    with pytest.raises(ValueError, match="classifier/model made the fraud decision"):
        generate_investigation_summary(_case_packet(), fake_client)


def test_generate_investigation_summary_rejects_missing_limitations():
    def fake_client(_messages):
        output = _valid_llm_output()
        output["limitations"] = []
        return output

    with pytest.raises(ValueError, match="must include limitations"):
        generate_investigation_summary(_case_packet(), fake_client)


def test_generate_investigation_summary_rejects_invented_business_claims():
    def fake_client(_messages):
        output = _valid_llm_output()
        output["summary"] = "The classifier decision is APPROVE. The merchant appears high-risk."
        return output

    with pytest.raises(ValueError, match="Unsupported claim"):
        generate_investigation_summary(_case_packet(), fake_client)
