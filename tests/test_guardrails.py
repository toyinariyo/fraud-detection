import pytest

from fraud_assistant.guardrails import validate_investigation_summary


def _case_packet(**extra_transaction_fields):
    transaction = {
        "transaction_id": 1,
        "Time": 0.0,
        "Amount": 10.0,
        **extra_transaction_fields,
    }
    return {
        "transaction_id": 1,
        "transaction": transaction,
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


def _valid_output(**overrides):
    output = {
        "summary": "The classifier decision is APPROVE for transaction 1.",
        "model_decision_explanation": (
            "The model decision is based on a fraud probability of 0.100000 and the frozen threshold."
        ),
        "key_observed_facts": ["Amount is 10.0.", "V14 is an anonymized PCA feature."],
        "limitations": [
            "No customer identity is available in this dataset.",
            "Merchant data is not available.",
            "Device, IP, and location data are not present in the case packet.",
            "Account history is not available.",
        ],
        "suggested_follow_up_questions": ["Can an analyst review any external case notes if available?"],
        "analyst_note_draft": "The classifier made the decision; the assistant only summarizes packet facts.",
    }
    output.update(overrides)
    return output


def test_guardrails_accept_neutral_summary_and_absence_statements():
    validate_investigation_summary(_valid_output(), _case_packet())


@pytest.mark.parametrize(
    "claim",
    [
        "The customer has a suspicious history.",
        "The merchant appears high-risk.",
        "The IP address suggests fraud.",
        "The device fingerprint is suspicious.",
        "The location pattern is risky.",
        "This transaction has a prior chargeback.",
        "This case raises AML concerns.",
        "This case raises KYC concerns.",
        "The cardholder appears suspicious.",
        "The account history is suspicious.",
    ],
)
def test_guardrails_reject_unsupported_positive_restricted_claims(claim):
    output = _valid_output(summary=f"The classifier decision is APPROVE. {claim}")

    with pytest.raises(ValueError, match="Unsupported claim"):
        validate_investigation_summary(output, _case_packet())


def test_guardrails_allow_restricted_terms_when_fact_exists_in_packet():
    output = _valid_output(summary="The classifier decision is APPROVE. Merchant category is grocery.")

    validate_investigation_summary(output, _case_packet(merchant_category="grocery"))


def test_guardrails_reject_fake_pca_business_meanings():
    output = _valid_output(key_observed_facts=["V14 means merchant risk."])

    with pytest.raises(ValueError, match="Unsupported business meaning"):
        validate_investigation_summary(output, _case_packet())


def test_guardrails_require_limitations():
    output = _valid_output(limitations=[])

    with pytest.raises(ValueError, match="must include limitations"):
        validate_investigation_summary(output, _case_packet())
