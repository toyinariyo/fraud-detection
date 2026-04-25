import math

import numpy as np

from fraud_assistant.features import FEATURE_COLUMNS, prepare_features


def _sample_transaction():
    transaction = {f"V{i}": float(i) for i in range(1, 29)}
    transaction.update(
        {
            "transaction_id": 123,
            "Time": 42.0,
            "Amount": 99.5,
            "is_fraud": 1,
        }
    )
    return transaction


def test_feature_columns_are_exact_model_order():
    assert FEATURE_COLUMNS == [f"V{i}" for i in range(1, 29)] + ["Amount_log"]


def test_prepare_features_calculates_amount_log():
    features = prepare_features(_sample_transaction())

    assert math.isclose(features.loc[0, "Amount_log"], np.log1p(99.5))


def test_prepare_features_excludes_non_model_fields():
    features = prepare_features(_sample_transaction())

    assert list(features.columns) == FEATURE_COLUMNS
    assert "Time" not in features.columns
    assert "Amount" not in features.columns
    assert "is_fraud" not in features.columns
    assert "transaction_id" not in features.columns
