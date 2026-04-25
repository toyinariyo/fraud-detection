"""Feature preparation for deterministic model inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount_log"]


def prepare_features(transaction: Mapping[str, Any]) -> pd.DataFrame:
    """Return a one-row model input frame in the exact training feature order."""
    feature_values = {column: transaction[column] for column in FEATURE_COLUMNS if column != "Amount_log"}
    feature_values["Amount_log"] = np.log1p(transaction["Amount"])

    return pd.DataFrame([[feature_values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
