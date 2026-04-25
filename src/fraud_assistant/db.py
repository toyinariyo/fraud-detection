"""SQLite transaction lookup helpers."""

from __future__ import annotations

import sqlite3
from typing import Any

from fraud_assistant.config import DB_PATH


def get_transaction(transaction_id: int, include_ground_truth: bool = False) -> dict[str, Any]:
    """Fetch one transaction by SQLite rowid."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT rowid AS transaction_id, * FROM transactions WHERE rowid = ?",
            (transaction_id,),
        ).fetchone()

    if row is None:
        raise LookupError(f"Transaction ID {transaction_id} not found")

    transaction = dict(row)
    if not include_ground_truth:
        transaction.pop("is_fraud", None)

    return transaction
