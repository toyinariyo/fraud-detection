"""
setup_db.py
-----------
One-time script to load creditcard.csv into a SQLite database.
Run this before opening any notebooks:  python setup_db.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = Path("creditcard.csv")
DB_PATH  = Path("data/fraud_detection.db")

def main():
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # Rename 'Class' to 'is_fraud' for clarity in SQL queries
    df = df.rename(columns={"Class": "is_fraud"})

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    print(f"Fraud cases : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.3f}%)")
    print(f"Legit cases : {(df['is_fraud'] == 0).sum():,}")

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print(f"\nWriting to {DB_PATH} ...")
    df.to_sql("transactions", conn, if_exists="replace", index=False)

    # Useful indexes for the queries used in notebooks
    conn.execute("CREATE INDEX IF NOT EXISTS idx_is_fraud ON transactions (is_fraud);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_amount  ON transactions (Amount);")
    conn.commit()
    conn.close()

    print("Done. Database ready at", DB_PATH)

if __name__ == "__main__":
    main()
