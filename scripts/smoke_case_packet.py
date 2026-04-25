"""Build and print a deterministic case packet for one transaction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_assistant.case_packets import build_case_packet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test deterministic case packet generation.")
    parser.add_argument("transaction_id", nargs="?", type=int, default=1)
    args = parser.parse_args()

    packet = build_case_packet(args.transaction_id)
    model = packet["model"]

    print(f"transaction_id: {packet['transaction_id']}")
    print(f"model_name: {model['model_name']}")
    print(f"fraud_probability: {model['fraud_probability']:.6f}")
    print(f"threshold: {model['threshold']:.6f}")
    print(f"decision: {model['decision']}")
    print("dataset_limitations:")
    for limitation in packet["dataset_limitations"]:
        print(f"- {limitation}")


if __name__ == "__main__":
    main()
