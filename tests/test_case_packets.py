from fraud_assistant.case_packets import DATASET_LIMITATIONS, build_case_packet


def test_case_packet_does_not_expose_ground_truth_by_default():
    packet = build_case_packet(1)

    assert "is_fraud" not in packet["transaction"]


def test_case_packet_includes_dataset_limitations():
    packet = build_case_packet(1)

    for limitation in DATASET_LIMITATIONS:
        assert limitation in packet["dataset_limitations"]
