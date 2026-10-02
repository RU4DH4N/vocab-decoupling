import math

import pytest
import torch

from data.communication import ProtocolBatch
from execution.protocol_evaluation import score_rows
from models.protocol.receiver import ReceiverReadout


def test_path_bpb_uses_summed_utf8_bytes_not_mean_row_bpb():
    targets = torch.tensor([[1, -100, -100], [0, 1, 0]])
    batch = ProtocolBatch(
        sender_ids=torch.zeros(2, 2, dtype=torch.long),
        receiver_ids=torch.zeros_like(targets),
        targets=targets,
        candidate_event_ids=torch.tensor([0, 0, 1]).expand(2, 3, 3),
        frontier=torch.ones_like(targets),
        alignment_targets=torch.tensor([[0, -1, -1], [0, 0, 0]]),
        target_bytes=torch.tensor([1, 9]),
    )
    logs = torch.tensor([0.4, 0.4, 0.2]).log().expand(2, 3, 3)
    readout = ReceiverReadout(torch.zeros(2, 3, 2), (logs, logs))
    scores = score_rows(readout, batch)
    torch.testing.assert_close(
        scores.nll, torch.tensor([1, 3], dtype=torch.double) * math.log(2)
    )
    summary = scores.aggregate()
    assert summary["tokens_including_stop"] == 4
    assert summary["utf8_bytes"] == 10
    assert summary["canonical_path_bpb"] == pytest.approx(0.4)
    assert summary["top1"] == 0.5
    assert summary["alignment_count"] == 8
    assert summary["correct_event_mass"] == pytest.approx(0.8)
    assert summary["hard_event_accuracy"] == 1
    assert summary["expected_event_error"] == pytest.approx(0.2)


def test_empty_supervision_is_explicitly_undefined_not_zero_performance():
    targets = torch.full((1, 2), -100)
    batch = ProtocolBatch(
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros_like(targets),
        targets,
        torch.full((1, 2, 1), -1),
        torch.full_like(targets, -1),
        torch.full_like(targets, -1),
        torch.tensor([0]),
    )
    scores = score_rows(ReceiverReadout(torch.zeros(1, 2, 2), ()), batch)
    summary = scores.aggregate()
    assert summary["nll_nats"] == 0
    assert summary["ce"] is None
    assert summary["canonical_path_bpb"] is None
    assert summary["top1"] is None
    assert summary["correct_event_mass"] is None
