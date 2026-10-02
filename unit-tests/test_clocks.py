import numpy as np
import torch
from test_protocol_stages import tiny_model

from data.communication import ReceiverUnits
from data.protocol_corpus import EventCorpus
from execution.clocks import distort, reading, split_word


def test_split_word_rebuilds_the_word_in_order():
    rng = np.random.default_rng(0)
    for word in ("a", " hello", "日本語", " x"):
        for pieces in (1, 2, 3):
            parts = split_word(word, pieces, rng)
            assert "".join(parts) == word
            assert 1 <= len(parts) <= min(pieces, len(word))
            assert all(parts)


def test_availability_follows_completed_pieces_and_never_leaks_the_word_end():
    corpus = EventCorpus(["ab cde fghi jk lmn"], ["x"], 15)
    table = ReceiverUnits.bytes(corpus.vocab)
    model = tiny_model().eval()
    rows = corpus.windows("train", 4, 2, np.random.default_rng(1))
    config = {
        "receiver_steps": 64,
        "corpus": {"max_unit_bytes": 15},
        "interface": {"candidate_window": 3},
    }
    batch, rules = distort(
        model,
        table,
        corpus,
        rows,
        3,
        np.random.default_rng(2),
        config,
        torch.device("cpu"),
    )
    frontier = table.batch(
        rows, device="cpu", max_receiver_steps=64, candidate_window=3
    ).frontier
    live = frontier.ge(0)
    latest, oracle, fixed = rules["latest"], rules["oracle"], rules["fixed"]
    assert torch.all(latest[live] >= oracle[live])
    assert torch.all(fixed[live] == frontier[live])
    starts = batch.receiver_ids.eq(table.inventory.stop) & live
    assert torch.equal(latest[starts], oracle[starts])
    assert torch.all(latest[live] - oracle[live] <= 2)
    assert torch.any(latest[live] > oracle[live])
    single = reading(batch, rules, "oracle")
    assert single.candidate_event_ids.shape[-1] == 1
    assert torch.equal(single.frontier, oracle)
