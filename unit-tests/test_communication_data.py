import numpy as np
import pytest
import torch

from data.communication import ReceiverInventory, ReceiverUnits
from data.tokenizers import train_bpe
from models.protocol.correspondence import causal_event_mask


def test_first_unit_is_native_context_not_an_unseen_sender_only_prefix():
    table = ReceiverUnits.bytes(["a", " bc", " d"])
    batch = table.batch(
        torch.tensor([[0, 1, 2]]),
        device="cpu",
        max_receiver_steps=16,
        candidate_window=8,
    )
    stop, start = table.inventory.stop, table.inventory.start
    assert batch.receiver_ids.tolist() == [[start, 97, stop, 32, 98, 99, stop, 32, 100]]
    assert batch.targets.tolist() == [[-100, -100, 32, 98, 99, stop, 32, 100, stop]]
    assert batch.frontier.tolist() == [[-1, -1, 0, 0, 0, 0, 1, 1, 1]]
    assert batch.alignment_targets.tolist() == batch.frontier.tolist()
    assert batch.target_bytes.tolist() == [5]


def test_frontier_only_advances_after_an_observed_stop():
    table = ReceiverUnits.bytes(["hi", " bank", " balance"])
    batch = table.batch(
        torch.tensor([[0, 1, 2]]),
        device="cpu",
        max_receiver_steps=32,
        candidate_window=8,
    )
    for u in range(batch.receiver_ids.shape[1]):
        consumed = batch.receiver_ids[0, : u + 1]
        expected = int(consumed.eq(table.inventory.stop).sum()) - 1
        assert batch.frontier[0, u] == expected


def test_future_word_changes_do_not_change_earlier_native_inputs_or_frontiers():
    table = ReceiverUnits.bytes(["hi", " bank", " balance", " bakers"])
    a = table.batch(
        torch.tensor([[0, 1, 2]]),
        device="cpu",
        max_receiver_steps=32,
        candidate_window=8,
    )
    b = table.batch(
        torch.tensor([[0, 3, 2]]),
        device="cpu",
        max_receiver_steps=32,
        candidate_window=8,
    )
    prefix = 1 + len("hi".encode()) + 1 + len(" ba".encode())
    torch.testing.assert_close(a.receiver_ids[:, :prefix], b.receiver_ids[:, :prefix])
    torch.testing.assert_close(a.frontier[:, :prefix], b.frontier[:, :prefix])


def test_padding_and_byte_accounting_are_independent_of_utf8_token_count():
    table = ReceiverUnits.bytes(["a", " café", "🙂", " x"])
    batch = table.batch(
        torch.tensor([[0, 1], [2, 3]]),
        device="cpu",
        max_receiver_steps=32,
        candidate_window=8,
    )
    assert batch.target_bytes.tolist() == [len(" café".encode()), 2]
    live = batch.targets.ne(-100)
    assert live.sum(1).tolist() == [len(" café".encode()) + 1, 3]
    assert batch.alignment_targets[~live].eq(-1).all()


def test_bpe_and_byte_tables_share_sender_ids_and_target_bytes():
    units = ["hello", " bank", " balance", " café"]
    tokenizer = train_bpe(280, units * 4, False)
    byte_table = ReceiverUnits.bytes(units)
    bpe_table = ReceiverUnits.bpe(units, tokenizer)
    rows = torch.tensor([[0, 1, 2], [3, 1, 2]])
    byte_batch = byte_table.batch(
        rows, device="cpu", max_receiver_steps=64, candidate_window=8
    )
    bpe_batch = bpe_table.batch(
        rows, device="cpu", max_receiver_steps=64, candidate_window=8
    )
    torch.testing.assert_close(byte_batch.sender_ids, bpe_batch.sender_ids)
    torch.testing.assert_close(byte_batch.target_bytes, bpe_batch.target_bytes)
    assert bpe_batch.receiver_ids.shape[1] < byte_batch.receiver_ids.shape[1]


def test_windowed_candidates_allow_the_same_events_as_every_event():
    table = ReceiverUnits.bytes(["a", " bb", " c", " dddd"])
    rows = torch.from_numpy(np.random.default_rng(3).integers(0, 4, (4, 12)))
    for window in (1, 3, 20):
        batch = table.batch(
            rows, device="cpu", max_receiver_steps=128, candidate_window=window
        )
        every = torch.arange(rows.shape[1] - 1).expand(*batch.frontier.shape, -1)
        allowed = causal_event_mask(every, batch.frontier, window)
        windowed = causal_event_mask(batch.candidate_event_ids, batch.frontier, window)
        assert batch.candidate_event_ids.shape[-1] == window
        for b, t in np.ndindex(*batch.frontier.shape):
            assert (
                batch.candidate_event_ids[b, t][windowed[b, t]].tolist()
                == every[b, t][allowed[b, t]].tolist()
            )


@pytest.mark.parametrize(
    "rows",
    [
        torch.empty(0, 2, dtype=torch.long),
        torch.tensor([[0]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([[0, 4]]),
    ],
)
def test_invalid_batches_are_rejected_on_cpu(rows):
    table = ReceiverUnits.bytes(["a", "b"])
    with pytest.raises(ValueError):
        table.batch(rows, device="cpu", max_receiver_steps=16, candidate_window=8)


def test_long_sequences_raise_instead_of_losing_stop():
    table = ReceiverUnits.bytes(["a", " bc"])
    with pytest.raises(ValueError, match="includes the context prefix"):
        table.batch(
            torch.tensor([[0, 1]]),
            device="cpu",
            max_receiver_steps=4,
            candidate_window=8,
        )


def test_stop_cannot_appear_as_an_ordinary_inventory_piece():
    with pytest.raises(ValueError, match="ordinary inventory IDs"):
        ReceiverUnits(["a"], [[256]], ReceiverInventory(256))


def reference_batch(table, rows):
    previous_rows, target_rows, frontier_rows, byte_counts = [], [], [], []
    for row in rows.tolist():
        symbols, frontiers = [], []
        for event, unit in enumerate(row):
            encoded = [*table.pieces[unit], table.inventory.stop]
            symbols.extend(encoded)
            frontiers.extend([event - 1] * len(encoded))
        prefix = len(table.pieces[row[0]]) + 1
        previous_rows.append([table.inventory.start, *symbols[:-1]])
        target_rows.append([-100] * prefix + symbols[prefix:])
        frontier_rows.append(frontiers)
        byte_counts.append(sum(table.byte_lengths[unit] for unit in row[1:]))
    width = max(map(len, previous_rows))
    return (
        torch.tensor([r + [0] * (width - len(r)) for r in previous_rows]),
        torch.tensor([r + [-100] * (width - len(r)) for r in target_rows]),
        torch.tensor([r + [-1] * (width - len(r)) for r in frontier_rows]),
        torch.tensor(byte_counts),
    )


@pytest.mark.parametrize("kind", ["bytes", "bpe"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorised_batches_match_the_reference_construction(kind, seed):
    units = ["a", " bb", " ccc", " dé", "日本", " x", " yyyyy"]
    table = (
        ReceiverUnits.bytes(units)
        if kind == "bytes"
        else ReceiverUnits.bpe(units, train_bpe(280, units * 4, False))
    )
    rng = np.random.default_rng(seed)
    rows = torch.from_numpy(rng.integers(0, len(units), (5, 6)))
    batch = table.batch(rows, device="cpu", max_receiver_steps=256, candidate_window=3)
    previous, targets, frontier, byte_counts = reference_batch(table, rows)
    assert torch.equal(batch.receiver_ids, previous)
    assert torch.equal(batch.targets, targets)
    assert torch.equal(batch.frontier, frontier)
    assert torch.equal(
        batch.alignment_targets, frontier.masked_fill(targets.eq(-100), -1)
    )
    assert torch.equal(batch.target_bytes, byte_counts)
    expected = frontier[..., None] + torch.tensor([-2, -1, 0])
    assert torch.equal(
        batch.candidate_event_ids, expected.masked_fill(expected < 0, -1)
    )
    assert torch.equal(batch.sender_ids, rows[:, :-1])
