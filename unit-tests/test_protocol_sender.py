import pytest
import torch
from designs import smoke_config

from models.protocol.sender import CoarseSender


@pytest.fixture
def sender():
    config = smoke_config()["sender"]
    torch.manual_seed(7)
    return CoarseSender(**config).eval()


def test_sender_owns_no_receiver_parameters(sender):
    names = list(sender.state_dict())
    assert names
    assert not any("head" in name for name in names)
    assert "trunk_table" not in names


def test_input_table_required(sender):
    with pytest.raises(RuntimeError, match="build_table"):
        sender(torch.tensor([[0]]))


def test_failed_table_build_keeps_previous_table(sender, monkeypatch):
    sender.build_table(["one", "two"])
    original = sender.trunk_table.clone()
    monkeypatch.setattr(
        sender,
        "encode_trunk_vocab",
        lambda vocab: (torch.zeros(1, original.shape[-1]), torch.tensor([True])),
    )
    with pytest.raises(ValueError, match="1 directionless entries"):
        sender.build_table(["bad"])
    torch.testing.assert_close(original, sender.trunk_table, rtol=0, atol=0)


def test_table_follows_consumer_not_codebook_dtype(sender):
    sender.to(torch.bfloat16)
    sender.trunk_code.float()
    sender.build_table(["one", "two"])
    assert sender.trunk_table.dtype == torch.bfloat16
    assert sender.trunk_table.device == sender.trunk.input_reference.device


def test_incremental_messages_match_full_forward(sender):
    sender.build_table(["one", "two", "three"])
    ids = torch.tensor([[0, 1, 2, 0]])
    caches = sender.new_caches(8)
    with torch.no_grad():
        full = sender(ids)
        incremental = torch.cat(
            [
                sender(ids[:, index : index + 1], caches)
                for index in range(ids.shape[1])
            ],
            dim=1,
        )
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-5)


def test_sender_future_input_cannot_change_earlier_messages(sender):
    sender.build_table(["one", "two", "three"])
    with torch.no_grad():
        first = sender(torch.tensor([[0, 1, 2]]))
        changed = sender(torch.tensor([[0, 1, 0]]))
    torch.testing.assert_close(first[:, :2], changed[:, :2], rtol=0, atol=0)
