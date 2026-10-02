import torch

from models.protocol.receiver import LowRankMessageChannel


def test_message_channel_is_an_exact_no_op_at_initialization():
    channel = LowRankMessageChannel(10, 8, 4)
    hidden = torch.randn(3, 10)
    message = torch.randn(3, 8)

    torch.testing.assert_close(channel(hidden, message), hidden, rtol=0, atol=0)
