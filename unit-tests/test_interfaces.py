import pytest
import torch
from test_receiver import inputs, receiver

from models.protocol.interfaces import KINDS, InterfaceReceiver, layerwise_parameters


def interface(kind):
    reference = receiver()
    return reference, InterfaceReceiver(reference.native, 8, 4, 4, 3, 32, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_every_interface_starts_as_an_exact_no_op(kind):
    reference, model = interface(kind)
    ids, candidates, events, frontier = inputs()
    native = reference.native(ids)
    readout = model(ids, candidates, events, frontier, hard=False)
    torch.testing.assert_close(readout.logits, native, rtol=0, atol=0)
    assert readout.correspondence
    for log_probs in readout.correspondence:
        assert log_probs.shape == events.shape
        probs = log_probs.exp().sum(-1)
        live = log_probs.isfinite().any(-1)
        torch.testing.assert_close(probs[live], torch.ones_like(probs[live]))


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_reach_scorers_and_channels_but_not_the_native_model(kind):
    _, model = interface(kind)
    with torch.no_grad():
        for channel in model.channels:
            for parameter in channel.parameters():
                parameter.normal_(std=0.1)
    ids, candidates, events, frontier = inputs()
    model(ids, candidates, events, frontier, hard=False).logits.sum().backward()
    assert all(p.grad is not None for p in model.channels.parameters())
    assert all(p.grad is not None for p in model.scorers.parameters())
    assert all(p.grad is None for p in model.native.parameters())


def test_cross_attention_matches_layerwise_capacity():
    reference, model = interface("attention")
    budget = layerwise_parameters(16, 8, 4, 4) * reference.native.trunk.config.n_layers
    actual = sum(
        p.numel() for p in (*model.scorers.parameters(), *model.channels.parameters())
    )
    heads = reference.native.trunk.config.n_heads
    assert (
        abs(actual - budget)
        <= heads * 2 * (16 + 8) * reference.native.trunk.config.n_layers
    )
