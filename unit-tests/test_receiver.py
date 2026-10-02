import pytest
import torch

from models.bpe.bpe import BPEModel
from models.protocol.receiver import Receiver


def receiver():
    torch.manual_seed(17)
    native = BPEModel(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        dropout=0.2,
        multiple_of=8,
        max_seq_len=16,
        rope_theta=10000,
        norm_eps=1e-5,
        initialiser_range=0.02,
        tie_word_embeddings=True,
    )
    return Receiver(native, 8, 4, 4, 3, output_symbols=32)


def inputs():
    torch.manual_seed(23)
    ids = torch.randint(0, 32, (2, 5))
    candidates = torch.randn(2, 5, 5, 8)
    events = torch.arange(5).expand(2, 5, 5)
    frontier = torch.arange(5).expand(2, 5)
    return ids, candidates, events, frontier


def activate(model):
    with torch.no_grad():
        for channel in model.channels:
            channel.up.weight.normal_(std=0.1)


def test_step_zero_is_exactly_native_even_while_interface_is_training():
    model = receiver().train()
    ids, candidates, events, frontier = inputs()
    assert not model.native.training
    assert all(not p.requires_grad for p in model.native.parameters())
    with torch.no_grad():
        native = model.native(ids)
        readout = model(ids, candidates, events, frontier, hard=False)
    torch.testing.assert_close(native, readout.logits, rtol=0, atol=0)
    assert len(readout.correspondence) == 2


@pytest.mark.parametrize("hard", [False, True])
def test_cached_and_full_receiver_logits_match(hard):
    model = receiver().eval()
    activate(model)
    ids, candidates, events, frontier = inputs()
    caches = model.native.new_caches(16)
    with torch.no_grad():
        full = model(ids, candidates, events, frontier, hard=hard).logits
        incremental = torch.cat(
            [
                model(
                    ids[:, t : t + 1],
                    candidates[:, t : t + 1],
                    events[:, t : t + 1],
                    frontier[:, t : t + 1],
                    hard=hard,
                    caches=caches,
                ).logits
                for t in range(ids.shape[1])
            ],
            dim=1,
        )
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-5)


def test_future_symbols_and_future_sender_events_cannot_change_prefix():
    model = receiver().eval()
    activate(model)
    ids, candidates, events, frontier = inputs()
    changed_ids = ids.clone()
    changed_ids[:, 3:] = (changed_ids[:, 3:] + 1) % 32
    changed_candidates = candidates.clone()
    changed_candidates[events > frontier[..., None]] = torch.nan
    changed_candidates[:, 3:] = 1000
    with torch.no_grad():
        base = model(ids, candidates, events, frontier, hard=False).logits
        changed = model(
            changed_ids, changed_candidates, events, frontier, hard=False
        ).logits
    torch.testing.assert_close(changed[:, :3], base[:, :3], rtol=0, atol=0)


def test_live_channel_trains_interface_not_native_weights():
    model = receiver().train()
    activate(model)
    ids, candidates, events, frontier = inputs()
    candidates.requires_grad_(True)
    logits = model(ids, candidates, events, frontier, hard=False).logits
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids.flatten()).backward()
    assert all(p.grad is None for p in model.native.parameters())
    assert all(p.grad is not None for p in model.scorers.parameters())
    assert all(p.grad is not None for p in model.channels.parameters())
    assert candidates.grad is not None and candidates.grad.isfinite().all()
    assert torch.count_nonzero(candidates.grad[events > frontier[..., None]]) == 0


def test_empty_memory_preserves_native_computation_after_adapter_training():
    model = receiver().eval()
    activate(model)
    ids, candidates, events, frontier = inputs()
    events = torch.full_like(events, -1)
    with torch.no_grad():
        native = model.native(ids)
        readout = model(ids, candidates, events, frontier, hard=False)
    torch.testing.assert_close(readout.logits, native, rtol=0, atol=0)
    assert all(torch.isneginf(p).all() for p in readout.correspondence)


def test_invalid_cache_count_is_checked_before_accessing_cache():
    model = receiver().eval()
    ids, candidates, events, frontier = inputs()
    with pytest.raises(ValueError, match="expected 2 caches"):
        model(ids, candidates, events, frontier, hard=False, caches=[])
