import pytest
import torch

from models.protocol.correspondence import (
    CompatibilityScorer,
    alignment_diagnostics,
    alignment_nll,
    causal_event_mask,
    correspondence_log_probs,
    select_message,
)


def test_scorer_capacity_and_candidate_permutation():
    scorer = CompatibilityScorer(256, 256, 64)
    assert sum(p.numel() for p in scorer.parameters()) == 32768
    query = torch.randn(2, 3, 256)
    candidates = torch.randn(2, 3, 5, 256)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_close(
        scorer(query, candidates[..., permutation, :]),
        scorer(query, candidates)[..., permutation],
    )


def test_mask_uses_event_ids_not_duplicated_slot_or_receiver_index():
    ids = torch.tensor([[0, 0, 1, 2, 2, 3, -1]])
    allowed = causal_event_mask(ids, torch.tensor([2]), window=2)
    assert allowed.tolist() == [[False, False, True, True, True, False, False]]


@pytest.mark.parametrize("hard", [False, True])
def test_forbidden_future_message_cannot_change_selected_content(hard):
    scorer = CompatibilityScorer(4, 6, 3)
    receiver = torch.randn(1, 2, 6)
    messages = torch.randn(1, 2, 5, 4, requires_grad=True)
    ids = torch.arange(5).expand(1, 2, 5)
    mask = causal_event_mask(ids, torch.tensor([[1, 2]]), window=3)

    def read(values):
        return select_message(
            values,
            correspondence_log_probs(scorer(receiver, values), mask),
            hard=hard,
        )

    selected = read(messages)
    changed = messages.detach().clone()
    changed[~mask] = 1e4
    torch.testing.assert_close(selected, read(changed), rtol=0, atol=0)
    selected.sum().backward()
    assert messages.grad is not None
    assert torch.count_nonzero(messages.grad[~mask]) == 0


@pytest.mark.parametrize("hard", [False, True])
def test_unavailable_memory_is_zero_and_backward_is_finite(hard):
    scores = torch.randn(2, 4, requires_grad=True)
    candidates = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[False] * 4, [True, False, True, False]])
    log_probs = correspondence_log_probs(scores, mask)
    selected = select_message(candidates, log_probs, hard=hard)
    assert torch.equal(selected[0], torch.zeros(3))
    assert not log_probs.isnan().any()
    selected.square().sum().backward()
    assert candidates.grad is not None and candidates.grad.isfinite().all()
    if not hard:
        assert scores.grad is not None and scores.grad.isfinite().all()
        assert torch.equal(scores.grad[0], torch.zeros(4))


def test_duplicate_copies_share_positive_mass_and_event_error():
    ids = torch.tensor([[4, 5, 5]])
    log_probs = torch.tensor([[0.2, 0.3, 0.5]]).log()
    target = torch.tensor([5])
    loss = alignment_nll(log_probs, ids, target, torch.tensor([True]))
    torch.testing.assert_close(loss, -torch.tensor([0.8]).log())
    metrics = alignment_diagnostics(log_probs, ids, target)
    assert metrics.valid.all() and metrics.hard_correct.all()
    torch.testing.assert_close(metrics.correct_mass, torch.tensor([0.8]))
    torch.testing.assert_close(metrics.expected_event_error, torch.tensor([0.2]))


def test_missing_live_target_is_not_silently_ignored():
    scores = torch.zeros(2, 3, requires_grad=True)
    ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    log_probs = correspondence_log_probs(scores, torch.ones_like(ids, dtype=torch.bool))
    nll = alignment_nll(
        log_probs, ids, torch.tensor([3, -1]), torch.tensor([True, False])
    )
    assert nll[0].isinf() and nll[1] == 0
    nll[1].backward()
    assert scores.grad is not None and torch.equal(
        scores.grad, torch.zeros_like(scores)
    )


def test_masked_target_copy_is_not_counted():
    ids = torch.tensor([[1, 1, 0]])
    log_probs = correspondence_log_probs(
        torch.zeros(1, 3), torch.tensor([[False, True, True]])
    )
    loss = alignment_nll(log_probs, ids, torch.tensor([1]), torch.tensor([True]))
    torch.testing.assert_close(loss, torch.tensor([2.0]).log())


def test_soft_route_trains_scorer_and_sender_with_frozen_native_states():
    scorer = CompatibilityScorer(4, 6, 3)
    receiver = torch.randn(2, 5, 6)
    messages = torch.randn(2, 5, 3, 4, requires_grad=True)
    allowed = torch.ones(2, 5, 3, dtype=torch.bool)
    probabilities = correspondence_log_probs(scorer(receiver, messages), allowed)
    select_message(messages, probabilities, hard=False).square().mean().backward()
    assert receiver.grad is None
    assert messages.grad is not None and messages.grad.abs().sum() > 0
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.parameters()
    )


def test_bfloat16_content_keeps_consumer_precision():
    candidates = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    log_probs = correspondence_log_probs(
        torch.randn(2, 4, dtype=torch.bfloat16), torch.ones(2, 4, dtype=torch.bool)
    )
    assert log_probs.dtype == torch.float32
    assert select_message(candidates, log_probs, hard=False).dtype == torch.bfloat16


def test_all_missing_alignment_labels_do_not_create_nan_gradients():
    scores = torch.randn(2, 3, requires_grad=True)
    probs = correspondence_log_probs(scores, torch.zeros(2, 3, dtype=torch.bool))
    loss = alignment_nll(
        probs,
        torch.full((2, 3), -1),
        torch.full((2,), -1),
        torch.zeros(2, dtype=torch.bool),
    )
    loss.sum().backward()
    assert scores.grad is not None and scores.grad.isfinite().all()
    metrics = alignment_diagnostics(probs, torch.full((2, 3), -1), torch.full((2,), -1))
    assert not metrics.valid.any()


def test_invalid_shapes_and_masks_fail_at_interface():
    scorer = CompatibilityScorer(4, 6, 3)
    with pytest.raises(ValueError, match="width"):
        scorer(torch.zeros(2, 5), torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="boolean"):
        correspondence_log_probs(torch.zeros(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="nonempty"):
        correspondence_log_probs(torch.zeros(2, 0), torch.ones(2, 0, dtype=torch.bool))
    with pytest.raises(TypeError, match="integer"):
        causal_event_mask(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), 5)
