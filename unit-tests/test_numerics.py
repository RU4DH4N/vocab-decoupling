import pytest
import torch

import models.shared.numerics as numerics
from models.shared.numerics import masked_cross_entropy, normalise, normalise_table


def test_normalise_produces_unit_vectors_and_preserves_zero():
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    result = normalise(values, eps=1e-6)
    torch.testing.assert_close(result[0].norm(), torch.tensor(1.0))
    torch.testing.assert_close(result[1], torch.zeros(2))


def test_normalise_rejects_non_floating_tensors():
    with pytest.raises(TypeError, match="floating-point"):
        normalise(torch.tensor([[3, 4]]), eps=1e-6)


def test_masked_cross_entropy_ignores_targets_and_handles_all_ignored():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    targets = torch.tensor([0, -100])
    expected = torch.nn.functional.cross_entropy(logits[:1], targets[:1])
    actual = masked_cross_entropy(logits, targets, ignore=-100)
    torch.testing.assert_close(actual, expected)

    empty = masked_cross_entropy(logits, torch.tensor([-100, -100]), ignore=-100)
    assert empty.item() == 0.0
    empty.backward()
    assert logits.grad is not None


def test_normalise_table_marks_only_directionless_rows_dead():
    table, dead = normalise_table(
        torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
        eps=1e-6,
    )
    assert dead.tolist() == [False, True]
    torch.testing.assert_close(table[0].norm(), torch.tensor(1.0))
    torch.testing.assert_close(table[1], torch.zeros(2))


def test_blockwise_table_normalisation_matches_whole_table(monkeypatch):
    codes = torch.randn(10, 6)
    codes[3] = 0
    expected = normalise(codes.clone(), eps=1e-6)
    monkeypatch.setattr(numerics, "TABLE_ROWS", 4)
    table, dead = normalise_table(codes.clone(), eps=1e-6)
    torch.testing.assert_close(table, expected, rtol=0, atol=0)
    assert dead.tolist() == [i == 3 for i in range(10)]


def test_normalise_table_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="codes must have shape"):
        normalise_table(torch.ones(2), eps=1e-6)
