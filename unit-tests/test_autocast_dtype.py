import pytest
import torch

from framework.runtime import autocast_context, autocast_dtype


@pytest.mark.parametrize(
    ("device", "precision", "expected"),
    [
        ("cpu", "auto", torch.float32),
        ("cpu", "fp32", torch.float32),
        ("cuda", "auto", torch.bfloat16),
        ("cuda", "bf16", torch.bfloat16),
        ("cuda", "fp32", torch.float32),
    ],
)
def test_autocast_dtype_follows_the_context_rule(device, precision, expected):
    assert autocast_dtype(torch.device(device), precision) is expected


def test_bf16_requires_cuda_and_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        autocast_dtype(torch.device("cpu"), "bf16")
    with pytest.raises(ValueError, match="unknown precision"):
        autocast_dtype(torch.device("cpu"), "fp8")


def test_fp32_context_is_a_no_op():
    with autocast_context(torch.device("cpu"), "auto"):
        assert torch.ones(1).dtype is torch.float32
