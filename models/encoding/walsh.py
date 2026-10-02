import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


def hadamard_rows(
    n: int,
    idx: ArrayLike,
    dtype: DTypeLike,
) -> NDArray:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be integral, got {type(n).__name__}")
    n = int(n)
    if n < 1 or n & (n - 1):
        raise ValueError(f"n must be a power of two, n={n}")
    if n >= 2**32:
        raise ValueError(f"n exceeds uint32 index capacity, n={n}")

    output_dtype = np.dtype(dtype)
    if output_dtype.kind not in "if":
        raise TypeError(
            f"dtype must be a signed integer or floating type, dtype={output_dtype}"
        )

    indices = np.asarray(idx)
    if indices.size and not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"idx must be integral, dtype={indices.dtype}")
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"row indices must be in [0, {n})")
    indices = indices.astype(np.uint32, copy=False).reshape(-1, 1)

    j = np.arange(n, dtype=np.uint32)[None, :]
    out = (np.bitwise_count(indices & j) & 1).astype(output_dtype, copy=False)
    np.multiply(out, -2, out=out)
    np.add(out, 1, out=out)
    return out
