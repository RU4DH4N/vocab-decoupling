import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from models.encoding.walsh import hadamard_rows

POWERS_OF_TWO = [1, 2, 4, 8, 16, 32, 64, 128]

powers_of_two = st.sampled_from(POWERS_OF_TWO)

thorough = settings(max_examples=200, deadline=None)


def rows(n, idx):
    return hadamard_rows(n, idx, dtype=np.float64)


def reference(n, idx):
    idx = np.asarray(idx).reshape(-1)

    return np.array(
        [
            [-1.0 if (int(i) & j).bit_count() & 1 else 1.0 for j in range(n)]
            for i in idx
        ],
        dtype=np.float64,
    ).reshape(idx.size, n)


class TestMatchesTheDefinition:
    @given(
        n=powers_of_two,
        idx=st.lists(st.integers(min_value=0, max_value=127), max_size=32),
    )
    @thorough
    def test_selected_rows_match_the_reference(self, n, idx):
        idx = [i % n for i in idx]

        np.testing.assert_array_equal(rows(n, idx), reference(n, idx))

    @pytest.mark.parametrize("n", POWERS_OF_TWO)
    def test_full_matrix_matches_the_reference(self, n):
        idx = np.arange(n)

        np.testing.assert_array_equal(rows(n, idx), reference(n, idx))

    def test_parity_is_used_rather_than_the_popcount_itself(self):
        row = rows(16, [15])[0]

        assert row[15] == 1
        assert row[7] == -1
        assert row[3] == 1


class TestStructure:
    @given(
        n=powers_of_two,
        idx=st.lists(st.integers(min_value=0, max_value=127), max_size=32),
    )
    @thorough
    def test_every_entry_is_exactly_plus_or_minus_one(self, n, idx):
        result = rows(n, [i % n for i in idx])

        assert np.all((result == 1) | (result == -1))

    @pytest.mark.parametrize("n", POWERS_OF_TWO)
    def test_full_matrix_is_symmetric_and_orthogonal(self, n):
        h = rows(n, np.arange(n))

        np.testing.assert_array_equal(h, h.T)

        np.testing.assert_array_equal(h @ h.T, n * np.eye(n))

    @pytest.mark.parametrize("n", POWERS_OF_TWO)
    def test_row_zero_and_column_zero_are_all_ones(self, n):
        h = rows(n, np.arange(n))

        np.testing.assert_array_equal(h[0], np.ones(n))
        np.testing.assert_array_equal(h[:, 0], np.ones(n))

    def test_large_n_selects_only_the_requested_rows(self):
        idx = np.array([0, 1, 2, 3, 17, 255, 511, 1023])

        result = rows(1024, idx)

        assert result.shape == (len(idx), 1024)
        np.testing.assert_array_equal(result, reference(1024, idx))


class TestIndexHandling:
    @pytest.mark.parametrize(
        "idx",
        [
            3,
            [],
            [0, 1, 3],
            (0, 1, 3),
            np.array(3),
            np.array([0, 1, 3]),
            np.array([[0, 1], [2, 3]]),
            np.array([[[0, 1]], [[2, 3]]]),
            np.empty((3, 0, 4), dtype=np.int64),
        ],
    )
    def test_indices_of_any_shape_are_flattened_to_rows(self, idx):
        result = rows(8, idx)

        assert result.shape == (np.asarray(idx).size, 8)
        np.testing.assert_array_equal(result, reference(8, idx))

    @pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.uint32])
    def test_integer_index_dtypes_are_accepted(self, dtype):
        idx = np.array([0, 1, 3, 7], dtype=dtype)

        np.testing.assert_array_equal(rows(8, idx), reference(8, idx))

    def test_the_callers_index_array_is_left_alone(self):
        idx = np.array([[0, 1], [2, 3]], dtype=np.int64)

        rows(4, idx)

        np.testing.assert_array_equal(idx, np.array([[0, 1], [2, 3]]))


class TestDtype:
    def test_float64_dtype_is_respected(self):
        assert hadamard_rows(8, [0, 1, 2], dtype=np.float64).dtype == np.float64

    @pytest.mark.parametrize(
        "dtype", [np.float16, np.float32, np.float64, np.int8, np.int32, np.int64]
    )
    def test_signed_dtypes_are_respected_without_changing_values(self, dtype):
        result = hadamard_rows(8, [0, 1, 3, 7], dtype=dtype)

        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result, reference(8, [0, 1, 3, 7]).astype(dtype))

    @pytest.mark.parametrize(
        "dtype",
        [np.uint8, np.uint32, np.bool_, np.complex64, "timedelta64[s]"],
    )
    def test_non_real_output_dtypes_are_rejected(self, dtype):
        with pytest.raises(TypeError, match="signed integer or floating"):
            hadamard_rows(2, [1], dtype=dtype)


class TestRejectsBadInput:
    @pytest.mark.parametrize("n", [8.0, True, "8"])
    def test_n_must_be_integral(self, n):
        with pytest.raises(TypeError, match="n must be integral"):
            rows(n, [])

    def test_n_must_fit_the_internal_index_representation(self):
        with pytest.raises(ValueError, match="uint32 index capacity"):
            rows(2**32, [])

    @pytest.mark.parametrize("idx", [[3.9], np.array([1.5]), [True], ["3"]])
    def test_indices_must_be_integral(self, idx):
        with pytest.raises(TypeError, match="idx must be integral"):
            rows(8, idx)

    @pytest.mark.parametrize("n", [0, -1, -4, 3, 5, 6, 7, 9, 12, 15, 17, 33])
    def test_n_must_be_a_power_of_two(self, n):
        with pytest.raises(ValueError, match="power of two"):
            rows(n, [])

    @pytest.mark.parametrize("n, idx", [(1, [1]), (8, [8]), (8, [100]), (8, [0, 8])])
    def test_index_at_or_above_n_is_rejected(self, n, idx):
        with pytest.raises(ValueError, match=r"\[0,"):
            rows(n, idx)

    @given(n=powers_of_two, i=st.integers(min_value=0, max_value=10_000))
    @thorough
    def test_no_index_above_n_slips_through(self, n, i):
        with pytest.raises(ValueError, match=r"\[0,"):
            rows(n, [n + i])

    @pytest.mark.parametrize("idx", [[-1], [0, -1], [-100], np.array([-1])])
    def test_negative_indices_are_rejected(self, idx):
        with pytest.raises(ValueError, match=r"\[0, 8\)"):
            rows(8, idx)
