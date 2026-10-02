import random

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings, strategies as st

from models.encoding.code import (
    BLANK,
    N_BYTES,
    N_ROWS,
    Code,
    byte_matrix,
    position_codes,
)

light = settings(max_examples=25, deadline=None)
medium = settings(max_examples=60, deadline=None)

units = st.text(min_size=0, max_size=12)
widths = st.integers(min_value=1, max_value=24)
seeds = st.integers(min_value=0, max_value=10_000)


@st.composite
def fitting_vocab_and_width(draw, max_size):
    vocab = draw(st.lists(units, min_size=1, max_size=max_size))
    required = max(len(unit.encode("utf-8")) for unit in vocab)
    width = draw(st.integers(min_value=max(required, 1), max_value=max(required, 24)))
    return vocab, width


@st.composite
def fitting_unit_and_width(draw):
    unit = draw(units)
    required = len(unit.encode("utf-8"))
    width = draw(st.integers(min_value=max(required, 1), max_value=max(required, 24)))
    return unit, width


def test_position_codes_are_reproducible_and_position_specific():
    first = position_codes(4, 32, seed=17)
    second = position_codes(4, 32, seed=17)
    assert np.array_equal(first, second)
    assert len({row.tobytes() for row in first}) == 4


def test_position_code_mapping_is_part_of_the_checkpoint_contract():
    expected = [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
    assert position_codes(1, 8, seed=17)[0].tolist() == expected


def test_position_codes_handle_an_empty_request():
    assert position_codes(0, 32, seed=17).shape == (0, 32)


class TestByteMatrix:
    @given(fitting_vocab_and_width(8))
    @medium
    def test_shape_and_dtype(self, case):
        vocab, width = case
        out = byte_matrix(vocab, width)

        assert out.shape == (len(vocab), width)
        assert out.dtype == np.int32

    @given(fitting_vocab_and_width(8))
    @medium
    def test_every_slot_is_a_byte_or_pad(self, case):
        vocab, width = case
        out = byte_matrix(vocab, width)

        assert out.min() >= 0
        assert out.max() <= BLANK

    @given(fitting_vocab_and_width(6))
    @medium
    def test_content_then_pad_never_interleaved(self, case):
        vocab, width = case
        out = byte_matrix(vocab, width)

        for row in out:
            pads = np.flatnonzero(row == BLANK)
            if pads.size:
                assert np.array_equal(pads, np.arange(pads[0], width)), (
                    "pad symbols appear before the end of the unit"
                )

    @given(fitting_unit_and_width())
    @medium
    def test_prefix_matches_utf8(self, case):
        unit, width = case
        row = byte_matrix([unit], width)[0]
        expected = list(unit.encode("utf-8"))

        assert list(row[: len(expected)]) == expected

    @given(fitting_unit_and_width())
    @medium
    def test_a_unit_filling_the_width_carries_no_blank(self, case):
        unit, width = case
        row = byte_matrix([unit], width)[0]
        n = len(unit.encode("utf-8"))

        if n == width:
            assert not (row == BLANK).any(), (
                "a unit at least as long as the width should use every slot"
            )
        else:
            assert (row[n:] == BLANK).all()

    def test_multibyte_characters_occupy_multiple_slots(self):
        row = byte_matrix(["量"], 10)[0]

        assert list(row[:3]) == list("量".encode("utf-8"))
        assert (row[3:] == BLANK).all()

    def test_empty_unit_is_all_pad(self):
        assert (byte_matrix([""], 10)[0] == BLANK).all()

    def test_explicit_width_rejects_a_long_unit_instead_of_aliasing_it(self):
        with pytest.raises(ValueError, match="bytes=100.*width=10"):
            byte_matrix(["x" * 100], 10)

    def test_width_defaults_to_the_longest_unit(self):
        out = byte_matrix(["a", "abcd", "ab"])

        assert out.shape == (3, 4)


class TestCodeConstruction:
    def test_rejects_dims_below_byte_count(self):
        with pytest.raises(ValueError, match="dims=256 must exceed"):
            Code(dims=256, seed=1)

    def test_rejects_non_power_of_two_dims(self):
        with pytest.raises(ValueError, match="power of two"):
            Code(dims=513, seed=1)

    @given(seeds)
    @light
    def test_buffers_are_float32(self, seed):
        c = Code(dims=512, seed=seed)

        assert c.C.dtype == torch.float32
        assert c.P.dtype == torch.float32

    def test_buffers_are_not_parameters(self):
        assert list(Code(dims=512, seed=1).parameters()) == [], (
            "the fixed code must not be trainable"
        )

    def test_buffers_absent_from_state_dict(self):
        assert list(Code(dims=512, seed=1).state_dict().keys()) == []

    def test_buffers_are_registered_position_then_character(self):
        code = Code(dims=512, seed=1)

        assert [name for name, _ in code.named_buffers()] == ["P", "C"]

    @given(widths)
    @light
    def test_codebook_covers_every_byte_plus_blank(self, width):
        c = Code(dims=512, seed=1)

        assert c.C.shape == (N_ROWS, 512)
        assert c.C[BLANK].abs().max().item() == 0.0, "blank must contribute nothing"
        assert (c.C[:N_BYTES].abs().sum(dim=-1) > 0).all(), "a byte has no code"


class TestSeedIndependence:
    @given(seeds)
    @light
    def test_byte_atoms_are_exactly_orthogonal_for_any_seed(self, seed):
        c = Code(dims=512, seed=seed)
        byte_atoms = c.C[:N_BYTES]

        torch.testing.assert_close(
            byte_atoms @ byte_atoms.T, 512.0 * torch.eye(N_BYTES)
        )

    @given(seeds, seeds)
    @light
    def test_byte_atoms_do_not_depend_on_the_seed(self, a, b):
        assert torch.equal(Code(dims=512, seed=a).C, Code(dims=512, seed=b).C), (
            "C is fixed Hadamard rows; only the position codes are seeded"
        )

    @given(seeds)
    @medium
    def test_position_codes_are_balanced_signs_for_any_seed(self, seed):
        c = Code(dims=512, seed=seed)
        c(torch.from_numpy(byte_matrix(["abcdefgh"], 8)).long())

        assert set(c.P.unique().tolist()) <= {-1.0, 1.0}
        assert abs(c.P.mean().item()) < 0.1, "position sign pattern is imbalanced"

    @given(seeds, seeds)
    @light
    def test_distinct_seeds_give_distinct_position_codes(self, a, b):
        assume(a != b)

        x = torch.from_numpy(byte_matrix(["quantum"], 8)).long()

        assert not torch.allclose(Code(dims=512, seed=a)(x), Code(dims=512, seed=b)(x))

    @given(seeds)
    @light
    def test_same_seed_is_reproducible(self, seed):
        x = torch.from_numpy(byte_matrix(["the", " quantum"], 10)).long()

        torch.testing.assert_close(
            Code(dims=512, seed=seed)(x), Code(dims=512, seed=seed)(x)
        )

    @given(seeds)
    @medium
    def test_unrelated_units_stay_near_orthogonal_for_any_seed(self, seed):
        c = Code(dims=512, seed=seed)
        rng = random.Random(seed)
        vocab = ["".join(rng.choices("abcdefghijklmnop", k=6)) for _ in range(64)]

        v = c(torch.from_numpy(byte_matrix(vocab, 10)).long())
        v = v / v.norm(dim=-1, keepdim=True)
        cos = v @ v.T
        off = cos - torch.diag_embed(torch.diagonal(cos))

        assert off.abs().mean().item() < 0.25, (
            "unrelated units are not close to orthogonal for this seed"
        )

    @given(seeds)
    @light
    def test_shared_prefixes_produce_similar_codes(self, seed):
        c = Code(dims=512, seed=seed)
        v = c(
            torch.from_numpy(
                byte_matrix(["quantity", "quantise", "zzzzzzzz"], 10)
            ).long()
        )
        v = v / v.norm(dim=-1, keepdim=True)

        assert (v[0] @ v[1]).item() > (v[0] @ v[2]).item(), (
            "a shared prefix is no closer than an unrelated unit"
        )


class TestPadMasking:
    @given(widths, seeds)
    @medium
    def test_all_blank_unit_encodes_to_zero(self, width, seed):
        c = Code(dims=512, seed=seed)
        v = c(torch.full((1, width), BLANK))

        assert v.abs().max().item() == 0.0, "pad slots contribute to the encoding"

    @given(
        units,
        st.integers(min_value=4, max_value=12),
        st.integers(min_value=13, max_value=25),
    )
    @medium
    def test_encoding_is_independent_of_width(self, unit, small, large):
        assume(len(unit.encode("utf-8")) <= small)

        c = Code(dims=512, seed=1)
        a = c(torch.from_numpy(byte_matrix([unit], small)).long())
        b = c(torch.from_numpy(byte_matrix([unit], large)).long())

        torch.testing.assert_close(
            a,
            b,
            rtol=0.0,
            atol=1e-4,
            msg="the same string encodes differently at different padded width",
        )

    @given(units)
    @medium
    def test_growing_the_position_cache_is_consistent(self, unit):
        assume(len(unit.encode("utf-8")) <= 8)

        fresh = Code(dims=512, seed=1)
        direct = fresh(torch.from_numpy(byte_matrix([unit], 20)).long())

        grown = Code(dims=512, seed=1)
        grown(torch.from_numpy(byte_matrix(["x"], 3)).long())
        after = grown(torch.from_numpy(byte_matrix([unit], 20)).long())

        torch.testing.assert_close(direct, after, rtol=0.0, atol=1e-5)

    @given(units, seeds)
    @medium
    def test_energy_scales_with_byte_count_not_cap(self, unit, seed):
        k = len(unit.encode("utf-8"))
        assume(k > 0)

        c = Code(dims=512, seed=seed)
        v = c(torch.from_numpy(byte_matrix([unit])).long())

        ratio = (v.norm().item() ** 2) / 512

        assert 0.5 * k <= ratio <= 2.0 * k, (
            f"energy {ratio:.2f} is not on the order of the {k} bytes summed"
        )

    @given(seeds)
    @medium
    def test_expected_energy_equals_byte_count_on_average(self, seed):
        c = Code(dims=512, seed=seed)
        rng = random.Random(seed)
        k = 5
        vocab = ["".join(rng.choices("abcdefghijklmnop", k=k)) for _ in range(128)]

        v = c(torch.from_numpy(byte_matrix(vocab, 10)).long())
        mean_ratio = ((v.norm(dim=-1) ** 2) / 512).mean().item()

        assert abs(mean_ratio - k) < 0.5 * k, (
            f"mean energy {mean_ratio:.2f} does not match the {k} atoms summed"
        )


class TestEncodingSemantics:
    @given(seeds)
    @light
    def test_order_matters(self, seed):
        c = Code(dims=512, seed=seed)
        v = c(torch.from_numpy(byte_matrix(["ab", "ba"], 10)).long())

        assert not torch.allclose(v[0], v[1], atol=1e-5), (
            "the encoding is invariant to byte order"
        )

    @given(st.lists(units, min_size=1, max_size=8, unique=True))
    @medium
    def test_distinct_units_get_distinct_codes(self, vocab):
        c = Code(dims=512, seed=1)
        v = c(torch.from_numpy(byte_matrix(vocab)).long())

        for i in range(len(vocab)):
            for j in range(i + 1, len(vocab)):
                assert not torch.allclose(v[i], v[j], atol=1e-5)

    @given(units)
    @medium
    def test_any_string_encodes_without_a_vocabulary(self, unit):
        c = Code(dims=512, seed=1)
        v = c(torch.from_numpy(byte_matrix([unit])).long())

        assert v.shape == (1, 512)
        assert torch.isfinite(v).all()

    def test_unseen_scripts_encode(self):
        c = Code(dims=512, seed=1)
        exotic = ["量子", "بروت", "\U0001f9ee", "क्ष"]
        v = c(torch.from_numpy(byte_matrix(exotic)).long())

        assert torch.isfinite(v).all()
        assert (v.norm(dim=-1) > 0).all()

    @given(st.integers(min_value=1, max_value=6), st.integers(min_value=1, max_value=6))
    @light
    def test_batch_dims_are_preserved(self, B, T):
        c = Code(dims=512, seed=1)
        b = torch.randint(0, N_ROWS, (B, T, 10))

        assert c(b).shape == (B, T, 512)

    @given(st.lists(units, min_size=1, max_size=10))
    @medium
    def test_encoding_is_row_independent(self, vocab):
        c = Code(dims=512, seed=1)
        bm = torch.from_numpy(byte_matrix(vocab)).long()

        together = c(bm)
        apart = torch.cat([c(bm[i : i + 1]) for i in range(len(vocab))], dim=0)

        torch.testing.assert_close(together, apart, rtol=0.0, atol=1e-5)


class TestTable:
    @given(
        st.integers(min_value=1, max_value=40), st.integers(min_value=1, max_value=16)
    )
    @medium
    def test_table_matches_direct_encoding(self, n, chunk):
        c = Code(dims=512, seed=1)
        bm = torch.from_numpy(byte_matrix([f"u{i}" for i in range(n)], 10)).long()

        torch.testing.assert_close(
            c.table(bm, budget=chunk * 10 * 512 * 4), c(bm), rtol=0.0, atol=1e-5
        )

    @given(st.integers(min_value=1, max_value=16))
    @light
    def test_budget_does_not_change_results(self, chunk):
        c = Code(dims=512, seed=1)
        bm = torch.from_numpy(byte_matrix([f"u{i}" for i in range(37)], 10)).long()

        torch.testing.assert_close(
            c.table(bm, budget=chunk * 10 * 512 * 4),
            c.table(bm, budget=64_000_000),
            rtol=0.0,
            atol=1e-5,
        )

    def test_table_does_not_build_a_graph(self):
        c = Code(dims=512, seed=1)
        bm = torch.from_numpy(byte_matrix(["a", "b"], 10)).long()

        assert not c.table(bm, budget=64_000_000).requires_grad

    def test_vocabulary_table_matches_global_encoding_and_preserves_order(self):
        c = Code(dims=512, seed=1)
        vocab = ["a", "quantum", "量子", "x" * 80, " end"]
        expected = c(torch.from_numpy(byte_matrix(vocab)).long())
        actual = c.vocabulary_table(vocab, budget=5 * 512 * 4)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    def test_vocabulary_table_accepts_one_unit_wider_than_its_budget(self):
        c = Code(dims=512, seed=1)
        unit = "x" * 100
        actual = c.vocabulary_table([unit], budget=512 * 4)
        assert c.P.shape[0] == 0
        expected = Code(dims=512, seed=1)(torch.from_numpy(byte_matrix([unit])).long())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    def test_large_following_row_does_not_stream_a_small_row(self):
        code = Code(dims=512, seed=1)
        code.vocabulary_table(["a", "x" * 100], budget=512 * 12)

        assert code.P.shape[0] == 1

    def test_vocabulary_table_rejects_nonpositive_budget(self):
        with pytest.raises(ValueError, match="budget"):
            Code(dims=512, seed=1).vocabulary_table(["a"], budget=0)

    def test_empty_vocabulary_preserves_codebook_dtype(self):
        code = Code(dims=512, seed=1).to(dtype=torch.bfloat16)

        assert code.vocabulary_table([], budget=1).dtype == torch.bfloat16

    def test_vocabulary_table_encodes_each_string_once(self):
        calls = []

        class CountedString(str):
            def encode(self, encoding="utf-8", errors="strict"):
                calls.append(self)
                return super().encode(encoding, errors)

        vocab: list[str] = [CountedString("one"), CountedString(" longer")]
        Code(dims=512, seed=1).vocabulary_table(vocab, budget=512 * 4)
        assert calls == vocab

    def test_position_growth_preserves_module_dtype(self):
        code = Code(dims=512, seed=1).to(dtype=torch.bfloat16)
        matrix = torch.from_numpy(byte_matrix(["a long unit"])).long()
        encoded = code(matrix)
        assert code.P.dtype == torch.bfloat16
        assert encoded.dtype == torch.bfloat16
