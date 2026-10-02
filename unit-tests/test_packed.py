import pytest
import torch
from hypothesis import given, strategies as st
from strategies import hypothesis_settings

from data.packed import PackedUnits
from models.shared.symbols import BYTE_IGNORE_INDEX, START, STOP


@given(st.lists(st.text(max_size=12), min_size=1, max_size=12), st.data())
@hypothesis_settings
def test_gathered_rows_match_utf8_reference(units, data):
    packed = PackedUnits.from_units(units, "cpu")
    rows = data.draw(st.lists(st.integers(0, len(units) - 1), min_size=1, max_size=8))
    encoded = [list(units[row].encode("utf-8")) for row in rows]
    width = max(map(len, encoded)) + 1
    expected_previous = [
        [START] + raw + [START] * (width - len(raw) - 1) for raw in encoded
    ]
    expected_targets = [
        raw + [STOP] + [BYTE_IGNORE_INDEX] * (width - len(raw) - 1) for raw in encoded
    ]
    previous, targets, lengths = packed.teacher_forcing(torch.tensor(rows))
    assert previous.tolist() == expected_previous
    assert targets.tolist() == expected_targets
    assert lengths.tolist() == list(map(len, encoded))


def test_rows_are_padded_to_the_longest_selected_unit():
    packed = PackedUnits.from_units(["ab", "abcd"], "cpu")
    previous, targets, _ = packed.teacher_forcing(torch.tensor([0, 1]))
    assert previous.tolist()[0] == [START, ord("a"), ord("b"), START, START]
    assert targets.tolist()[0] == [
        ord("a"),
        ord("b"),
        STOP,
        BYTE_IGNORE_INDEX,
        BYTE_IGNORE_INDEX,
    ]
    assert packed.teacher_forcing(torch.tensor([0]))[1].shape == (1, 3)


def test_rows_must_be_a_non_empty_cpu_index_vector():
    packed = PackedUnits.from_units(["abc", "d"], "cpu")
    with pytest.raises(ValueError, match="CPU index"):
        packed.teacher_forcing(torch.tensor([], dtype=torch.long))


def test_requires_units():
    with pytest.raises(ValueError, match="at least one"):
        PackedUnits.from_units([], "cpu")


def test_empty_unit_retains_stop_and_explicit_width():
    packed = PackedUnits.from_units([""], "cpu")
    previous, targets, lengths = packed.teacher_forcing(torch.tensor([0]), width=3)
    assert previous.tolist() == [[START, START, START]]
    assert targets.tolist() == [[STOP, BYTE_IGNORE_INDEX, BYTE_IGNORE_INDEX]]
    assert lengths.tolist() == [0]


def test_width_cannot_truncate_stop():
    packed = PackedUnits.from_units(["abc"], "cpu")
    with pytest.raises(ValueError, match="below required width"):
        packed.teacher_forcing(torch.tensor([0]), width=3)
