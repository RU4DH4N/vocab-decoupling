import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings, strategies as st

from data.corpus import cap_split, read_text, split_words, to_units

thorough = settings(max_examples=300, deadline=None)


def parquet(tmp_path, name, rows):
    path = tmp_path / name
    pq.write_table(pa.table({"text": rows}), path)

    return path


class TestReadText:
    def test_concatenates_rows_across_files(self, tmp_path):
        a = parquet(tmp_path, "a.parquet", ["ab", "cd"])
        b = parquet(tmp_path, "b.parquet", ["ef"])

        assert read_text([a, b], max_chars=100) == "ab\ncd\nef"

    def test_empty_rows_are_skipped(self, tmp_path):
        path = parquet(tmp_path, "gaps.parquet", ["ab", "", "cd", ""])

        assert read_text([path], max_chars=100) == "ab\ncd"

    def test_stops_once_the_budget_is_reached(self, tmp_path):
        path = parquet(tmp_path, "rows.parquet", ["123", "456", "789"])

        assert read_text([path], max_chars=6) == "123\n45"
        assert read_text([path], max_chars=4) == "123"

    def test_does_not_return_a_separator_without_row_content(self, tmp_path):
        path = parquet(tmp_path, "rows.parquet", ["abc", "def"])
        assert read_text([path], max_chars=4) == "abc"

    def test_rows_cannot_fuse_into_a_synthetic_word(self, tmp_path):
        path = parquet(tmp_path, "chapters.parquet", ["the end", "Chapter one"])
        assert read_text([path], 100) == "the end\nChapter one"

    @pytest.mark.parametrize("max_chars", [0, -1, -999])
    def test_a_non_positive_budget_reads_nothing(self, tmp_path, max_chars):
        path = parquet(tmp_path, "rows.parquet", ["123"])

        assert read_text([path], max_chars=max_chars) == ""

    def test_no_files_is_empty(self):
        assert read_text([], max_chars=100) == ""


class TestSplitWords:
    def test_whitespace_attaches_to_the_word_that_follows_it(self):
        assert split_words("ab cd") == ["ab", " cd"]
        assert split_words("  ab  cd ") == ["  ab", "  cd", " "]

    def test_zero_width_space_is_not_a_split_point(self):
        assert split_words("​​") == ["​​"]

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " ",
            " \n \t \r ",
            "A" * 10_000,
            "mixed \x00 control and \U0001f600 emoji",
        ],
    )
    def test_degenerate_text_survives_a_round_trip(self, text):
        assert "".join(split_words(text)) == text

    @given(st.text())
    @thorough
    def test_splitting_is_lossless(self, text):
        assert "".join(split_words(text)) == text


class TestCapSplit:
    def test_short_units_are_returned_whole(self):
        assert cap_split("test", max_bytes=4) == ["test"]

    def test_splits_on_encoded_width_not_character_count(self):
        assert cap_split("aé", max_bytes=2) == ["a", "é"]
        assert cap_split("éa", max_bytes=3) == ["éa"]

    @pytest.mark.parametrize(
        "unit, max_bytes, expected",
        [
            ("\U0001f60a", 1, ["\U0001f60a"]),
            ("a\U0001f60ab", 1, ["a", "\U0001f60a", "b"]),
            ("test", -5, ["t", "e", "s", "t"]),
        ],
    )
    def test_a_character_wider_than_the_cap_is_never_broken(
        self, unit, max_bytes, expected
    ):
        assert cap_split(unit, max_bytes) == expected

    @given(st.text(), st.integers(min_value=1, max_value=100))
    @thorough
    def test_chunks_fit_the_cap_unless_a_single_character_cannot(self, text, max_bytes):
        chunks = cap_split(text, max_bytes)

        assert "".join(chunks) == text

        for chunk in chunks:
            if len(chunk.encode("utf-8")) > max_bytes:
                assert len(chunk) == 1, f"{chunk!r} exceeds {max_bytes} bytes"


class TestToUnits:
    def test_words_are_split_then_capped(self):
        assert to_units("hello  world", max_bytes=3) == ["hel", "lo", "  w", "orl", "d"]

    @given(st.text(), st.integers(min_value=1, max_value=50))
    @thorough
    def test_segmentation_is_lossless_and_respects_the_cap(self, text, max_bytes):
        units = to_units(text, max_bytes)

        assert "".join(units) == text

        for unit in units:
            if len(unit.encode("utf-8")) > max_bytes:
                assert len(unit) == 1, f"{unit!r} exceeds {max_bytes} bytes"


class TestUncappedUnits:
    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_uncapped_is_lossless(self, text):
        assert "".join(to_units(text, max_bytes=None)) == text

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_uncapped_is_exactly_split_words(self, text):
        assert to_units(text, max_bytes=None) == split_words(text)

    def test_long_words_survive_intact(self):
        text = "the kaksikymmentaviisikirjaiminen sana"

        assert "kaksikymmentaviisikirjaiminen" in "".join(
            to_units(text, max_bytes=None)
        )
        assert " kaksikymmentaviisikirjaiminen" in to_units(text, max_bytes=None)

    def test_capping_is_opt_in(self):
        text = "the internationalisation of things"

        assert len(to_units(text, 5)) > len(to_units(text, max_bytes=None))


def _split_words_by_hand(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for char in text:
        if char.isspace() and cur and not cur[-1].isspace():
            out.append("".join(cur))
            cur = []
        cur.append(char)
    if cur:
        out.append("".join(cur))
    return out


class TestSplitWordsMatchesTheCharacterWalk:
    @given(st.text(alphabet=st.characters(), min_size=0, max_size=60))
    @thorough
    def test_arbitrary_text(self, text):
        assert split_words(text) == _split_words_by_hand(text)

    @given(
        st.lists(
            st.sampled_from(
                [" ", "\t", "\n", "\xa0", " ", "　", "\x1c", "\x85", "a", "é", "日", "​"]
            ),
            max_size=30,
        )
    )
    @thorough
    def test_whitespace_heavy_text(self, pieces):
        text = "".join(pieces)
        assert split_words(text) == _split_words_by_hand(text)

    def test_regex_whitespace_agrees_with_str_isspace_for_every_code_point(self):

        pattern = re.compile(r"\s")
        for point in range(0x110000):
            char = chr(point)
            assert bool(pattern.fullmatch(char)) == char.isspace(), hex(point)
