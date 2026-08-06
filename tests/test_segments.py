"""Tests for segment construction and the merge back.

The load-bearing property is the round trip: cutting a document into segments
and reassembling them must return the original text byte-for-byte, including
whitespace that `body.split()` would have destroyed.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from breakingnews.segments import (
    FLAG_SHORT,
    Segment,
    drop_flagged,
    group_by_record,
    make_segment_id,
    merge_segments,
    parse_segment_id,
    to_segments,
    word_char_starts,
)

BODY = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
MESSY = "  alpha bravo\n\ncharlie   delta\techo  "


class TestSegmentId:
    def test_zero_padded(self):
        assert make_segment_id("abc", 0) == "abc#000"
        assert make_segment_id("abc", 42) == "abc#042"

    def test_wide_index_is_not_truncated(self):
        assert make_segment_id("abc", 1234) == "abc#1234"

    def test_round_trips(self):
        assert parse_segment_id(make_segment_id("abc", 7)) == ("abc", 7)

    def test_record_id_containing_the_separator_is_rejected(self):
        # Otherwise the id cannot be parsed back unambiguously.
        with pytest.raises(ValueError, match="separator"):
            make_segment_id("a#b", 0)

    def test_parsing_a_non_id_raises(self):
        with pytest.raises(ValueError, match="not a segment id"):
            parse_segment_id("no-index-here")


class TestWordCharStarts:
    def test_length_is_word_count_plus_one(self):
        assert len(word_char_starts(BODY)) == len(BODY.split()) + 1

    def test_last_entry_is_the_document_length(self):
        assert word_char_starts(BODY)[-1] == len(BODY)

    def test_offsets_point_at_word_starts_in_messy_text(self):
        starts = word_char_starts(MESSY)
        for i, w in enumerate(MESSY.split()):
            assert MESSY[starts[i] : starts[i] + len(w)] == w


class TestToSegments:
    def test_no_boundaries_is_one_segment_covering_everything(self):
        segs = to_segments("r", BODY, [])
        assert len(segs) == 1
        assert segs[0].word_start == 0
        assert segs[0].word_end == len(BODY.split())
        assert segs[0].text == BODY

    def test_boundaries_partition_the_words(self):
        segs = to_segments("r", BODY, [3, 7])
        assert [(s.word_start, s.word_end) for s in segs] == [(0, 3), (3, 7), (7, 10)]
        assert [s.n_words for s in segs] == [3, 4, 3]

    def test_every_segment_carries_its_parent(self):
        for s in to_segments("rec-1", BODY, [3, 7]):
            assert s.record_id == "rec-1"

    def test_ids_are_ordinal_and_contiguous(self):
        segs = to_segments("r", BODY, [3, 7])
        assert [s.segment_id for s in segs] == ["r#000", "r#001", "r#002"]

    def test_char_spans_are_gapless(self):
        segs = to_segments("r", BODY, [3, 7])
        assert segs[0].char_start == 0
        assert segs[-1].char_end == len(BODY)
        for a, b in pairwise(segs):
            assert a.char_end == b.char_start

    def test_text_is_an_exact_substring(self):
        for s in to_segments("r", MESSY, [2]):
            assert s.text == MESSY[s.char_start : s.char_end]

    def test_clean_text_strips(self):
        segs = to_segments("r", MESSY, [2])
        assert segs[0].clean_text == "alpha bravo"

    def test_duplicate_and_unsorted_boundaries_are_normalised(self):
        assert to_segments("r", BODY, [7, 3, 3]) == to_segments("r", BODY, [3, 7])

    @pytest.mark.parametrize("bad", [0, 10, 11, -1])
    def test_boundaries_outside_the_document_raise(self, bad):
        # Offset 0 is not a boundary: every document starts a story.
        with pytest.raises(ValueError, match="outside"):
            to_segments("r", BODY, [bad])

    def test_min_words_flags_but_does_not_drop(self):
        segs = to_segments("r", BODY, [3, 7], min_words=4)
        assert len(segs) == 3
        assert [s.flags for s in segs] == [(FLAG_SHORT,), (), (FLAG_SHORT,)]

    def test_no_flags_without_min_words(self):
        assert all(s.flags == () for s in to_segments("r", BODY, [3, 7]))

    def test_include_text_false_keeps_offsets(self):
        segs = to_segments("r", BODY, [3], include_text=False)
        assert all(s.text == "" for s in segs)
        assert segs[1].word_start == 3
        assert segs[1].char_end == len(BODY)


class TestRoundTrip:
    @pytest.mark.parametrize("body", [BODY, MESSY, "single", "a b"])
    @pytest.mark.parametrize("breaks", [[], [1]])
    def test_merge_reproduces_the_source_byte_for_byte(self, body, breaks):
        n = len(body.split())
        cuts = [b for b in breaks if 0 < b < n]
        record_id, rebuilt = merge_segments(to_segments("r", body, cuts))
        assert record_id == "r"
        assert rebuilt == body

    def test_whitespace_a_naive_rejoin_would_destroy_survives(self):
        _, rebuilt = merge_segments(to_segments("r", MESSY, [2]))
        assert rebuilt == MESSY
        assert rebuilt != " ".join(MESSY.split())  # the naive version differs

    def test_merge_accepts_segments_in_any_order(self):
        segs = to_segments("r", BODY, [3, 7])
        assert merge_segments(list(reversed(segs)))[1] == BODY


class TestMergeRefusals:
    """Each of these would otherwise yield a plausible but wrong document."""

    def test_empty_input(self):
        with pytest.raises(ValueError, match="zero segments"):
            merge_segments([])

    def test_mixed_records(self):
        a = to_segments("a", BODY, [])
        b = to_segments("b", BODY, [])
        with pytest.raises(ValueError, match="span 2 record_ids"):
            merge_segments([*a, *b])

    def test_a_missing_segment(self):
        segs = to_segments("r", BODY, [3, 7])
        with pytest.raises(ValueError, match="missing or duplicated"):
            merge_segments([segs[0], segs[2]])

    def test_segments_without_text(self):
        segs = to_segments("r", BODY, [3], include_text=False)
        with pytest.raises(ValueError, match="carry no text"):
            merge_segments(segs)

    def test_a_gap_in_the_cover(self):
        segs = to_segments("r", BODY, [3])
        shifted = Segment(**{**segs[1].__dict__, "char_start": segs[1].char_start + 1})
        with pytest.raises(ValueError, match="gapless"):
            merge_segments([segs[0], shifted])


class TestGrouping:
    def test_groups_and_orders(self):
        segs = [*to_segments("a", BODY, [3]), *to_segments("b", BODY, [])]
        grouped = group_by_record(reversed(segs))
        assert sorted(grouped) == ["a", "b"]
        assert [s.index for s in grouped["a"]] == [0, 1]

    def test_drop_flagged_returns_both_halves(self):
        segs = to_segments("r", BODY, [3, 7], min_words=4)
        kept, dropped = drop_flagged(segs)
        assert len(kept) == 1
        assert len(dropped) == 2

    def test_dropping_breaks_the_cover_on_purpose(self):
        # Once rows are gone the remainder is not a document; merge must refuse.
        kept, _ = drop_flagged(to_segments("r", BODY, [3, 7], min_words=4))
        with pytest.raises(ValueError, match="missing or duplicated"):
            merge_segments(kept)

    def test_nothing_dropped_when_nothing_is_flagged(self):
        segs = to_segments("r", BODY, [3, 7])
        kept, dropped = drop_flagged(segs)
        assert kept == segs
        assert dropped == []
