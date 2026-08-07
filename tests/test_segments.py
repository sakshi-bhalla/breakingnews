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
    id_map,
    make_segment_id,
    merge_segments,
    parse_segment_id,
    reconcile,
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
        # Shift the whole span, so its length still matches the text and the
        # gap check is what fires rather than the length check.
        segs = to_segments("r", BODY, [3])
        shifted = Segment(
            **{
                **segs[1].__dict__,
                "char_start": segs[1].char_start + 1,
                "char_end": segs[1].char_end + 1,
            }
        )
        with pytest.raises(ValueError, match="gapless"):
            merge_segments([segs[0], shifted])

    def test_text_that_disagrees_with_its_own_span(self):
        # Stripped downstream: still a gapless cover, but joining it glues
        # words together and returns a wrong document as a success.
        segs = to_segments("r", MESSY, [2])
        stripped = [Segment(**{**s.__dict__, "text": s.text.strip()}) for s in segs]
        with pytest.raises(ValueError, match="characters"):
            merge_segments(stripped)


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


class TestReconcile:
    """Two runs of the same model rarely agree exactly; joining on
    `segment_id` would be wrong, because it is an ordinal that renumbers."""

    def _segs(self, breaks):
        return to_segments("r", BODY, breaks)

    def test_identical_runs_are_all_same(self):
        c = reconcile(self._segs([3, 7]), self._segs([3, 7]))
        assert [x.status for x in c] == ["same"] * 3
        assert all(x.start_shift == 0 for x in c)

    def test_a_shifted_boundary_is_moved_not_added(self):
        # A re-run finds the same stories a few words over.
        c = reconcile(self._segs([3, 7]), self._segs([4, 7]))
        assert {x.status for x in c} == {"same", "moved"}
        assert [x.start_shift for x in c] == [0, 1, 0]

    def test_ids_still_map_across_a_shift(self):
        c = reconcile(self._segs([3, 7]), self._segs([4, 7]))
        assert id_map(c) == {"r#000": "r#000", "r#001": "r#001", "r#002": "r#002"}

    def test_an_extra_boundary_renumbers_but_reconciles(self):
        # Old #001 is words 3-10; new splits it, so new #002 is old #001's tail.
        # A naive join on segment_id would silently mismatch every later row.
        c = reconcile(self._segs([3]), self._segs([3, 7]))
        by_old = {x.old_id: x for x in c}
        assert by_old["r#000"].status == "same"
        assert by_old["r#001"].status == "split"

    def test_split_and_merged_are_excluded_from_the_id_map(self):
        # No single successor exists; picking one would corrupt a carried result.
        c = reconcile(self._segs([3]), self._segs([3, 7]))
        assert "r#001" not in id_map(c)

    def test_a_removed_boundary_reports_merged(self):
        c = reconcile(self._segs([3, 7]), self._segs([3]))
        assert any(x.status == "merged" for x in c)

    def test_nothing_is_dropped_from_either_side(self):
        old, new = self._segs([3, 7]), self._segs([5])
        c = reconcile(old, new)
        assert {x.old_id for x in c if x.old_id} == {s.segment_id for s in old}
        assert {x.new_id for x in c if x.new_id} == {s.segment_id for s in new}

    def test_a_record_only_in_the_old_run_is_all_removed(self):
        c = reconcile(to_segments("gone", BODY, []), to_segments("kept", BODY, []))
        assert {x.status for x in c} == {"removed", "added"}

    def test_records_are_never_matched_across_each_other(self):
        c = reconcile(to_segments("a", BODY, []), to_segments("b", BODY, []))
        assert all(x.old_id is None or x.new_id is None for x in c)

    def test_overlap_threshold_blocks_the_same_story_verdict(self):
        # Raising the bar stops segments being called the same story...
        c = reconcile(self._segs([1]), self._segs([9]), min_overlap=0.95)
        assert not {x.status for x in c} & {"same", "moved"}

    def test_but_heavily_overlapping_text_is_never_called_added_or_removed(self):
        # ...without pretending the text vanished. It moved between segments.
        c = reconcile(self._segs([1]), self._segs([9]), min_overlap=0.95)
        assert not {x.status for x in c} & {"added", "removed"}

    def test_an_absorbed_segment_points_at_what_absorbed_it(self):
        # The losing side of a merge did not disappear; it changed owner.
        # Reporting it as `removed` would say text was lost when none was.
        c = reconcile(self._segs([3, 7]), self._segs([7]))
        absorbed = next(x for x in c if x.old_id == "r#001")
        assert absorbed.status == "merged"
        assert absorbed.new_id is not None

    def test_added_and_removed_mean_no_counterpart_at_all(self):
        # Disjoint records: nothing overlaps, so these are the real thing.
        c = reconcile(to_segments("a", BODY, []), to_segments("b", BODY, []))
        assert {x.status for x in c} == {"added", "removed"}


class TestNCuts:
    """`n_cuts` counts boundaries in the parent record, not in the segment."""

    def test_zero_cuts_means_the_record_was_never_segmented(self):
        segs = to_segments("r", BODY, [])
        assert len(segs) == 1
        assert segs[0].n_cuts == 0

    def test_it_equals_the_number_of_boundaries(self):
        assert all(s.n_cuts == 2 for s in to_segments("r", BODY, [3, 7]))

    def test_it_is_constant_across_a_records_rows(self):
        segs = to_segments("r", BODY, [3, 7])
        assert len({s.n_cuts for s in segs}) == 1

    def test_segments_are_always_one_more_than_cuts(self):
        for breaks in ([], [3], [3, 7], [1, 3, 5, 7, 9]):
            segs = to_segments("r", BODY, breaks)
            assert len(segs) == segs[0].n_cuts + 1

    def test_deduplicated_boundaries_are_counted_once(self):
        assert to_segments("r", BODY, [3, 3, 7])[0].n_cuts == 2

    def test_it_survives_the_round_trip_through_jsonl(self):
        import json

        seg = to_segments("r", BODY, [3, 7])[0]
        assert json.loads(json.dumps(seg.to_dict()))["n_cuts"] == 2


class TestCoverageGuarantees:
    """Segmentation is a partition, not extraction. These are the invariants a
    sharded corpus run reconciles against, so they are asserted, not assumed.
    """

    @pytest.mark.parametrize(
        "body", [BODY, MESSY, "one", "a b", "  leading", "trailing  "]
    )
    @pytest.mark.parametrize("cuts", [[], [1], [1, 2]])
    def test_every_character_lands_in_exactly_one_segment(self, body, cuts):
        n = len(body.split())
        segs = to_segments("r", body, [c for c in cuts if 0 < c < n])
        covered = sum(s.char_end - s.char_start for s in segs)
        assert covered == len(body)
        assert segs[0].char_start == 0
        assert segs[-1].char_end == len(body)
        for a, b in pairwise(segs):
            assert a.char_end == b.char_start  # no gap, no overlap

    @pytest.mark.parametrize("body", [BODY, MESSY, "one"])
    def test_words_are_partitioned_too(self, body):
        n = len(body.split())
        segs = to_segments("r", body, [1] if n > 1 else [])
        assert sum(s.n_words for s in segs) == n

    @pytest.mark.parametrize("body", ["", "   ", "\n\t "])
    def test_an_empty_body_yields_a_row_rather_than_vanishing(self, body):
        # Empty broadcasts must be re-emitted, not quietly dropped for having
        # no text -- otherwise records in != records out.
        segs = to_segments("r", body, [])
        assert len(segs) == 1
        assert segs[0].n_words == 0
        assert "empty" in segs[0].flags
        assert segs[0].record_id == "r"

    @pytest.mark.parametrize("body", ["", "   ", "\n\t "])
    def test_an_empty_body_still_round_trips(self, body):
        assert merge_segments(to_segments("r", body, []))[1] == body

    def test_record_ids_out_match_record_ids_in(self):
        ids_in = ["a", "b", "c"]
        segs = [s for r in ids_in for s in to_segments(r, BODY, [3])]
        assert sorted(group_by_record(segs)) == sorted(ids_in)


class TestEmptySegmentsPairWithEachOther:
    """A zero-word segment has Jaccard 0/0. Returning 0.0 made it unpairable,
    so reconciling a run against ITSELF reported one segment as both `split`
    and `merged`, and dropped its id from the map.
    """

    def test_two_identical_empty_runs_are_the_same(self):
        segs = to_segments("e", "", [])
        pairs = reconcile(segs, segs)
        assert len(pairs) == 1
        assert pairs[0].status == "same"
        assert id_map(pairs) == {"e#000": "e#000"}

    def test_an_empty_record_only_in_the_old_run_is_removed(self):
        pairs = reconcile(to_segments("e", "", []), [])
        assert [p.status for p in pairs] == ["removed"]

    def test_a_mixed_corpus_is_unaffected(self):
        old = [*to_segments("e", "", []), *to_segments("n", BODY, [3])]
        pairs = reconcile(old, old)
        assert {p.status for p in pairs} == {"same"}
        assert len(id_map(pairs)) == 3
