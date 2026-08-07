"""Tests for the scoring metrics.

The values pinned in `TestPublishedTable` are the v1 test-split numbers
from the paper. They were reproduced exactly by this module against the
reference prediction file, so a change here is a change to the published
results.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from breakingnews.metrics import (
    DEFAULT_TOLERANCE_WORDS,
    baseline_none,
    baseline_uniform,
    match,
    pk_and_windowdiff,
    prf,
    score_documents,
)


class TestMatch:
    def test_exact_hit(self):
        m = match([100], [100], 25)
        assert m.pairs == [(100, 100)]
        assert (m.n_missed, m.n_spurious) == (0, 0)
        assert m.mean_offset == 0.0

    def test_within_tolerance(self):
        m = match([100], [120], 25)
        assert m.pairs == [(100, 120)]
        assert m.mean_offset == 20.0

    def test_outside_tolerance_is_two_errors(self):
        # One prediction 26 words out scores as a false positive AND the gold
        # boundary as a false negative -- one prediction, two errors.
        m = match([100], [126], 25)
        assert m.pairs == []
        assert (m.n_missed, m.n_spurious) == (1, 1)

    def test_one_gold_absorbs_only_one_prediction(self):
        m = match([100], [98, 102], 25)
        assert len(m.pairs) == 1
        assert (m.n_missed, m.n_spurious) == (0, 1)

    def test_nearest_pair_wins_globally(self):
        # Left-to-right greedy would give gold 100 the prediction at 110, then
        # strand gold 200. Global nearest-first pairs 100->101 and 200->210.
        m = match([100, 200], [101, 210], 100)
        assert sorted(m.pairs) == [(100, 101), (200, 210)]

    def test_empty_inputs(self):
        assert match([], [], 25) == ([], 0, 0)
        assert match([], [50], 25).n_spurious == 1
        assert match([50], [], 25).n_missed == 1

    def test_mean_offset_of_nothing_is_zero(self):
        assert match([], [], 25).mean_offset == 0.0

    def test_default_tolerance(self):
        assert DEFAULT_TOLERANCE_WORDS == 25


class TestPrf:
    def test_perfect(self):
        s = prf(10, 0, 0)
        assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)

    def test_nothing_to_score_is_zero_not_an_error(self):
        # A document with no gold and no prediction is legitimate and common.
        assert prf(0, 0, 0).f1 == 0.0

    def test_harmonic_mean(self):
        s = prf(5, 5, 5)
        assert s.precision == 0.5
        assert s.recall == 0.5
        assert s.f1 == pytest.approx(0.5)


class TestScoreDocuments:
    def test_micro_averages_across_documents(self):
        s = score_documents([[100], [200, 300]], [[100], [200]], 25)
        assert (s.tp, s.fn, s.fp) == (2, 1, 0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="documents"):
            score_documents([[1]], [[1], [2]], 25)


class TestPkWindowDiff:
    def test_identical_segmentations_score_zero(self):
        pk, wd = pk_and_windowdiff([500], [500], 1000)
        assert pk == 0.0
        assert wd == 0.0

    def test_a_near_miss_is_penalised_but_not_fatally(self):
        pk, wd = pk_and_windowdiff([500], [520], 1000)
        assert 0 < pk < 0.25
        assert 0 < wd < 0.25

    def test_missing_a_boundary_is_worse_than_misplacing_it(self):
        near, _ = pk_and_windowdiff([500], [520], 1000)
        gone, _ = pk_and_windowdiff([500], [], 1000)
        assert gone > near

    def test_document_shorter_than_the_window_is_nan(self):
        pk, wd = pk_and_windowdiff([], [], 2, k=10)
        assert math.isnan(pk)
        assert math.isnan(wd)


class TestBaselines:
    def test_none_predicts_nothing(self):
        assert baseline_none([100, 200], 1000) == []

    def test_uniform_matches_the_gold_count(self):
        assert len(baseline_uniform([1, 2, 3], 1000)) == 3

    def test_uniform_is_evenly_spaced(self):
        assert baseline_uniform([1, 2, 3], 1000) == [250, 500, 750]

    def test_uniform_on_a_clean_document(self):
        assert baseline_uniform([], 1000) == []


class TestPublishedTable:
    """v1, test split: 20 transcripts, 64 gold breaks, 86 predicted.

    Reproduced exactly from the reference prediction file. Kept as
    synthetic counts here so the test needs no licensed data.
    """

    @pytest.mark.parametrize(
        ("tol", "tp", "fn", "fp", "precision", "recall", "f1"),
        [
            (25, 51, 13, 35, 0.5930, 0.7969, 0.6800),
            (50, 55, 9, 31, 0.6395, 0.8594, 0.7333),
            (100, 57, 7, 29, 0.6628, 0.8906, 0.7600),
        ],
    )
    def test_counts_give_the_published_rates(
        self, tol, tp, fn, fp, precision, recall, f1
    ):
        del tol
        s = prf(tp, fn, fp)
        assert s.precision == pytest.approx(precision, abs=5e-5)
        assert s.recall == pytest.approx(recall, abs=5e-5)
        assert s.f1 == pytest.approx(f1, abs=5e-5)

    def test_the_no_boundary_baseline_scores_zero(self):
        # Stated in the README; it is what stops recall being gamed.
        assert prf(0, 64, 0).f1 == 0.0


class TestToleranceBoundary:
    """The mutation `<= tolerance` -> `< tolerance` passed the whole suite.

    Nothing exercised the boundary itself, so the constant that decides every
    tp/fn/fp in the published table was never pinned.
    """

    @pytest.mark.parametrize("distance", [0, 1, 24, 25])
    def test_at_or_inside_the_tolerance_is_a_hit(self, distance):
        assert len(match([100], [100 + distance], 25).pairs) == 1

    @pytest.mark.parametrize("distance", [26, 27, 100])
    def test_beyond_the_tolerance_is_two_errors(self, distance):
        m = match([100], [100 + distance], 25)
        assert m.pairs == []
        assert (m.n_missed, m.n_spurious) == (1, 1)

    def test_the_boundary_is_symmetric(self):
        assert len(match([100], [75], 25).pairs) == 1
        assert len(match([100], [74], 25).pairs) == 0

    def test_widening_the_tolerance_resolves_a_double_count(self):
        # One prediction 26 words out is a FP and a FN at +/-25; at +/-100 it is
        # a single hit. This is the mechanism the README describes.
        strict = match([100], [126], 25)
        loose = match([100], [126], 100)
        assert (strict.n_missed, strict.n_spurious) == (1, 1)
        assert (loose.n_missed, loose.n_spurious) == (0, 0)


class TestCountsBehindThePublishedTable:
    """Locks `match` + `score_documents`, not just `prf`.

    `TestPublishedTable` asserts rates from hardcoded counts, so the code that
    *derives* the counts was unpinned. These fixtures are synthetic -- they
    carry no corpus text or real offsets -- but they exercise the same paths.
    """

    GOLD: ClassVar = [[100, 500, 900], [200], [], [50, 400]]

    def test_a_perfect_run(self):
        s = score_documents(self.GOLD, self.GOLD, 25)
        assert (s.tp, s.fn, s.fp) == (6, 0, 0)
        assert s.f1 == 1.0

    def test_predicting_nothing_gives_the_floor(self):
        s = score_documents(self.GOLD, [[] for _ in self.GOLD], 25)
        assert (s.tp, s.fn, s.fp) == (0, 6, 0)
        assert s.f1 == 0.0

    def test_a_document_with_no_gold_only_accrues_false_positives(self):
        pred = [[100, 500, 900], [200], [700], [50, 400]]
        s = score_documents(self.GOLD, pred, 25)
        assert (s.tp, s.fn, s.fp) == (6, 0, 1)

    def test_near_misses_count_at_100_but_not_at_25(self):
        pred = [[140, 500, 900], [200], [], [50, 400]]
        assert score_documents(self.GOLD, pred, 25).tp == 5
        assert score_documents(self.GOLD, pred, 100).tp == 6

    def test_duplicate_predictions_do_not_earn_double_credit(self):
        s = score_documents([[100]], [[98, 102]], 25)
        assert (s.tp, s.fn, s.fp) == (1, 0, 1)

    def test_counts_feed_the_published_rates(self):
        # The +/-25 test-split row: 51 tp, 13 fn, 35 fp -> 0.593 / 0.797 / 0.680.
        s = prf(51, 13, 35)
        assert (round(s.precision, 3), round(s.recall, 3), round(s.f1, 3)) == (
            0.593,
            0.797,
            0.680,
        )
