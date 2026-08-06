"""Tests for the pure half of the package -- no model, no GPU.

Expected values are pinned against the research implementation
(`lora_segment_lib/src`), which produced every number in `results/`. A change
here is a change to the predictions, so these are regression locks rather than
examples.
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from breakingnews.anchors import Anchor, localize, locate_anchor, parse_anchors
from breakingnews.config import DEFAULT_TAU, Geometry, PromptSpec
from breakingnews.postprocess import dedupe, in_guard_zone, spans
from breakingnews.windows import window_starts

SPEC = PromptSpec()
V4_GEOMETRY = Geometry(window_tokens=3072, stride_tokens=1536, edge_guard_lo=200)


class TestGeometry:
    def test_reads_the_adapter_not_a_default(self, tmp_path):
        (tmp_path / "segmentation_config.json").write_text(
            json.dumps(
                {
                    "window_tokens": 3072,
                    "stride_tokens": 1536,
                    "edge_guard_lo": 200,
                    "max_seq_len": 4096,
                }
            )
        )
        g = Geometry.from_adapter(tmp_path)
        assert (g.window_tokens, g.stride_tokens, g.edge_guard_lo) == (3072, 1536, 200)

    def test_missing_geometry_raises_rather_than_defaulting(self, tmp_path):
        # The research code warned and fell back to config.py's 4096/3072, which
        # is a silent training/inference mismatch that looks like a bad model.
        with pytest.raises(FileNotFoundError, match="Refusing to guess"):
            Geometry.from_adapter(tmp_path)

    def test_is_frozen(self):
        # apply_saved_geometry() used to rewrite module globals; two adapters in
        # one process clobbered each other.
        with pytest.raises(AttributeError):
            V4_GEOMETRY.window_tokens = 4096  # type: ignore[misc]

    def test_guard_fraction_defaults_to_the_published_value(self):
        # Not derived from edge_guard_lo (200/3072 = 6.5%): every number in
        # results/ was produced at 10%.
        assert V4_GEOMETRY.guard_fraction == 0.10


class TestPromptContract:
    def test_anchor_widths(self):
        assert (SPEC.anchor_pre_words, SPEC.anchor_post_words) == (12, 8)

    def test_sentinels(self):
        assert SPEC.story_break_token == "<|STORY_BREAK|>"  # noqa: S105
        assert SPEC.no_break_target == "NONE"

    def test_render_substitutes_every_field(self):
        out = SPEC.render("alpha bravo")
        assert "{" not in out.replace("{}", "")
        assert out.endswith("Transcript:\nalpha bravo\n\nBoundaries:\n")
        assert "quote the 12 words" in out
        assert "then the 8 words" in out

    def test_default_tau(self):
        assert DEFAULT_TAU == 0.010


class TestWindowStarts:
    def test_short_document_is_one_window(self):
        assert window_starts(500, V4_GEOMETRY) == [0]
        assert window_starts(3072, V4_GEOMETRY) == [0]

    def test_stride_grid(self):
        assert window_starts(6144, V4_GEOMETRY) == [0, 1536, 3072]

    def test_tail_window_appended_when_it_advances_enough(self):
        # last grid start 3072, tail start 4000-3072=928 -> no tail window
        assert window_starts(4000, V4_GEOMETRY) == [0, 928]
        # 8000: grid 0,1536,3072,4608; tail 4928, advance 320 >= 256 -> appended
        assert window_starts(8000, V4_GEOMETRY) == [0, 1536, 3072, 4608, 4928]

    def test_tail_window_suppressed_below_min_advance(self):
        # tail advance of 100 < 256
        assert window_starts(4708, V4_GEOMETRY) == [0, 1536]


WINDOW_TEXT = (
    "the president spoke today in washington about the economy and jobs "
    "meanwhile a powerful storm is moving across the gulf coast tonight"
)
AMBIGUOUS_TEXT = "a b c x a b c y"


class TestAnchors:
    WINDOW = WINDOW_TEXT.split()

    def test_none_answer_yields_nothing(self):
        assert parse_anchors("NONE", SPEC) == []

    def test_numbered_lines_are_stripped(self):
        got = parse_anchors("1. alpha bravo <|STORY_BREAK|> charlie delta", SPEC)
        assert got == [Anchor("alpha bravo", "charlie delta")]

    def test_multiple_anchors(self):
        text = "1) a b <|STORY_BREAK|> c\n2) d e <|STORY_BREAK|> f"
        assert len(parse_anchors(text, SPEC)) == 2

    def test_text_without_the_marker_is_ignored(self):
        assert parse_anchors("garbage with no marker at all", SPEC) == []

    def test_empty_pre_context_is_dropped(self):
        assert parse_anchors("<|STORY_BREAK|> only post", SPEC) == []

    def test_localize_returns_offset_after_the_match(self):
        hit = localize(self.WINDOW, Anchor("about the economy and jobs", "meanwhile"))
        assert hit.offset == 11
        assert self.WINDOW[hit.offset] == "meanwhile"

    def test_localize_falls_back_to_post_context(self):
        hit = localize(self.WINDOW, Anchor("words that are not present", "meanwhile a"))
        assert hit.offset == 11

    def test_localize_falls_back_to_a_shorter_pre_tail(self):
        hit = localize(
            self.WINDOW,
            Anchor("PARAPHRASED PREFIX washington about the economy and jobs", ""),
        )
        assert hit.offset == 11

    def test_unlocatable_anchor_returns_none(self):
        assert (
            localize(self.WINDOW, Anchor("nowhere in this text", "nor this")).offset
            is None
        )

    def test_ambiguity_is_reported_but_resolved_to_the_first_hit(self):
        # Preserves the research behaviour, whose ambiguity check was dead.
        words = AMBIGUOUS_TEXT.split()
        hit = locate_anchor(words, "a b c")
        assert hit.offset == 3
        assert hit.ambiguous is True

    def test_strict_mode_rejects_an_ambiguous_match(self):
        words = AMBIGUOUS_TEXT.split()
        assert localize(words, Anchor("a b c", "zzz"), strict=True).offset is None


class TestGuardZone:
    def test_interior_prediction_survives(self):
        assert not in_guard_zone(500, 1000, is_first_window=False, is_last_window=False)

    def test_leading_edge_dropped_unless_first_window(self):
        assert in_guard_zone(50, 1000, is_first_window=False, is_last_window=False)
        assert not in_guard_zone(50, 1000, is_first_window=True, is_last_window=False)

    def test_trailing_edge_dropped_unless_last_window(self):
        assert in_guard_zone(950, 1000, is_first_window=False, is_last_window=False)
        assert not in_guard_zone(950, 1000, is_first_window=False, is_last_window=True)

    def test_boundary_is_exclusive_at_the_margin(self):
        assert not in_guard_zone(100, 1000, is_first_window=False, is_last_window=True)
        assert in_guard_zone(99, 1000, is_first_window=False, is_last_window=True)


class TestDedupe:
    def test_empty(self):
        assert dedupe([], 25) == []

    def test_groups_within_tolerance_collapse_to_their_mean(self):
        # Chaining is against the group's LAST member, not its first: 40 is 28
        # from 12, so it opens a new group rather than joining [10, 12].
        assert dedupe([10, 12, 40, 41, 42, 500], 25) == [11, 41, 500]

    def test_is_order_independent(self):
        assert dedupe([500, 10, 42, 41, 12, 40], 25) == dedupe(
            [10, 12, 40, 41, 42, 500], 25
        )


class TestSpans:
    def test_no_breaks_is_one_span(self):
        assert spans([], 100) == [(0, 100)]

    def test_breaks_partition_the_document(self):
        assert spans([30, 70], 100) == [(0, 30), (30, 70), (70, 100)]

    def test_spans_are_contiguous_and_cover_everything(self):
        got = spans([30, 70], 100)
        assert got[0][0] == 0
        assert got[-1][1] == 100
        assert all(a[1] == b[0] for a, b in pairwise(got))


class TestGeometryOverrides:
    def _adapter(self, tmp_path):
        (tmp_path / "segmentation_config.json").write_text(
            json.dumps(
                {
                    "window_tokens": 3072,
                    "stride_tokens": 1536,
                    "edge_guard_lo": 200,
                    "max_seq_len": 4096,
                }
            )
        )
        return tmp_path

    def test_inference_field_may_be_overridden(self, tmp_path):
        g = Geometry.from_adapter(self._adapter(tmp_path), guard_fraction=0.065)
        assert g.guard_fraction == 0.065
        assert g.window_tokens == 3072

    @pytest.mark.parametrize(
        "field", ["window_tokens", "stride_tokens", "edge_guard_lo", "max_seq_len"]
    )
    def test_trained_in_field_is_rejected(self, tmp_path, field):
        # Changing window geometry without retraining is breakage, not tuning.
        with pytest.raises(ValueError, match="cannot override"):
            Geometry.from_adapter(self._adapter(tmp_path), **{field: 4096})
