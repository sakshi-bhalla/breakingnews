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
from breakingnews.postprocess import (
    dedupe,
    in_guard_zone,
    is_degenerate_boundary,
    spans,
)
from breakingnews.segments import to_segments
from breakingnews.windows import window_starts, word_to_token_index

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


class TestFastTokenizerRequired:
    """Window geometry is in tokens; every public offset is a word index. The
    map between them needs an offset mapping, which only a fast tokenizer has.
    """

    class SlowTokenizer:
        is_fast = False

        def __call__(self, *args, **kwargs):
            msg = "a slow tokenizer should never be reached"
            raise AssertionError(msg)

    def test_word_to_token_index_rejects_a_slow_tokenizer(self):
        with pytest.raises(TypeError, match="not a fast tokenizer"):
            word_to_token_index(["a", "b"], self.SlowTokenizer())

    def test_the_error_names_the_cause_and_the_fix(self):
        with pytest.raises(TypeError) as exc:
            word_to_token_index(["a"], self.SlowTokenizer())
        assert "use_fast=True" in str(exc.value)

    def test_a_tokenizer_with_no_is_fast_attribute_is_rejected(self):
        with pytest.raises(TypeError, match="not a fast tokenizer"):
            word_to_token_index(["a"], object())


class TestDegenerateBoundary:
    """Word 0 and the offset past the last word are not boundaries. They are
    reachable only at a document's first and last window, because that is
    exactly where the guard band is deliberately not applied.
    """

    def test_word_zero_on_the_first_window_is_degenerate(self):
        assert is_degenerate_boundary(
            0, 100, is_first_window=True, is_last_window=False
        )

    def test_past_the_end_on_the_last_window_is_degenerate(self):
        assert is_degenerate_boundary(
            100, 100, is_first_window=False, is_last_window=True
        )

    def test_the_same_offsets_mid_document_are_legitimate(self):
        # Mid-document, word_start is positive and the window has a neighbour,
        # so neither offset maps to a document edge.
        assert not is_degenerate_boundary(
            0, 100, is_first_window=False, is_last_window=False
        )
        assert not is_degenerate_boundary(
            100, 100, is_first_window=False, is_last_window=False
        )

    def test_interior_offsets_always_survive(self):
        for first in (True, False):
            for last in (True, False):
                assert not is_degenerate_boundary(
                    50, 100, is_first_window=first, is_last_window=last
                )

    def test_what_survives_is_exactly_what_to_segments_accepts(self):
        # The two functions must agree, or the pipeline raises mid-corpus.
        n = 100
        survivors = [
            w
            for w in range(n + 1)
            if not is_degenerate_boundary(
                w, n, is_first_window=True, is_last_window=True
            )
        ]
        to_segments("r", " ".join(f"w{i}" for i in range(n)), survivors)


class TestSegmentsCliSurvivesABadRecord:
    """A corpus run must not die partway through and leave a file that is
    short but perfectly valid JSON -- the failure that looks like success."""

    def _corpus(self, tmp_path, bad_index):
        import json

        docs = [
            {
                "record_id": f"doc{i:02d}",
                "body": " ".join(f"w{j}" for j in range(50)),
                "word_count": 50,
            }
            for i in range(10)
        ]
        t = tmp_path / "t.jsonl"
        p = tmp_path / "p.jsonl"
        t.write_text("\n".join(json.dumps(d) for d in docs))
        p.write_text(
            "\n".join(
                json.dumps(
                    {
                        "record_id": d["record_id"],
                        "word_count": 50,
                        "pred_breaks": [0] if i == bad_index else [25],
                    }
                )
                for i, d in enumerate(docs)
            )
        )
        return t, p

    def test_the_other_records_are_still_written(self, tmp_path):
        import json

        from breakingnews.cli import main

        t, p = self._corpus(tmp_path, bad_index=5)
        out = tmp_path / "s.jsonl"
        code = main(
            [
                "segments",
                "--transcripts",
                str(t),
                "--predictions",
                str(p),
                "--out",
                str(out),
            ]
        )
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert len({r["record_id"] for r in rows}) == 9
        assert code == 1  # short output must never exit zero

    def test_a_clean_corpus_still_exits_zero(self, tmp_path):
        from breakingnews.cli import main

        t, p = self._corpus(tmp_path, bad_index=-1)
        out = tmp_path / "s.jsonl"
        assert (
            main(
                [
                    "segments",
                    "--transcripts",
                    str(t),
                    "--predictions",
                    str(p),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )


class TestMinimalColumns:
    """`--minimal` is the four-column deliverable; it is lossy on purpose."""

    def _run(self, tmp_path, extra):
        import json

        from breakingnews.cli import main

        body = " ".join(f"w{j}" for j in range(50))
        (tmp_path / "t.jsonl").write_text(
            json.dumps(
                {"record_id": "d0", "body": body, "word_count": 50, "outlet": "CNN"}
            )
        )
        (tmp_path / "p.jsonl").write_text(
            json.dumps({"record_id": "d0", "word_count": 50, "pred_breaks": [25]})
        )
        out = tmp_path / "s.jsonl"
        main(
            [
                "segments",
                "--transcripts",
                str(tmp_path / "t.jsonl"),
                "--predictions",
                str(tmp_path / "p.jsonl"),
                "--out",
                str(out),
                *extra,
            ]
        )
        return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]

    def test_minimal_emits_exactly_the_four_columns(self, tmp_path):
        rows = self._run(tmp_path, ["--minimal"])
        assert all(
            set(r) == {"record_id", "segment_id", "text", "n_cuts"} for r in rows
        )

    def test_the_default_is_the_full_record(self, tmp_path):
        rows = self._run(tmp_path, [])
        assert {"word_start", "char_end", "n_words", "outlet"} <= set(rows[0])

    def test_minimal_still_carries_the_cut_count(self, tmp_path):
        assert all(r["n_cuts"] == 1 for r in self._run(tmp_path, ["--minimal"]))

    def test_minimal_output_is_refused_with_a_useful_message(self, tmp_path):
        # Losing the offsets is the trade. Reading one back must name the cause,
        # not raise a KeyError about some individual field.
        from breakingnews.segments import Segment

        rows = self._run(tmp_path, ["--minimal"])
        with pytest.raises(ValueError, match="--minimal"):
            Segment.from_dict(rows[0])

    def test_a_full_row_reads_back_fine(self, tmp_path):
        from breakingnews.segments import Segment

        assert Segment.from_dict(self._run(tmp_path, [])[0]).n_cuts == 1


class TestSegmentsSurvivesMalformedTranscripts:
    """A corpus-level check must not abort the run a per-record handler exists
    to protect. The drift check once dereferenced `body` outside the try, so
    one bodyless transcript destroyed a completed run's entire output.
    """

    def _corpus(self, tmp_path, bad_body):
        import json

        good = {"record_id": "ok", "body": "a b c d e", "word_count": 5}
        bad = {"record_id": "bad", "word_count": 5}
        if bad_body is not ...:
            bad["body"] = bad_body
        (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(d) for d in (good, bad)))
        (tmp_path / "p.jsonl").write_text(
            "\n".join(
                json.dumps({"record_id": r, "word_count": 5, "pred_breaks": [2]})
                for r in ("ok", "bad")
            )
        )
        return tmp_path / "t.jsonl", tmp_path / "p.jsonl"

    @pytest.mark.parametrize("bad_body", [..., None, 12345, ["a", "b"]])
    def test_one_bad_transcript_does_not_lose_the_others(self, tmp_path, bad_body):
        import json

        from breakingnews.cli import main

        t, p = self._corpus(tmp_path, bad_body)
        out = tmp_path / "s.jsonl"
        code = main(
            [
                "segments",
                "--transcripts",
                str(t),
                "--predictions",
                str(p),
                "--out",
                str(out),
            ]
        )
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert {r["record_id"] for r in rows} == {"ok"}
        assert len(rows) == 2  # the good record still produced its segments
        assert code == 1

    def test_drift_still_aborts_the_whole_run(self, tmp_path):
        # Drift is corpus-level: if predictions were made against different
        # text, every offset is wrong and there is nothing worth writing.
        import json

        from breakingnews.cli import main

        (tmp_path / "t.jsonl").write_text(
            json.dumps({"record_id": "x", "body": "a " * 105, "word_count": 105})
        )
        (tmp_path / "p.jsonl").write_text(
            json.dumps({"record_id": "x", "word_count": 100, "pred_breaks": [50]})
        )
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "segments",
                    "--transcripts",
                    str(tmp_path / "t.jsonl"),
                    "--predictions",
                    str(tmp_path / "p.jsonl"),
                    "--out",
                    str(tmp_path / "s.jsonl"),
                ]
            )
        assert "different text" in str(exc.value)


class TestMergeAccountsForVanishedRecords:
    """`--drop-flagged` can remove every segment of a record, so the record
    disappears entirely. The input universe must be captured before the drop,
    or the loss is invisible: the record never reaches the merge loop, never
    lands in `failures`, and the command exits 0.
    """

    def _corpus(self, tmp_path):
        import json

        from breakingnews.cli import main

        docs = [
            {
                "record_id": "A",
                "body": " ".join(f"w{i}" for i in range(60)),
                "word_count": 60,
            },
            {"record_id": "B", "body": "a b c", "word_count": 3},
        ]
        (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
        (tmp_path / "p.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "record_id": d["record_id"],
                        "word_count": d["word_count"],
                        "pred_breaks": [],
                    }
                )
                for d in docs
            )
        )
        segs = tmp_path / "s.jsonl"
        main(
            [
                "segments",
                "--transcripts",
                str(tmp_path / "t.jsonl"),
                "--predictions",
                str(tmp_path / "p.jsonl"),
                "--out",
                str(segs),
                "--min-words",
                "10",
            ]
        )
        return segs

    def test_a_record_losing_every_segment_is_reported_not_silent(
        self, tmp_path, capsys
    ):
        import json

        from breakingnews.cli import main

        segs = self._corpus(tmp_path)
        out = tmp_path / "r.jsonl"
        code = main(
            ["merge", "--segments", str(segs), "--out", str(out), "--drop-flagged"]
        )
        printed = capsys.readouterr().out
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]

        assert {r["record_id"] for r in rows} == {"A"}  # B is gone
        assert "B" in printed  # and it is named
        assert "biased sample" in printed  # with the caveat
        assert code == 1  # and it cannot pass silently

    def test_the_summary_reports_both_in_and_out(self, tmp_path, capsys):
        from breakingnews.cli import main

        segs = self._corpus(tmp_path)
        main(
            [
                "merge",
                "--segments",
                str(segs),
                "--out",
                str(tmp_path / "r.jsonl"),
                "--drop-flagged",
            ]
        )
        printed = capsys.readouterr().out
        assert "1 records" in printed
        assert "2 record(s) in" in printed

    def test_without_dropping_nothing_vanishes(self, tmp_path):
        from breakingnews.cli import main

        segs = self._corpus(tmp_path)
        out = tmp_path / "r.jsonl"
        assert main(["merge", "--segments", str(segs), "--out", str(out)]) == 0
        assert len(out.read_text().strip().splitlines()) == 2


class TestAnchorWidthsTravelWithTheAdapter:
    """Anchor lengths are part of the trained contract, exactly as window
    geometry is. A 24/16 model decoded at the 12/8 default parses its own
    output at the wrong widths and mislocates every prediction, silently.
    """

    def _adapter(self, tmp_path, **extra):
        import json

        (tmp_path / "segmentation_config.json").write_text(
            json.dumps(
                {
                    "window_tokens": 3072,
                    "stride_tokens": 1536,
                    "edge_guard_lo": 200,
                    "max_seq_len": 4096,
                    **extra,
                }
            )
        )
        return tmp_path

    def test_an_adapter_predating_the_fields_is_12_8(self, tmp_path):
        # The v1 config has no anchor fields. Their value IS 12/8, so the
        # fallback states a fact about those runs rather than guessing.
        spec = PromptSpec.from_adapter(self._adapter(tmp_path))
        assert (spec.anchor_pre_words, spec.anchor_post_words) == (12, 8)

    def test_recorded_widths_are_honoured(self, tmp_path):
        spec = PromptSpec.from_adapter(
            self._adapter(tmp_path, anchor_pre_words=24, anchor_post_words=16)
        )
        assert (spec.anchor_pre_words, spec.anchor_post_words) == (24, 16)

    def test_the_widths_reach_the_prompt(self, tmp_path):
        spec = PromptSpec.from_adapter(
            self._adapter(tmp_path, anchor_pre_words=24, anchor_post_words=16)
        )
        rendered = spec.render("x")
        assert "quote the 24 words" in rendered
        assert "then the 16 words" in rendered

    def test_a_directory_with_no_config_falls_back(self, tmp_path):
        assert PromptSpec.from_adapter(tmp_path).anchor_pre_words == 12


class TestScoreDriftIsChecked:
    """Gold and predictions must describe the same text, or every offset is
    shifted and the score is meaningless while looking perfect.
    """

    def _files(self, tmp_path, pred_wc):
        import json

        (tmp_path / "g.jsonl").write_text(
            json.dumps({"record_id": "d0", "word_count": 105, "breaks": [50]})
        )
        (tmp_path / "p.jsonl").write_text(
            json.dumps({"record_id": "d0", "word_count": pred_wc, "pred_breaks": [50]})
        )
        return tmp_path / "p.jsonl", tmp_path / "g.jsonl"

    def test_a_word_count_mismatch_aborts(self, tmp_path):
        from breakingnews.cli import main

        p, g = self._files(tmp_path, pred_wc=100)
        with pytest.raises(SystemExit) as exc:
            main(["score", "--predictions", str(p), "--gold", str(g)])
        assert "different text" in str(exc.value)

    def test_matching_counts_score_normally(self, tmp_path):
        from breakingnews.cli import main

        p, g = self._files(tmp_path, pred_wc=105)
        assert main(["score", "--predictions", str(p), "--gold", str(g)]) == 0


class TestSegmentsSkipsUnusableOffsets:
    """`except ValueError` alone was too narrow: a schema-legal float offset
    raises TypeError deep in the slicing, and a missing key raises KeyError.
    Either would have killed the corpus run rather than skipping one record.
    """

    @pytest.mark.parametrize(
        ("breaks", "exc_name"),
        [([50.0], "TypeError"), ([None], "TypeError"), (["30"], "TypeError")],
    )
    def test_an_unusable_offset_skips_only_its_record(self, tmp_path, breaks, exc_name):
        import json

        from breakingnews.cli import main

        body = " ".join(f"w{i}" for i in range(100))
        (tmp_path / "t.jsonl").write_text(
            "\n".join(
                json.dumps({"record_id": r, "body": body, "word_count": 100})
                for r in ("ok", "bad")
            )
        )
        (tmp_path / "p.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "record_id": r,
                        "word_count": 100,
                        "pred_breaks": [50] if r == "ok" else breaks,
                    }
                )
                for r in ("ok", "bad")
            )
        )
        out = tmp_path / "s.jsonl"
        code = main(
            [
                "segments",
                "--transcripts",
                str(tmp_path / "t.jsonl"),
                "--predictions",
                str(tmp_path / "p.jsonl"),
                "--out",
                str(out),
            ]
        )
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert {r["record_id"] for r in rows} == {"ok"}
        assert code == 1
        del exc_name
