"""Frozen configuration objects.

The research code kept all of this as module-level globals in `config.py`, and
`infer.apply_saved_geometry()` *rewrote* those globals at load time::

    C.WINDOW_TOKENS = g["window_tokens"]
    C.STRIDE_TOKENS = g["stride_tokens"]

That is workable in a script and unusable in a library: two adapters with
different geometry cannot coexist in one process, the mutation leaks between
tests, and nothing can be constructed twice. Geometry is therefore instance
state here, carried by the `Segmenter` that read it, and every object in this
module is frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TAU = 0.010
"""Decision threshold on `P("1")`, as used for the published results.

**This is a guard, not a tuning knob, and specifically not a precision dial.**

V4_hard's confidences are saturated and bimodal: 49% of validation windows sit
above 0.5 and 38% below 0.001, p25 = 0.000 and p75 = 0.996. The entire 500x
sweep of tau moves 12% of windows; F1 across that whole range spans 0.589-0.602
and precision 0.548-0.583. For the w3072 family, `docs/MODELS.md` records greedy
and thresholded F1 identical at 0.6250.

The one place tau matters is zero, where every window fires and F1 collapses. So
any value in roughly [0.005, 0.5] is equivalent, and the parameter exists to
exclude that degenerate case.

**Caveat C7 belongs anywhere tau is exposed.** The deployed geometry has no
high-precision regime: `V2_w3072` cannot exceed precision 0.564 at *any*
threshold, while `V2_w2048` -- same peak F1, confidences better spread --
reaches 0.810. A downstream use sensitive to false boundaries cannot be served
by raising tau on this model. The fix is a different geometry.

0.010 is the default because it produced the published numbers.
"""

DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
"""Ungated mirror of meta-llama/Llama-3.1-8B-Instruct.

Same four bf16 shards and the same `LlamaForCausalLM` config, but no HF token
required. Swap to the meta-llama repo if you would rather have official
provenance and can supply a token.
"""

BREAK_TOKEN_ID = 128256
"""Vocabulary index of `<|STORY_BREAK|>` after the base model is resized.

Llama-3.1's stock vocabulary is 128,256 entries (0..128,255), so the added
token lands at exactly this index. `test_artifacts` asserts it.
"""

ROWS_FILENAME = "story_break_rows.safetensors"
"""Optional 16 KB artefact carrying the two absolute embedding rows.

See `loading.install_break_token` for why it exists and what happens without it.
"""


@dataclass(frozen=True)
class PromptSpec:
    """The trained prompt contract.

    These are not tuning knobs. The adapter was fine-tuned against this exact
    template with these exact anchor widths; changing any field produces inputs
    unlike anything the model saw in training. They are a dataclass only so
    that they travel as data rather than as importable globals.

    Attributes:
        story_break_token: Sentinel emitted between the pre- and post-context
            of each predicted boundary.
        no_break_target: Literal the model emits for a window with no boundary.
        anchor_pre_words: Words quoted before a boundary. Twelve pre-context
            words are unique within a +/-1600-word neighbourhood for 99.93% of
            annotated breaks, which is what makes an anchor localisable back to
            a word offset by string search.
        anchor_post_words: Words quoted after a boundary.
        template: Format string with `pre`, `post`, `tok`, `none` and
            `input_text` fields.
    """

    story_break_token: str = "<|STORY_BREAK|>"  # noqa: S105
    no_break_target: str = "NONE"
    anchor_pre_words: int = 12
    anchor_post_words: int = 8
    template: str = (
        "You are segmenting a television news transcript into distinct stories.\n"
        "Below is a slice of one broadcast. Mark every point where the broadcast "
        "moves from one story to a genuinely different story - a new topic, a new "
        "event, different actors. Do not mark a change of speaker, correspondent, "
        "location, or sub-angle within a continuing story.\n\n"
        "For each boundary, quote the {pre} words immediately before it, then "
        "{tok}, then the {post} words immediately after. "
        "If there are no boundaries, answer {none}.\n\n"
        "Transcript:\n{input_text}\n\nBoundaries:\n"
    )

    def render(self, window_text: str) -> str:
        """Build the prompt for one window.

        Args:
            window_text: The window's words, joined by single spaces.

        Returns:
            The full prompt string, ending at the point where the model must
            emit its decision token.
        """
        return self.template.format(
            pre=self.anchor_pre_words,
            post=self.anchor_post_words,
            tok=self.story_break_token,
            none=self.no_break_target,
            input_text=window_text,
        )


@dataclass(frozen=True)
class Geometry:
    """Window geometry, read from the adapter rather than from a default.

    Geometry travels with the adapter because it is a property of how the model
    was trained, not of how a caller wants to run it. V4_hard was trained at
    3072/1536 with a 200-token guard; the research `config.py` defaulted to
    4096/3072/400. Running the adapter at the defaults silently feeds it slices
    unlike anything in training and costs real accuracy, with no error raised.

    Attributes:
        window_tokens: Window width in tokens.
        stride_tokens: Step between window starts.
        edge_guard_lo: Training-time guard band, in tokens, at each window edge.
            Recorded for provenance; see `guard_fraction` for what inference
            actually applies.
        max_seq_len: Training-time sequence cap. Unused at inference.
        tail_window_min_advance: A trailing end-anchored window is appended only
            when it advances at least this far past the last grid window.
        guard_fraction: Fraction of a window's words discarded at each edge at
            inference time.

            This is deliberately *not* derived from `edge_guard_lo`. The
            research `in_guard_zone()` hardcoded `0.10 * n_words` and ignored
            the `edge_guard_lo` that `apply_saved_geometry()` had just loaded,
            so inference discarded a wider band (10%) than training clipped for
            (200/3072 = 6.5%). Every number in `results/` was produced with
            10%, so 10% is the default here and changing it invalidates the
            published metrics. It is exposed because narrowing it is a
            plausible free-recall experiment, not because it is a preference.
    """

    window_tokens: int
    stride_tokens: int
    edge_guard_lo: int
    max_seq_len: int = 4096
    tail_window_min_advance: int = 256
    guard_fraction: float = 0.10

    @classmethod
    def from_adapter(cls, adapter_dir: str | Path, **overrides: object) -> Geometry:
        """Read `segmentation_config.json` out of an adapter directory.

        Args:
            adapter_dir: A resolved local adapter directory.
            **overrides: Inference-time fields to override. Only
                `guard_fraction` and `tail_window_min_advance` may be
                overridden. `window_tokens`, `stride_tokens` and
                `edge_guard_lo` are properties of how the adapter was trained,
                not of how you want to run it -- changing them without
                retraining is breakage, not tuning, so they are rejected.

        Returns:
            The geometry the adapter was trained with.

        Raises:
            FileNotFoundError: If the adapter carries no geometry file. This is
                a hard error rather than a warning-and-default: the research
                code warned and fell back to `config.py`, which is precisely
                the silent mismatch described above.
            ValueError: If a trained-in field is passed as an override.
        """
        locked = {"window_tokens", "stride_tokens", "edge_guard_lo", "max_seq_len"}
        if bad := locked & set(overrides):
            msg = (
                f"cannot override {sorted(bad)}: these are properties of how the "
                "adapter was trained. Running at a different window size feeds the "
                "model slices unlike anything it saw and fails silently. Construct "
                "Geometry(...) directly if you genuinely mean to."
            )
            raise ValueError(msg)
        path = Path(adapter_dir) / "segmentation_config.json"
        if not path.exists():
            msg = (
                f"no segmentation_config.json in {adapter_dir}. Refusing to guess "
                "window geometry -- an adapter run at the wrong window size fails "
                "silently and looks like a bad model. Pass a Geometry explicitly "
                "if you know what this adapter was trained with."
            )
            raise FileNotFoundError(msg)
        saved = json.loads(path.read_text(encoding="utf-8"))
        fields = {
            "window_tokens": saved["window_tokens"],
            "stride_tokens": saved["stride_tokens"],
            "edge_guard_lo": saved["edge_guard_lo"],
            "max_seq_len": saved.get("max_seq_len", 4096),
        }
        return cls(**{**fields, **overrides})  # type: ignore[arg-type]


@dataclass(frozen=True)
class DecodeSpec:
    """Inference-time knobs that do not affect what the model is.

    Attributes:
        max_new_tokens: Generation cap. 256 is ample -- an 18-break window needs
            roughly 380 tokens of anchors and the median target is well under
            100 -- and generation stops at EOS anyway, so this only bounds a
            rambling model, which is the case worth bounding.
        gen_batch_size: Windows generated concurrently. Halved automatically on
            CUDA OOM; see `Segmenter._run_pool`.

            **Not purely a throughput knob.** Batching left-pads prompts to the
            longest in the batch, and in bf16 that padding perturbs the
            numerics enough to occasionally change a generated anchor, which
            moves the offset it localises to. Measured on one 8,909-word
            document: five of six boundaries were identical at every batch
            size, and one moved by +7 words at batch 1 and a different one by
            -29 words at batch 4, while batches 8 and 16 reproduced the
            reference exactly. Batch *composition* matters too, so pooling
            different documents together can shift a boundary at the same batch
            size.

            Offsets are therefore reproducible in aggregate, not bit-exact.
            Shifts of this size are small against the +/-25 and +/-100 scoring
            tolerances, but they are not zero. Fix the batch size if you need
            byte-identical output across runs.
        dedupe_tolerance_words: Radius within which two predictions from
            overlapping windows are merged into one.

            Separate from the *evaluation* tolerance on purpose. The research
            code used `MATCH_TOLERANCE_WORDS` for both, which silently couples
            a property of window overlap to a property of annotator
            disagreement and makes "score at +/-100" impossible to evaluate
            honestly against predictions deduped at 25.
        strict_anchors: When True, an anchor whose pre-context matches at more
            than one position in the window is discarded as ambiguous rather
            than resolved to the first match.

            Defaults to False, which reproduces the published numbers.
            `locate_anchor()` in the research code contained a dead ambiguity
            check -- both branches returned `hits[0]` -- so multi-match anchors
            have always silently taken the first hit. The rate is expected to
            be low (12 pre-context words are unique for 99.93% of annotated
            breaks) but it has never been measured; `WindowScore.ambiguous`
            counts it so it can be.
    """

    max_new_tokens: int = 256
    gen_batch_size: int = 8
    dedupe_tolerance_words: int = 25
    strict_anchors: bool = False
