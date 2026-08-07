"""Story segmentation for broadcast-news transcripts.

Locates the word offsets where one story ends and the next begins, using a LoRA
adapter on Llama-3.1-8B-Instruct.

    from breakingnews import Segmenter

    seg = Segmenter.from_pretrained("sakshib3/breakingnews", revision="v1")
    breaks = seg.segment(transcript)              # -> [word_offset, ...]
    stories = seg.segment_spans(transcript)       # -> [(start, end), ...]

Boundaries only: nothing here labels, classifies or summarises the resulting
segments.
"""

from __future__ import annotations

from .config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_TAU,
    DecodeSpec,
    Geometry,
    PromptSpec,
)
from .loading import export_break_token_rows, resolve_adapter, verify_adapter
from .metrics import (
    baseline_none,
    baseline_uniform,
    match,
    pk_and_windowdiff,
    prf,
    score_documents,
)
from .postprocess import spans
from .segmenter import Segmenter, WindowScore
from .segments import (
    Correspondence,
    Segment,
    drop_flagged,
    group_by_record,
    id_map,
    make_segment_id,
    merge_segments,
    parse_segment_id,
    reconcile,
    to_segments,
)

__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_TAU",
    "Correspondence",
    "DecodeSpec",
    "Geometry",
    "PromptSpec",
    "Segment",
    "Segmenter",
    "WindowScore",
    "baseline_none",
    "baseline_uniform",
    "drop_flagged",
    "export_break_token_rows",
    "group_by_record",
    "id_map",
    "make_segment_id",
    "match",
    "merge_segments",
    "parse_segment_id",
    "pk_and_windowdiff",
    "prf",
    "reconcile",
    "resolve_adapter",
    "score_documents",
    "spans",
    "to_segments",
    "verify_adapter",
]
