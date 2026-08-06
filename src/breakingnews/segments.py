"""Turning boundaries into segment records, and putting them back together.

`Segmenter` returns word offsets. Downstream analysis wants *rows* -- one per
story, each with a stable identifier that says which broadcast it came from --
and it wants the operation to be reversible, so a segment-level result can be
rolled back up to the broadcast that produced it.

Two properties this module guarantees:

**Provenance never leaves.** Every segment carries its parent `record_id`
verbatim as its own field, not merely encoded in its id. A join back to the
source broadcast is always available and never requires parsing a string.

**The round trip is exact.** `merge_segments(to_segments(...))` returns the
original body byte-for-byte, including whitespace the word split would have
destroyed. Segments are cut on *character* spans, contiguous and gapless, so
concatenating them in order reproduces the input exactly. This matters because
every offset in this project indexes `body.split()`, and naively rejoining with
single spaces silently rewrites any corpus that has newlines or runs of
whitespace in it.

Nothing here drops data. Short segments can be *flagged*, never removed;
`drop_flagged` exists so that discarding is an explicit call by the caller with
a count attached, matching the rest of the pipeline's refusal to lose rows
quietly.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

SEGMENT_ID_SEPARATOR = "#"
SEGMENT_ID_PATTERN = re.compile(r"^(?P<record_id>.+)#(?P<index>\d+)$")
FLAG_SHORT = "short"
FLAG_EMPTY = "empty"


@dataclass(frozen=True)
class Segment:
    """One story, cut out of one broadcast.

    Attributes:
        segment_id: `"{record_id}#{index:03d}"`.

            Stable across re-runs *as long as the number of segments does not
            change*. It deliberately does not encode an offset: predicted
            offsets are not bit-reproducible across batch sizes (LIMITATIONS
            L1b), so an offset-derived or content-hashed id would churn on a
            re-run that found the same stories. An id built from the ordinal
            survives that. What it does not survive is a re-segmentation that
            finds a different *count* -- then everything after the change
            renumbers. Join on `record_id` plus `word_start` if you need to
            reconcile two runs.
        record_id: The parent broadcast, carried verbatim so provenance never
            depends on parsing `segment_id`.
        index: Position within the broadcast, from 0.
        word_start: First word offset, inclusive, into `body.split()`.
        word_end: Last word offset, exclusive.
        char_start: First character offset, inclusive, into `body`.
        char_end: Last character offset, exclusive.
        n_words: `word_end - word_start`.
        text: The exact substring `body[char_start:char_end]`. Includes the
            whitespace that separates this segment from the next, which is what
            makes concatenation lossless; use `clean_text` for display.
        flags: Advisory markers such as `"short"`. Never a reason a row was
            removed -- nothing here removes rows.
    """

    segment_id: str
    record_id: str
    index: int
    word_start: int
    word_end: int
    char_start: int
    char_end: int
    n_words: int
    text: str = field(repr=False)
    flags: tuple[str, ...] = ()

    @property
    def clean_text(self) -> str:
        """The segment text with surrounding whitespace removed.

        Returns:
            `text.strip()`. Use this for analysis and display; use `text` when
            reassembling.
        """
        return self.text.strip()

    def to_dict(self, *, include_text: bool = True) -> dict:
        """Convert to a JSON-serialisable dict.

        Args:
            include_text: When False, omit `text`. Useful when the source is
                licensed and only offsets may be published.

        Returns:
            A dict with `flags` as a list.
        """
        d = asdict(self)
        d["flags"] = list(self.flags)
        if not include_text:
            d.pop("text")
        return d


def make_segment_id(record_id: str, index: int) -> str:
    """Build a segment id.

    Args:
        record_id: The parent broadcast id.
        index: Position within the broadcast, from 0.

    Returns:
        `"{record_id}#{index:03d}"`.

    Raises:
        ValueError: If `record_id` contains the separator, which would make the
            id ambiguous to parse.
    """
    if SEGMENT_ID_SEPARATOR in record_id:
        msg = (
            f"record_id {record_id!r} contains {SEGMENT_ID_SEPARATOR!r}, which "
            "is the segment-id separator. Rename it or segment ids cannot be "
            "parsed back."
        )
        raise ValueError(msg)
    return f"{record_id}{SEGMENT_ID_SEPARATOR}{index:03d}"


def parse_segment_id(segment_id: str) -> tuple[str, int]:
    """Recover the parent record id and index from a segment id.

    Prefer the `record_id` field on a `Segment`; this is for reading ids that
    have travelled through a system that kept only the string.

    Args:
        segment_id: An id produced by `make_segment_id`.

    Returns:
        A `(record_id, index)` pair.

    Raises:
        ValueError: If the id is not in the expected form.
    """
    m = SEGMENT_ID_PATTERN.match(segment_id)
    if not m:
        msg = f"not a segment id: {segment_id!r}"
        raise ValueError(msg)
    return m["record_id"], int(m["index"])


def word_char_starts(body: str) -> list[int]:
    """Map each word index to where that word starts in the original string.

    Computed against `body` itself rather than a normalised rejoin, so the
    spans stay valid for a corpus with newlines or repeated whitespace.

    Args:
        body: The document text.

    Returns:
        A list of length `len(body.split()) + 1`. Entry `i` is the character
        offset of word `i`; the final entry is `len(body)`, so `starts[i:j]`
        slicing behaves like any other half-open range.
    """
    starts = [m.start() for m in re.finditer(r"\S+", body)]
    starts.append(len(body))
    return starts


def to_segments(
    record_id: str,
    body: str,
    breaks: Sequence[int],
    *,
    min_words: int | None = None,
    include_text: bool = True,
) -> list[Segment]:
    """Cut a document into segments at the given boundaries.

    Args:
        record_id: The parent broadcast id.
        body: The document text. Offsets index `body.split()`.
        breaks: Boundary word offsets, each the first word of a new story.
            Sorted and de-duplicated here; out-of-range values are an error
            rather than something to clamp silently.
        min_words: When set, segments shorter than this are flagged `"short"`.
            They are **not** removed -- see `drop_flagged`.
        include_text: When False, `text` is empty. Offsets remain exact, so a
            consumer holding the source can slice it themselves. Use this when
            the text is licensed and cannot be redistributed.

    Returns:
        Segments in document order, contiguous and covering the whole document.
        A document with no boundaries yields exactly one segment.

    Raises:
        ValueError: If a boundary is not strictly inside the document.
    """
    words = body.split()
    n_words = len(words)
    cut = sorted(set(breaks))
    outside = [b for b in cut if not 0 < b < n_words]
    if outside:
        msg = (
            f"{record_id}: boundaries outside (0, {n_words}): {outside}. Offset 0 "
            "is not a boundary -- every document starts a story -- and neither "
            "is the final offset."
        )
        raise ValueError(msg)

    starts = word_char_starts(body)
    edges = [0, *cut, n_words]

    out: list[Segment] = []
    for i, (w0, w1) in enumerate(pairwise(edges)):
        # Segment 0 starts at character 0 and the last ends at len(body), so
        # concatenating every segment reproduces the input exactly -- leading
        # and trailing whitespace included.
        c0 = 0 if i == 0 else starts[w0]
        c1 = len(body) if w1 == n_words else starts[w1]
        length = w1 - w0

        flags: list[str] = []
        if length == 0:
            flags.append(FLAG_EMPTY)
        elif min_words is not None and length < min_words:
            flags.append(FLAG_SHORT)

        out.append(
            Segment(
                segment_id=make_segment_id(record_id, i),
                record_id=record_id,
                index=i,
                word_start=w0,
                word_end=w1,
                char_start=c0,
                char_end=c1,
                n_words=length,
                text=body[c0:c1] if include_text else "",
                flags=tuple(flags),
            )
        )
    return out


def merge_segments(segments: Sequence[Segment]) -> tuple[str, str]:
    """Reassemble one broadcast's segments into the original document.

    The inverse of `to_segments`. Exact: the returned body is byte-identical to
    the input it was cut from.

    Args:
        segments: Segments from a single `record_id`, any order.

    Returns:
        A `(record_id, body)` pair.

    Raises:
        ValueError: If the segments are empty, span more than one `record_id`,
            carry no text, or do not form a gapless cover. Each of those would
            produce a plausible-looking but wrong document, so none is repaired
            silently.
    """
    if not segments:
        msg = "cannot merge zero segments"
        raise ValueError(msg)

    ids = {s.record_id for s in segments}
    if len(ids) != 1:
        msg = f"segments span {len(ids)} record_ids: {sorted(ids)}"
        raise ValueError(msg)
    record_id = ids.pop()

    ordered = sorted(segments, key=lambda s: s.index)
    expected = list(range(len(ordered)))
    if [s.index for s in ordered] != expected:
        msg = (
            f"{record_id}: indices {[s.index for s in ordered]} are not "
            f"0..{len(ordered) - 1}; segments are missing or duplicated"
        )
        raise ValueError(msg)
    if any(not s.text and s.n_words for s in ordered):
        msg = (
            f"{record_id}: segments carry no text (built with include_text="
            "False), so the document cannot be reassembled from them"
        )
        raise ValueError(msg)

    cursor = 0
    for s in ordered:
        if s.char_start != cursor:
            msg = (
                f"{record_id}: segment {s.index} starts at char {s.char_start}, "
                f"expected {cursor} -- segments are not a gapless cover"
            )
            raise ValueError(msg)
        cursor = s.char_end
    return record_id, "".join(s.text for s in ordered)


def group_by_record(segments: Iterable[Segment]) -> dict[str, list[Segment]]:
    """Group segments by their parent broadcast.

    Args:
        segments: Segments from any number of records.

    Returns:
        A mapping from `record_id` to that record's segments, in index order.
    """
    out: dict[str, list[Segment]] = {}
    for s in segments:
        out.setdefault(s.record_id, []).append(s)
    for group in out.values():
        group.sort(key=lambda s: s.index)
    return out


def drop_flagged(
    segments: Sequence[Segment], flags: Iterable[str] = (FLAG_SHORT, FLAG_EMPTY)
) -> tuple[list[Segment], list[Segment]]:
    """Split segments into kept and dropped by flag.

    Deliberately returns both halves rather than filtering in place. Discarding
    rows is a research decision with a count attached, and the count should be
    reportable -- and after this, the survivors no longer form a gapless cover,
    so `merge_segments` will refuse them. That refusal is the point.

    Args:
        segments: Segments to partition.
        flags: Flags that mark a segment for dropping.

    Returns:
        A `(kept, dropped)` pair.
    """
    wanted = set(flags)
    kept, dropped = [], []
    for s in segments:
        (dropped if wanted & set(s.flags) else kept).append(s)
    return kept, dropped
