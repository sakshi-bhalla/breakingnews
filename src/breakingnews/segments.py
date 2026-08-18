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
            offsets are not bit-reproducible across batch sizes, so an
            offset-derived or content-hashed id would churn on a re-run that
            found the same stories. An id built from the ordinal
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
        n_cuts: How many boundaries were found in the **parent record**, not in
            this segment. Constant across a record's rows and equal to
            `n_segments - 1`, so 0 means the broadcast was never cut.

            Denormalised onto every row on purpose: it is the field that lets a
            segment-level table be filtered or weighted by how fragmented its
            source broadcast was, without a join back to the predictions. A
            record with 0 cuts is a whole broadcast wearing a segment's
            clothing, and treating it as one story alongside genuine segments
            is the mistake this field exists to make visible.
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
    n_cuts: int
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

    @classmethod
    def from_dict(cls, row: dict) -> Segment:
        """Rebuild a segment from a JSONL row.

        Every reader goes through here rather than hand-rolling the field
        list, so a new field cannot be missed by one caller and not another.

        Args:
            row: A decoded segment row.

        Returns:
            The segment.

        Raises:
            ValueError: If the row carries no offsets. That is what `--minimal`
                output looks like, and the bare KeyError it would otherwise
                raise names a field rather than the cause.
        """
        missing = [
            k
            for k in ("index", "word_start", "word_end", "char_start", "char_end")
            if k not in row
        ]
        if missing:
            msg = (
                f"segment {row.get('segment_id', '?')} has no {missing} -- this "
                "looks like `--minimal` output, which drops the offsets that "
                "merge and reconcile need. Re-run `breakingnews segments` "
                "without --minimal."
            )
            raise ValueError(msg)
        return cls(
            segment_id=row["segment_id"],
            record_id=row["record_id"],
            index=row["index"],
            word_start=row["word_start"],
            word_end=row["word_end"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            n_words=row["n_words"],
            n_cuts=row.get("n_cuts", 0),
            text=row.get("text", ""),
            flags=tuple(row.get("flags", ())),
        )


def make_segment_id(record_id: str, index: int) -> str:
    """Build a segment id.

    Args:
        record_id: The parent broadcast id.
        index: Position within the broadcast, from 0.

    Returns:
        ``"{record_id}#{index:03d}"``. Double backticks are load-bearing:
        Napoleon splits a Returns line on its first colon into type and
        description, and a single-backtick literal does not protect the colon
        in the format spec.

    Raises:
        ValueError: If ``record_id`` contains the separator, which would make
            the id ambiguous to parse.
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
        msg = (
            f"not a segment id: {segment_id!r}. Expected "
            f"'{{record_id}}{SEGMENT_ID_SEPARATOR}{{index}}', e.g. 'abc123#000'."
        )
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
    n_cuts = len(cut)

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
                n_cuts=n_cuts,
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
            carry no text, carry text whose length disagrees with its own
            character span, or do not form a gapless cover. Each of those would
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
    # Keyed on span length, not word count: a whitespace-only record splits to
    # zero words, so an `n_words` test would exempt it, and the empty joins
    # below would then return a truncated body as a success.
    if any(not s.text and s.char_end > s.char_start for s in ordered):
        msg = (
            f"{record_id}: segments carry no text (built with include_text="
            "False), so the document cannot be reassembled from them"
        )
        raise ValueError(msg)

    cursor = 0
    for s in ordered:
        # Byte-exactness is the whole contract, and offsets alone do not
        # establish it: text that was stripped or rewritten downstream still
        # forms a gapless cover, and joining it silently glues words together.
        if s.text and len(s.text) != s.char_end - s.char_start:
            msg = (
                f"{record_id}: segment {s.index} carries {len(s.text)} characters "
                f"but its span is {s.char_end - s.char_start}. The text was "
                "modified after extraction (stripped?), so the merge cannot be "
                "exact."
            )
            raise ValueError(msg)
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


# --- Reconciling two runs ------------------------------------------------------
STATUS_SAME = "same"
STATUS_MOVED = "moved"
STATUS_SPLIT = "split"
STATUS_MERGED = "merged"
STATUS_ADDED = "added"
STATUS_REMOVED = "removed"

STABLE_STATUSES = (STATUS_SAME, STATUS_MOVED)
"""Statuses whose `old_id -> new_id` mapping is unambiguously one-to-one."""


@dataclass(frozen=True)
class Correspondence:
    """How one segment in an old run relates to one in a new run.

    Attributes:
        record_id: The broadcast both sides belong to.
        status: One of `same` (identical span), `moved` (same story, shifted
            boundary), `split` (the old segment became this one and at least one
            other), `merged` (this new segment absorbed more than one old),
            `added` (genuinely new text with no old counterpart), `removed`
            (old text with no new counterpart).

            A segment absorbed by a merge is reported as `merged` pointing at
            the segment that absorbed it, not as `removed` -- it did not vanish,
            it changed owner. `added` and `removed` therefore mean what they
            say, which matters when the counts are used to decide whether a
            re-run is safe to adopt.
        old_id: `segment_id` in the old run, or None when `added`.
        new_id: `segment_id` in the new run, or None when `removed`.
        overlap: Jaccard overlap of the two word ranges, 0.0 to 1.0.
        start_shift: `new.word_start - old.word_start`, or None when there is
            no counterpart at all.

            Read it as "how far the boundary moved" only for `same` and
            `moved`. On a `split` or `merged` row the two segments are not the
            same span, so it is the distance to the segment that absorbed this
            one -- pooling those into a shift statistic produces a meaningless
            maximum. A few words of movement means nothing.
    """

    record_id: str
    status: str
    old_id: str | None
    new_id: str | None
    overlap: float
    start_shift: int | None


def _overlap(a: Segment, b: Segment) -> tuple[int, float]:
    """Word-range intersection and Jaccard overlap of two segments.

    Args:
        a: One segment.
        b: The other.

    Returns:
        An `(intersection_words, jaccard)` pair.
    """
    inter = max(0, min(a.word_end, b.word_end) - max(a.word_start, b.word_start))
    union = a.n_words + b.n_words - inter
    if union:
        return inter, inter / union
    # Both segments are zero-length -- an empty-body record, which the package
    # emits rather than drops. Jaccard is 0/0 there; returning 0.0 would make
    # such a segment unpairable forever, so it would be reported as BOTH split
    # and merged against itself. Identical empty spans are identical.
    same = (a.word_start, a.word_end) == (b.word_start, b.word_end)
    return 0, (1.0 if same else 0.0)


def reconcile(
    old: Iterable[Segment],
    new: Iterable[Segment],
    *,
    min_overlap: float = 0.5,
    min_fragment: float = 0.5,
) -> list[Correspondence]:
    """Map segments from one run onto segments from another.

    `segment_id` is an ordinal, so it is *not* safe to join two runs on it: a
    re-segmentation that finds one extra boundary renumbers everything after it,
    and boundaries can shift by a few words even when the same stories are
    found. This pairs segments by how much text they actually share, which
    survives both.

    Segments are paired one-to-one, highest overlap first, within each
    `record_id`. A pair is reported as `split` or `merged` when a third segment
    also covers a real share of the same text, so a caller can see that the
    correspondence is not one-to-one rather than inferring it from a shift.

    Args:
        old: Segments from the earlier run.
        new: Segments from the later run.
        min_overlap: Jaccard overlap two segments must share to be paired at
            all. The default treats "shares more than half its extent" as the
            same story.
        min_fragment: Share of a *third* segment that must lie inside the
            paired counterpart before the correspondence is called `split` or
            `merged` rather than `moved`.

            Measured against the third segment's own length, not the pair's.
            That is the discriminator: in a real split the extra segment is
            mostly *contained* in the old one, whereas a boundary that merely
            shifted spills only the few words it moved by. Measuring against
            the pair instead makes any one-word shift look like a merge.

    Returns:
        One `Correspondence` per pairing plus one for every unpaired segment on
        each side, so nothing is dropped. An unpaired segment that still sits
        inside one on the other side is reported as `merged` or `split` with a
        pointer to it, rather than as `added`/`removed`.
    """
    old_by_record = group_by_record(old)
    new_by_record = group_by_record(new)

    out: list[Correspondence] = []
    for record_id in sorted(old_by_record.keys() | new_by_record.keys()):
        olds = old_by_record.get(record_id, [])
        news = new_by_record.get(record_id, [])

        # Every candidate pair, best overlap first -- the same global
        # nearest-first discipline `metrics.match` uses, for the same reason:
        # walking one side in order gives different pairings.
        candidates = sorted(
            (
                (-_overlap(o, n)[1], oi, ni)
                for oi, o in enumerate(olds)
                for ni, n in enumerate(news)
                if _overlap(o, n)[1] >= min_overlap
            ),
        )
        used_o: set[int] = set()
        used_n: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for neg_j, oi, ni in candidates:
            if oi in used_o or ni in used_n:
                continue
            used_o.add(oi)
            used_n.add(ni)
            pairs.append((oi, ni, -neg_j))

        for oi, ni, jaccard in sorted(pairs):
            o, n = olds[oi], news[ni]
            # A third segment covering a real share of the same text means the
            # correspondence is not one-to-one, whatever the shift says.
            # A third segment counts only if most of *it* sits inside the
            # counterpart. A shifted boundary spills only as many words as it
            # moved, which must not read as a structural change.
            split = any(
                other.n_words and _overlap(o, other)[0] >= min_fragment * other.n_words
                for j, other in enumerate(news)
                if j != ni
            )
            merged = any(
                other.n_words and _overlap(other, n)[0] >= min_fragment * other.n_words
                for j, other in enumerate(olds)
                if j != oi
            )
            if split:
                status = STATUS_SPLIT
            elif merged:
                status = STATUS_MERGED
            elif (o.word_start, o.word_end) == (n.word_start, n.word_end):
                status = STATUS_SAME
            else:
                status = STATUS_MOVED
            out.append(
                Correspondence(
                    record_id=record_id,
                    status=status,
                    old_id=o.segment_id,
                    new_id=n.segment_id,
                    overlap=jaccard,
                    start_shift=n.word_start - o.word_start,
                )
            )

        # An unpaired segment that nonetheless sits inside one on the other
        # side was absorbed, not lost. Reporting it as `removed` would say text
        # disappeared when it only changed owner -- so it is reported as the
        # losing side of a merge (or split), and keeps a pointer to where it
        # went. `added`/`removed` are then reserved for text with no
        # counterpart at all.
        for oi, o in enumerate(olds):
            if oi in used_o:
                continue
            host = max(
                (
                    x
                    for x in news
                    if o.n_words and _overlap(o, x)[0] >= min_fragment * o.n_words
                ),
                key=lambda x: _overlap(o, x)[0],
                default=None,
            )
            out.append(
                Correspondence(
                    record_id,
                    STATUS_MERGED if host else STATUS_REMOVED,
                    o.segment_id,
                    host.segment_id if host else None,
                    _overlap(o, host)[1] if host else 0.0,
                    host.word_start - o.word_start if host else None,
                )
            )

        for ni, n in enumerate(news):
            if ni in used_n:
                continue
            host = max(
                (
                    x
                    for x in olds
                    if n.n_words and _overlap(x, n)[0] >= min_fragment * n.n_words
                ),
                key=lambda x: _overlap(x, n)[0],
                default=None,
            )
            out.append(
                Correspondence(
                    record_id,
                    STATUS_SPLIT if host else STATUS_ADDED,
                    host.segment_id if host else None,
                    n.segment_id,
                    _overlap(host, n)[1] if host else 0.0,
                    n.word_start - host.word_start if host else None,
                )
            )
    return out


def id_map(correspondences: Iterable[Correspondence]) -> dict[str, str]:
    """Build an `old_id -> new_id` lookup for the unambiguous correspondences.

    Only `same` and `moved` are included. A `split` or `merged` segment has no
    single successor, and silently picking one would quietly corrupt any
    analysis carried across the two runs -- inspect those cases yourself.

    Args:
        correspondences: Output of `reconcile`.

    Returns:
        A mapping covering only the one-to-one pairings.
    """
    return {
        c.old_id: c.new_id
        for c in correspondences
        if c.status in STABLE_STATUSES and c.old_id and c.new_id
    }
