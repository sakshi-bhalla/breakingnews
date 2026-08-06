"""Guard-zone filtering and cross-window deduplication.

Two required steps between localising an anchor and reporting a boundary.
Ported from `infer.in_guard_zone` / `infer.dedupe`.
"""

from __future__ import annotations

from itertools import pairwise


def in_guard_zone(
    local_word: int,
    n_words: int,
    *,
    is_first_window: bool,
    is_last_window: bool,
    guard_fraction: float = 0.10,
) -> bool:
    """Whether a prediction falls in a window's edge band and should be dropped.

    The model lacks the context at a window's edges to confirm a macro shift,
    and the neighbouring window covers that same text from a better position.

    The document's very first and very last windows have no neighbour, so
    applying the guard there would blind the model to boundaries near the start
    and end of the transcript entirely.

    Args:
        local_word: Offset within the window.
        n_words: Window length in words.
        is_first_window: Whether this is the document's first window.
        is_last_window: Whether this is the document's last window.
        guard_fraction: Fraction of the window discarded at each edge. See
            `Geometry.guard_fraction` for why this is not derived from the
            adapter's `edge_guard_lo`.

    Returns:
        True if the prediction should be discarded.
    """
    margin = int(guard_fraction * n_words)
    if not is_first_window and local_word < margin:
        return True
    return bool(not is_last_window and local_word > n_words - margin)


def dedupe(offsets: list[int], tolerance: int) -> list[int]:
    """Collapse boundaries that overlapping windows found more than once.

    Args:
        offsets: Candidate boundaries in document coordinates, any order.
        tolerance: Merge radius in words. This is a property of window overlap,
            not of annotator disagreement -- do not pass an evaluation
            tolerance here.

    Returns:
        Merged boundaries, ascending, each the rounded mean of its group.
    """
    merged: list[list[int]] = []
    for o in sorted(offsets):
        if merged and o - merged[-1][-1] <= tolerance:
            merged[-1].append(o)
        else:
            merged.append([o])
    return [round(sum(g) / len(g)) for g in merged]


def spans(breaks: list[int], n_words: int) -> list[tuple[int, int]]:
    """Convert boundary offsets into half-open story spans.

    Args:
        breaks: Boundary offsets, ascending.
        n_words: Document length in words.

    Returns:
        `(start, end)` word ranges covering the document with no gaps. A
        document with no boundaries yields one span.
    """
    edges = [0, *breaks, n_words]
    return [(a, b) for a, b in pairwise(edges) if b > a]
