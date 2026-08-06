"""Segmentation metrics. Pure Python -- no torch, no GPU, no model.

Ported from the research `evaluate.py`, which produced every number in the
published tables. This module is importable from the base install precisely so
that scoring a prediction file does not require a 2 GB dependency.

Three families, measuring different things:

* **Boundary matching** (`match`, `prf`) -- did a prediction land within
  `tolerance` words of a gold boundary? Strict, and the headline number.
* **Segmentation agreement** (`pk`, `windowdiff`) -- how often do two
  segmentations disagree about whether a pair of positions belongs to the same
  story? Lower is better, and unlike F1 they degrade gracefully with near
  misses.
* **Baselines** (`baseline_none`, `baseline_uniform`) -- what a trivial
  predictor scores on the same documents. Without these an F1 means nothing.

Caveat C1 applies to everything here: a large share of scored false positives
are real topic changes the annotator grouped into one thematic block, so
precision is a lower bound. **Do not compute a derived statistic that treats a
false positive as clean error.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

DEFAULT_TOLERANCE_WORDS = 25
"""Words a prediction may miss a gold boundary by and still count as a hit.

Annotators placed offsets at the first word of the new story; a tolerance
absorbs disagreement about whether the tease or handoff belongs to the old story
or the new one. +/-25 is strict and is what the model was selected against;
+/-100 answers the looser question "did it find the seam at all?".
"""


class Match(NamedTuple):
    """Outcome of matching one document's predictions against its gold.

    Attributes:
        pairs: `(gold, pred)` offsets that matched, in gold order.
        n_missed: Gold boundaries with no prediction within tolerance.
        n_spurious: Predictions matching no gold boundary.
    """

    pairs: list[tuple[int, int]]
    n_missed: int
    n_spurious: int

    @property
    def mean_offset(self) -> float:
        """Mean absolute distance, in words, over matched pairs.

        Returns:
            The mean, or 0.0 when nothing matched. Reports *placement* error
            among hits, which is what separates a boundary found precisely from
            one found approximately.
        """
        if not self.pairs:
            return 0.0
        return sum(abs(g - p) for g, p in self.pairs) / len(self.pairs)


class Score(NamedTuple):
    """Precision, recall and F1.

    Attributes:
        precision: tp / (tp + fp). A **lower bound** under C1, not an estimate.
        recall: tp / (tp + fn).
        f1: Harmonic mean.
        tp: True positives.
        fn: False negatives.
        fp: False positives.
    """

    precision: float
    recall: float
    f1: float
    tp: int
    fn: int
    fp: int


def match(
    gold: Sequence[int],
    pred: Sequence[int],
    tolerance: int = DEFAULT_TOLERANCE_WORDS,
) -> Match:
    """Pair predictions to gold boundaries, nearest pair first.

    Every candidate pair within `tolerance` is ranked by distance globally, then
    consumed best-first; each gold boundary absorbs at most one prediction and
    vice versa. So two predictions clustered on one true boundary yield one hit
    and one false positive rather than free credit.

    The global ordering matters. Walking gold left-to-right and taking each
    one's nearest free prediction gives identical *counts* but different
    *pairings* once the tolerance is wide enough for candidates to compete --
    measured on the published test split, the two agree at +/-25 and +/-50 and
    diverge at +/-100 (mean offset 12.2 words against 14.8). This is the
    reference behaviour and reproduces the published tables.

    Args:
        gold: Gold boundary offsets.
        pred: Predicted boundary offsets.
        tolerance: Maximum distance in words for a pair to count.

    Returns:
        The pairing and its miss counts.
    """
    gold, pred = list(gold), list(pred)
    candidates = sorted(
        (abs(g - p), gi, pi)
        for gi, g in enumerate(gold)
        for pi, p in enumerate(pred)
        if abs(g - p) <= tolerance
    )

    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _dist, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gold[gi], pred[pi]))

    return Match(pairs, len(gold) - len(used_g), len(pred) - len(used_p))


def prf(tp: int, fn: int, fp: int) -> Score:
    """Precision, recall and F1 from raw counts.

    Args:
        tp: True positives.
        fn: False negatives.
        fp: False positives.

    Returns:
        The three rates plus the counts they came from. All zero when there is
        nothing to score, rather than raising -- a document with no gold and no
        prediction is a legitimate, common case.
    """
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return Score(p, r, f, tp, fn, fp)


def score_documents(
    gold: Sequence[Sequence[int]],
    pred: Sequence[Sequence[int]],
    tolerance: int = DEFAULT_TOLERANCE_WORDS,
) -> Score:
    """Micro-average boundary scores over a corpus.

    Counts are pooled across documents before the rates are computed, so long
    documents carry proportionally more weight -- matching how the published
    tables were produced.

    Args:
        gold: One list of gold offsets per document.
        pred: One list of predicted offsets per document, aligned with `gold`.
        tolerance: Match tolerance in words.

    Returns:
        The pooled score.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(gold) != len(pred):
        msg = f"gold has {len(gold)} documents, pred has {len(pred)}"
        raise ValueError(msg)
    tp = fn = fp = 0
    for g, p in zip(gold, pred, strict=True):
        m = match(g, p, tolerance)
        tp += len(m.pairs)
        fn += m.n_missed
        fp += m.n_spurious
    return prf(tp, fn, fp)


def _segment_ids(breaks: Sequence[int], n_words: int) -> list[int]:
    """Label every word with the index of the story it belongs to.

    Args:
        breaks: Boundary offsets.
        n_words: Document length in words.

    Returns:
        A list of length `n_words`.
    """
    ids, current, cuts = [], 0, set(breaks)
    for i in range(n_words):
        if i in cuts:
            current += 1
        ids.append(current)
    return ids


def pk_and_windowdiff(
    gold: Sequence[int],
    pred: Sequence[int],
    n_words: int,
    k: int | None = None,
) -> tuple[float, float]:
    """Compute Pk and WindowDiff for one document.

    Both slide a window of width `k` and ask whether the two segmentations
    agree. Pk asks "are these two positions in the same story?"; WindowDiff asks
    the stricter "do they contain the same number of boundaries?". Both are
    error rates -- **lower is better** -- and both reward a near miss that F1
    scores as two errors.

    Args:
        gold: Gold boundary offsets.
        pred: Predicted boundary offsets.
        n_words: Document length in words.
        k: Window width. Defaults to half the mean gold segment length, the
            standard choice.

    Returns:
        `(pk, windowdiff)`, or `(nan, nan)` when the document is too short for
        even one window.
    """
    if k is None:
        n_segments = len(gold) + 1
        k = max(1, round(n_words / (2 * n_segments)))
    if n_words <= k:
        return float("nan"), float("nan")

    g_ids = _segment_ids(gold, n_words)
    p_ids = _segment_ids(pred, n_words)
    g_cuts, p_cuts = set(gold), set(pred)

    pk_err = wd_err = n = 0
    for i in range(n_words - k):
        j = i + k
        n += 1
        if (g_ids[i] == g_ids[j]) != (p_ids[i] == p_ids[j]):
            pk_err += 1
        g_n = sum(1 for b in g_cuts if i < b <= j)
        p_n = sum(1 for b in p_cuts if i < b <= j)
        if g_n != p_n:
            wd_err += 1
    return pk_err / n, wd_err / n


def baseline_none(gold: Sequence[int], n_words: int) -> list[int]:
    """Predict no boundaries at all.

    Scores F1 0.000 on any document that has boundaries, which is the point:
    it is the floor a recall-only metric would hide.

    Args:
        gold: Unused; present so baselines share a signature.
        n_words: Unused; present so baselines share a signature.

    Returns:
        An empty list.
    """
    del gold, n_words
    return []


def baseline_uniform(gold: Sequence[int], n_words: int) -> list[int]:
    """Predict `len(gold)` boundaries at even spacing.

    The informative baseline: it is handed the true *number* of boundaries and
    still has to place them. On the published test split it scores F1 0.062, so
    knowing how many stories a broadcast contains is worth almost nothing
    without knowing where they start.

    Args:
        gold: Gold offsets, used only for their count.
        n_words: Document length in words.

    Returns:
        Evenly spaced interior offsets.
    """
    n = len(gold)
    if n == 0:
        return []
    step = n_words / (n + 1)
    return [round(step * (i + 1)) for i in range(n)]
