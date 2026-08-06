"""Word/token mapping and sliding-window enumeration.

Ported from `build_dataset.py` and `infer.enumerate_windows`, with the module
globals replaced by an explicit `Geometry` argument. The mapping must stay
byte-for-byte equivalent to the training-time version: a break the model saw in
training has to sit at the same relative position it will see here, or the
guard-zone routing no longer matches what the adapter learned.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Geometry, PromptSpec


@dataclass
class Window:
    """One sliding window over a document.

    Attributes:
        doc_index: Index of the source document within the batch.
        word_start: Offset of the window's first word in document coordinates.
        words: The window's words.
        is_first: Whether this is the document's first window.
        is_last: Whether this is the document's last window.
        prompt: Rendered prompt, ending at the decision position.
    """

    doc_index: int
    word_start: int
    words: list[str] = field(repr=False)
    is_first: bool
    is_last: bool
    prompt: str = field(repr=False)


def word_to_token_index(words: list[str], tokenizer: Any) -> tuple[list[int], int]:
    """Map each word index to the first token covering that word.

    Computed on the whitespace-joined text the rows actually carry, using the
    tokenizer's offset mapping rather than a per-word encode, so the result
    matches what the model will be fed exactly.

    Args:
        words: The document's words.
        tokenizer: A fast tokenizer exposing `return_offsets_mapping`.

    Returns:
        A `(w2t, n_tokens)` pair. `w2t` has length `len(words) + 1`; the final
        entry is the token count, so `w2t[i:j]` slicing behaves like any other
        half-open range.
    """
    text = " ".join(words)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    # Character offset at which each word starts (words joined by one space).
    starts, cursor = [], 0
    for w in words:
        starts.append(cursor)
        cursor += len(w) + 1

    w2t, tok_i = [], 0
    for char_start in starts:
        # Advance to the first token whose span reaches this character.
        while tok_i < len(offsets) and offsets[tok_i][1] <= char_start:
            tok_i += 1
        w2t.append(min(tok_i, len(offsets)))
    w2t.append(len(offsets))
    return w2t, len(offsets)


def window_starts(n_tokens: int, geometry: Geometry) -> list[int]:
    """Compute the stride grid of window start offsets, in tokens.

    A trailing end-anchored window is appended so the transcript tail still gets
    full-context coverage, gated on `tail_window_min_advance`. A barely-advanced
    tail window is not merely a duplicate: shifting the frame a few hundred
    tokens lifts tail breaks out of the guard band, so the gate trades
    duplicated windows against coverage rather than trimming pure waste.

    Args:
        n_tokens: Token length of the document.
        geometry: Window geometry from the adapter.

    Returns:
        Token offsets at which windows start, ascending.
    """
    if n_tokens <= geometry.window_tokens:
        return [0]
    starts = list(
        range(0, n_tokens - geometry.window_tokens + 1, geometry.stride_tokens)
    )
    tail_start = n_tokens - geometry.window_tokens
    if tail_start - starts[-1] >= geometry.tail_window_min_advance:
        starts.append(tail_start)
    return starts


def enumerate_windows(
    words: list[str],
    tokenizer: Any,
    geometry: Geometry,
    prompt: PromptSpec,
    doc_index: int = 0,
) -> list[Window]:
    """Window one document. Pure -- no model involved.

    Args:
        words: The document's words.
        tokenizer: Tokenizer used for the word/token map.
        geometry: Window geometry from the adapter.
        prompt: The trained prompt contract.
        doc_index: Index recorded on each window, for pooling across documents.

    Returns:
        The document's windows, in grid order.
    """
    w2t, n_tokens = word_to_token_index(words, tokenizer)
    starts = window_starts(n_tokens, geometry)

    out: list[Window] = []
    for wi, t0 in enumerate(starts):
        t1 = min(t0 + geometry.window_tokens, n_tokens)
        w0 = bisect.bisect_left(w2t, t0)
        w1 = bisect.bisect_left(w2t, t1)
        if w1 <= w0:
            continue
        window_words = words[w0:w1]
        out.append(
            Window(
                doc_index=doc_index,
                word_start=w0,
                words=window_words,
                is_first=wi == 0,
                is_last=wi == len(starts) - 1,
                prompt=prompt.render(" ".join(window_words)),
            )
        )
    return out
