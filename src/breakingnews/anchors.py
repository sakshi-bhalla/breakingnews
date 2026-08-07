"""Parsing generated anchors and grounding them back to word offsets.

The model does not regurgitate the window. It emits roughly twelve words of
context before each boundary and eight after, and the offset is recovered by
locating that quotation inside the window. That design is what makes inference
affordable, and it is also why localisation can fail: a paraphrased or
whitespace-mangled anchor may not match. Misses are counted, never silently
dropped.


"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import PromptSpec

ANCHOR_LINE = re.compile(r"^\s*\d+[.)]\s*(.*)$")


class Anchor(NamedTuple):
    """A quoted boundary emitted by the model.

    Attributes:
        pre: Context quoted immediately before the boundary.
        post: Context quoted immediately after the boundary.
    """

    pre: str
    post: str


class Localization(NamedTuple):
    """Result of grounding one anchor inside a window.

    Attributes:
        offset: Word offset within the window, or None if it could not be
            grounded.
        ambiguous: Whether the pre-context matched at more than one position.
            Reported whether or not the match was used, so the rate can be
            measured; see `DecodeSpec.strict_anchors`.
    """

    offset: int | None
    ambiguous: bool


def parse_anchors(generated: str, prompt: PromptSpec) -> list[Anchor]:
    """Pull anchor pairs out of the model's output.

    Tolerant of missing list numbering and of the model trailing off mid-line.

    Args:
        generated: Decoded model output for one window.
        prompt: The trained prompt contract, for its sentinel strings.

    Returns:
        The anchors found, in emission order. Empty when the model answered
        with the no-break literal.
    """
    if (
        prompt.no_break_target in generated
        and prompt.story_break_token not in generated
    ):
        return []

    anchors: list[Anchor] = []
    for raw in generated.splitlines():
        line = raw.strip()
        if not line or prompt.story_break_token not in line:
            continue
        m = ANCHOR_LINE.match(line)
        if m:
            line = m.group(1)
        pre, _, post = line.partition(prompt.story_break_token)
        pre, post = pre.strip(), post.strip()
        if pre:
            anchors.append(Anchor(pre, post))
    return anchors


def locate_anchor(window_words: list[str], anchor_pre_text: str) -> Localization:
    """Find the word offset a quoted pre-context refers to.

    Inverse of the training target's pre-context: returns the word index
    immediately after the match, which is where the new story begins.

    Args:
        window_words: The window's words.
        anchor_pre_text: The quoted pre-context.

    Returns:
        The offset and whether the match was ambiguous. When several positions
        match, the first is returned and `ambiguous` is True -- preserving the
        research code's behaviour, whose ambiguity check was dead (both
        branches returned `hits[0]`). Callers decide whether to honour it.
    """
    needle = anchor_pre_text.split()
    if not needle:
        return Localization(None, False)
    n = len(needle)
    hits = [
        i + n
        for i in range(len(window_words) - n + 1)
        if window_words[i : i + n] == needle
    ]
    if not hits:
        return Localization(None, False)
    return Localization(hits[0], len(hits) > 1)


def localize(
    window_words: list[str], anchor: Anchor, *, strict: bool = False
) -> Localization:
    """Map a generated anchor back to a word offset inside the window.

    Falls back to the post-context, then to progressively shorter pre-context
    tails, since a slightly paraphrased anchor still usually shares its ending.

    Args:
        window_words: The window's words.
        anchor: The anchor to ground.
        strict: When True, an ambiguous pre-context match is rejected instead of
            resolved to its first hit. Defaults to False, which reproduces the
            published numbers.

    Returns:
        The offset and its ambiguity flag. `offset` is None when the anchor
        cannot be grounded -- those are counted by the caller, not dropped.
    """
    hit = locate_anchor(window_words, anchor.pre)
    if hit.offset is not None and not (strict and hit.ambiguous):
        return hit

    post_words = anchor.post.split()
    if post_words:
        n = len(post_words)
        for i in range(len(window_words) - n + 1):
            if window_words[i : i + n] == post_words:
                return Localization(i, hit.ambiguous)

    tail = anchor.pre.split()
    for k in (8, 6, 4):
        if len(tail) >= k:
            retry = locate_anchor(window_words, " ".join(tail[-k:]))
            if retry.offset is not None and not (strict and retry.ambiguous):
                return Localization(retry.offset, hit.ambiguous or retry.ambiguous)
    return Localization(None, hit.ambiguous)
