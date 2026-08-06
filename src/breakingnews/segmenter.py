"""The segmenter.

The decision to emit a boundary is a **single token** -- `"1"` versus `NONE`.
Rather than take the argmax, this reads `P("1")` at the decision position and
fires when it exceeds tau.

**On this model that buys very little, and the library must not claim
otherwise.** V4_hard's confidences are saturated and bimodal: 49% of validation
windows sit above 0.5 and 38% below 0.001, with p25 = 0.000 and p75 = 0.996, so
there is almost nothing in the middle for a threshold to act on. The entire
500x sweep of tau moves 12% of windows and F1 across that whole range spans
0.589-0.602. For the w3072 family V4_hard belongs to, `docs/MODELS.md` records
greedy 0.6250 -> thresholded 0.6250 -- a gain of **0.000**.

The frequently quoted "+0.28 F1 from thresholding" (greedy 0.3291 ->
thresholded 0.6131) belongs to `V2_w2048`, a *different geometry* whose
confidences are spread. It does not describe this model and must not be
repeated of it.

Thresholding is still implemented, and greedy is still not offered as a
default, for three reasons that survive the above: tau = 0 is genuinely harmful
and must be excludable; the sweep-for-free structure lets a user check on their
own corpus whether the bimodality holds there too, which is the cheapest
available test for a domain shift; and a named threshold is auditable in a way
an implicit argmax is not.

The cheap way to threshold, and what `score_windows` implements: generate
**once** per window with the decision *forced* to `"1"`, so every window yields
its anchors regardless of what greedy would have chosen, and read `P("1")` from
a separate forward pass. The threshold then only decides which windows' outputs
to keep, which is free. An arbitrary number of thresholds costs one generation
pass rather than one pass each -- roughly 40x.

This logic previously existed only inside `threshold_sweep.py`'s `main()`,
welded to argparse and a gold set. Everything importable from the research tree
(`infer.predict_documents`) was the greedy path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .anchors import localize, parse_anchors
from .config import DEFAULT_TAU, DecodeSpec, Geometry, PromptSpec
from .loading import load_model, resolve_adapter
from .postprocess import dedupe, in_guard_zone, spans
from .windows import Window, enumerate_windows

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WindowScore:
    """One window's threshold-free result.

    Attributes:
        doc_index: Index of the source document.
        word_start: Offset of the window's first word in document coordinates.
        p_break: `P("1")` at the decision position.

            A thresholded logit, **not** a calibrated probability. It has had no
            temperature fitting and should not be reported as a confidence.
        p_none: `P(NONE)` at the same position, for comparison with what greedy
            decoding would have chosen.
        breaks: Boundary offsets in *document* coordinates, already
            guard-filtered. Present regardless of `p_break`; thresholding
            happens later.
        unlocatable: Anchors the model emitted that could not be grounded back
            to a word offset.
        ambiguous: Anchors whose pre-context matched at more than one position.
    """

    doc_index: int
    word_start: int
    p_break: float
    p_none: float
    breaks: list[int] = field(default_factory=list)
    unlocatable: int = 0
    ambiguous: int = 0

    @property
    def greedy_would_fire(self) -> bool:
        """Whether argmax decoding would have emitted a boundary here."""
        return self.p_break > self.p_none


class Segmenter:
    """Locates story boundaries in broadcast-news transcripts.

    Boundaries only. This does not label, classify or summarise the resulting
    segments.

    Attributes:
        geometry: Window geometry, read from the adapter.
        prompt: The trained prompt contract.
        decode: Inference-time decoding options.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        geometry: Geometry,
        *,
        prompt: PromptSpec | None = None,
        decode: DecodeSpec | None = None,
    ) -> None:
        """Wrap an already-loaded model. Most callers want `from_pretrained`.

        Args:
            model: A PEFT-wrapped causal LM in eval mode.
            tokenizer: Its tokenizer, with `<|STORY_BREAK|>` registered.
            geometry: Window geometry the adapter was trained with.
            prompt: The trained prompt contract. Defaults to the shipped one.
            decode: Decoding options. Defaults to the shipped ones.
        """
        self._model = model
        self._tokenizer = tokenizer
        self.geometry = geometry
        self.prompt = prompt or PromptSpec()
        self.decode = decode or DecodeSpec()

        self._yes_id = tokenizer("1", add_special_tokens=False)["input_ids"][0]
        self._no_id = tokenizer(self.prompt.no_break_target, add_special_tokens=False)[
            "input_ids"
        ][0]
        self._device = next(model.parameters()).device
        self._supports_logits_to_keep = True

    @classmethod
    def from_pretrained(
        cls,
        adapter: str | Path,
        *,
        base_model: str | Path | None = None,
        geometry: Geometry | None = None,
        prompt: PromptSpec | None = None,
        decode: DecodeSpec | None = None,
        **load_kwargs: Any,
    ) -> Segmenter:
        """Load an adapter, from a local directory or the Hugging Face Hub.

        Args:
            adapter: Local adapter directory or Hub repo id.
            base_model: Base to load the adapter onto. Defaults to the adapter's
                recorded base when present locally, else the ungated
                Llama-3.1-8B-Instruct mirror.
            geometry: Override the adapter's recorded geometry. Intended for
                experiments; overriding `window_tokens` or `stride_tokens`
                reintroduces a training/inference mismatch.
            prompt: Override the trained prompt contract. Rarely correct.
            decode: Decoding options.
            **load_kwargs: Forwarded to `loading.load_model` (`dtype`,
                `device_map`, `attn_implementation`).

        Returns:
            A ready segmenter.
        """
        adapter_dir = resolve_adapter(adapter)
        geometry = geometry or Geometry.from_adapter(adapter_dir)
        model, tokenizer = load_model(adapter_dir, base_model=base_model, **load_kwargs)
        return cls(model, tokenizer, geometry, prompt=prompt, decode=decode)

    # ── The threshold-free primitive ──────────────────────────────────────────

    def score_windows(self, documents: Sequence[str]) -> list[list[WindowScore]]:
        """Score every window of every document, without applying a threshold.

        This is the expensive call and the honest one: it runs the model once
        and returns `P("1")` per window alongside that window's anchors, so any
        number of thresholds can be evaluated afterwards for free. Use it to
        sweep tau on your own data rather than re-running generation per tau.

        Args:
            documents: Transcripts, as whitespace-separated text.

        Returns:
            One list of window scores per document, in document order.
        """
        pool: list[Window] = []
        for di, text in enumerate(documents):
            pool.extend(
                enumerate_windows(
                    text.split(), self._tokenizer, self.geometry, self.prompt, di
                )
            )
        scored = self._run_pool(pool)
        out: list[list[WindowScore]] = [[] for _ in documents]
        for s in scored:
            out[s.doc_index].append(s)
        for per_doc in out:
            per_doc.sort(key=lambda s: s.word_start)
        return out

    # ── Thresholded convenience API ───────────────────────────────────────────

    def segment(self, text: str, *, tau: float = DEFAULT_TAU) -> list[int]:
        """Locate story boundaries in one transcript.

        Args:
            text: The transcript, as whitespace-separated text.
            tau: Decision threshold on `P("1")`. Rarely worth changing: this
                model's confidences are saturated and bimodal, so any value in
                roughly [0.005, 0.5] gives the same answer within noise. Setting
                it to 0 is the one harmful choice. See `DEFAULT_TAU`.

        Returns:
            Word offsets at which a new story begins, ascending.

        Note:
            Three caveats belong with every call, not in a footnote.

            **C5 -- tau was itself fitted.** 0.010 was selected on the same
            117-document validation split as everything else, and applying it
            unchanged to test is the correct protocol but does not make it
            noise-free.

            **C7 -- there is no high-precision regime here.** The geometry
            V4_hard inherits (`V2_w3072`) cannot exceed precision 0.564 at *any*
            threshold. If your use is sensitive to false boundaries, raising tau
            will not deliver it; the fix is a different geometry, not a
            different threshold.

            **C10 -- boundaries only.** No topic, story type, or calibrated
            confidence. `P("1")` is a thresholded logit with no temperature
            fitting and must not be reported as a probability.

            See LIMITATIONS.md for all eleven.
        """
        return self.segment_documents([text], tau=tau)[0]

    def segment_documents(
        self, documents: Sequence[str], *, tau: float = DEFAULT_TAU
    ) -> list[list[int]]:
        """Locate boundaries in many transcripts at once.

        Windows are independent across documents as well as within one, so they
        are pooled and batched together. Substantially faster than looping
        `segment` per document.

        Args:
            documents: Transcripts, as whitespace-separated text.
            tau: Decision threshold on `P("1")`; see `segment`.

        Returns:
            One list of ascending word offsets per document.
        """
        return [
            self.apply_threshold(per_doc, tau=tau)
            for per_doc in self.score_windows(documents)
        ]

    def segment_spans(
        self, text: str, *, tau: float = DEFAULT_TAU
    ) -> list[tuple[int, int]]:
        """Slice a transcript into half-open story spans.

        Args:
            text: The transcript, as whitespace-separated text.
            tau: Decision threshold on `P("1")`; see `segment`.

        Returns:
            `(start, end)` word ranges covering the transcript with no gaps.
        """
        return spans(self.segment(text, tau=tau), len(text.split()))

    def apply_threshold(
        self, scores: Sequence[WindowScore], *, tau: float = DEFAULT_TAU
    ) -> list[int]:
        """Turn one document's window scores into boundaries. Free -- no model.

        Args:
            scores: Window scores for a single document.
            tau: Decision threshold on `P("1")`.

        Returns:
            Deduplicated word offsets, ascending.
        """
        fired = [o for s in scores if s.p_break > tau for o in s.breaks]
        return dedupe(fired, self.decode.dedupe_tolerance_words)

    # ── Generation ────────────────────────────────────────────────────────────

    def _run_pool(self, pool: list[Window]) -> list[WindowScore]:
        """Score a pool of windows, batching across documents.

        Sorted long-to-short so batches are length-homogeneous -- padding waste
        stays small and the worst memory case is hit first, failing fast rather
        than after most of the work is done.

        Args:
            pool: Windows from any number of documents.

        Returns:
            One score per window, in arbitrary order.

        Raises:
            _BatchOOMError: Propagated when a single window will not fit, i.e.
                the batch size is already 1 and there is nothing left to halve.
        """
        if not pool:
            return []
        order = sorted(range(len(pool)), key=lambda i: -len(pool[i].prompt))
        out: list[WindowScore] = []

        prev_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"  # required for batched generation
        try:
            batch_size = self.decode.gen_batch_size
            cursor = 0
            while cursor < len(order):
                idxs = order[cursor : cursor + batch_size]
                try:
                    out.extend(self._score_batch([pool[i] for i in idxs]))
                except _BatchOOMError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        "CUDA OOM; retrying at gen_batch_size=%d", batch_size
                    )
                    continue
                cursor += len(idxs)
        finally:
            self._tokenizer.padding_side = prev_side
        return out

    def _score_batch(self, windows: list[Window]) -> list[WindowScore]:
        """Run both passes over one batch of windows.

        Args:
            windows: Windows to score together.

        Returns:
            Their scores.

        Raises:
            _BatchOOMError: On CUDA out-of-memory, so the caller can halve the batch.
        """
        import torch

        try:
            with torch.no_grad():
                probs = self._decision_probs([w.prompt for w in windows])
                # Force the decision to "1" so every window yields anchors; the
                # threshold decides later which to keep.
                enc = self._tokenizer(
                    [w.prompt + "1" for w in windows],
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)
                generated = self._model.generate(
                    **enc,
                    max_new_tokens=self.decode.max_new_tokens,
                    do_sample=False,  # greedy: the anchor text must be verbatim
                    num_beams=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware
            torch.cuda.empty_cache()
            raise _BatchOOMError from exc

        gen = generated[:, enc["input_ids"].shape[1] :]
        out: list[WindowScore] = []
        for row, window in enumerate(windows):
            text = "1" + self._tokenizer.decode(gen[row], skip_special_tokens=False)
            score = WindowScore(
                doc_index=window.doc_index,
                word_start=window.word_start,
                p_break=probs[row][0],
                p_none=probs[row][1],
            )
            for anchor in parse_anchors(text, self.prompt):
                hit = localize(window.words, anchor, strict=self.decode.strict_anchors)
                score.ambiguous += int(hit.ambiguous)
                if hit.offset is None:
                    score.unlocatable += 1
                    continue
                if in_guard_zone(
                    hit.offset,
                    len(window.words),
                    is_first_window=window.is_first,
                    is_last_window=window.is_last,
                    guard_fraction=self.geometry.guard_fraction,
                ):
                    continue
                score.breaks.append(window.word_start + hit.offset)
            out.append(score)
        return out

    def _decision_probs(self, prompts: list[str]) -> list[tuple[float, float]]:
        """Read `P("1")` and `P(NONE)` at each prompt's decision position.

        Only the final position's logits are needed. Asking for all of them
        materialises a `batch x seq x 128k` tensor -- about 16 GB at batch 16
        and a 4096-token window -- which, held while `generate` allocates its KV
        cache, is what forced the research code into two separately freed
        passes. `logits_to_keep=1` reduces it to a few megabytes.

        Args:
            prompts: Rendered prompts, unmodified.

        Returns:
            One `(p_break, p_none)` pair per prompt.
        """
        import torch

        enc = self._tokenizer(prompts, return_tensors="pt", padding=True).to(
            self._device
        )
        if self._supports_logits_to_keep:
            try:
                logits = self._model(**enc, logits_to_keep=1).logits
            except TypeError:  # pragma: no cover - older transformers
                self._supports_logits_to_keep = False
                logits = self._model(**enc).logits
        else:  # pragma: no cover - older transformers
            logits = self._model(**enc).logits

        probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        yes = probs[:, self._yes_id].tolist()
        no = probs[:, self._no_id].tolist()
        del enc, logits, probs
        torch.cuda.empty_cache()
        return list(zip(yes, no, strict=True))


class _BatchOOMError(Exception):
    """Internal signal that a batch did not fit; the caller halves and retries."""
