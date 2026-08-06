# breakingnews

Story segmentation for broadcast-news transcripts. A LoRA adapter on
Llama-3.1-8B-Instruct that takes a raw transcript and returns the word offsets
where one story ends and the next begins.

A transcript arrives as an undifferentiated block of 2,000–17,000 words covering
several unrelated stories. The task is to mark where the broadcast moves to a
**genuinely different story** — new topic, new event, different actors — and not
at changes of speaker, correspondent, location or sub-angle within a continuing
story. That distinction is the entire difficulty.

Boundaries only. Nothing here labels, classifies or summarises the resulting
segments.

## Install

```bash
pip install breakingnews          # scoring + post-processing, pure Python
pip install "breakingnews[gpu]"   # + torch/transformers/peft for inference
```

The base install has no heavy dependencies and gives you the metrics
(`score_documents`, `match`, `prf`, `pk_and_windowdiff`, the two baselines),
segment/span construction, and `breakingnews score` — enough to evaluate a
prediction file on a laptop.

Inference needs the `[gpu]` extra and a GPU with ≥24 GB. The adapter is fetched
from the Hugging Face Hub on first use and cached.

## Use

```python
from breakingnews import Segmenter

seg = Segmenter.from_pretrained("OWNER/breakingnews-v4-hard")
breaks = seg.segment(transcript)  # -> [909, 1333, 2351, ...]
stories = seg.segment_spans(transcript)  # -> [(0, 909), (909, 1333), ...]
```

From the command line:

```bash
breakingnews check-adapter OWNER/breakingnews-v4-hard
breakingnews run   OWNER/breakingnews-v4-hard --input transcripts.jsonl --out preds.jsonl
breakingnews sweep OWNER/breakingnews-v4-hard --input transcripts.jsonl --gold gold.jsonl
breakingnews score --predictions preds.jsonl --gold gold.jsonl   # no GPU
```

## From boundaries to rows, and back

Boundaries are offsets; analysis usually wants one row per story. The expansion
carries provenance and is reversible:

```bash
breakingnews segments --transcripts t.jsonl --predictions p.jsonl --out s.jsonl
breakingnews merge    --segments s.jsonl --out rebuilt.jsonl
```

```python
from breakingnews import to_segments, merge_segments

segs = to_segments(record_id, body, breaks, min_words=100)
segs[1].segment_id  # '12791924207105a2#001'
segs[1].record_id  # '12791924207105a2'  -- provenance, always a field
segs[1].clean_text  # the story text

merge_segments(segs) == (record_id, body)  # byte-for-byte, whitespace included
```

`segment_id` is an ordinal rather than an offset or a content hash, because
offsets are not bit-reproducible across batch sizes (L1b) — an ordinal survives
a re-run that finds the same stories, though not one that finds a different
*number* of them. `--min-words` flags short segments; it never drops them.

Offsets are indices into `transcript.split()`. See
[`schemas/`](schemas/) for the JSONL formats.

## Results — `V4_hard`

Trained on 998 annotated transcripts (2,829 boundaries). τ = 0.010, selected on
validation and applied unchanged to the held-out test set.

| split | docs | boundaries | tolerance | precision | recall | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| **validation** | 117 | 322 | ±25 w | 0.573 | 0.621 | **0.596** |
| validation | 117 | 322 | ±100 w | 0.728 | 0.789 | 0.757 |
| test | 20 | 64 | ±25 w | 0.593 | 0.797 | 0.680 |
| test | 20 | 64 | ±100 w | 0.663 | 0.891 | 0.760 |

> **Quote the validation row.** It rests on 322 gold breaks against the test
> split's 64.
>
> **C2 — the test split is too small to carry a headline.** At n=64 the
> binomial standard error on recall alone is ≈0.05, putting test F1 0.680 at
> roughly **±0.07** before any other source of variation. The val→test rise is
> not evidence of generalisation; test recall is 0.797 against val's 0.621,
> which says the test transcripts have easier breaks.
>
> **C1 — precision is a lower bound, not an estimate.** A large share of scored
> false positives are *real* topic changes the annotator chose to group into one
> thematic block. 216 such cases are still awaiting adjudication. True precision
> is somewhere above 0.573 and nobody yet knows where. Never compute a derived
> statistic that treats a false positive as clean error.

Baselines on the same test set: predicting no boundaries scores F1 0.000;
predicting N boundaries at uniform spacing scores 0.062.

**On the two tolerances.** A prediction counts as correct if it lands within N
words of a true boundary. Widening N does not change the model, it changes the
question: ±25 asks "to within a sentence?", ±100 asks "did it find the seam at
all?". Mean offset on matched boundaries is 15.1 words even when scored inside a
100-word window — the extra matches are close, not sloppy. **±25 is the
default.**

## τ is not a tuning knob

The confidences are saturated and bimodal: 49% of validation windows sit above
0.5, 38% below 0.001, p25 = 0.000 and p75 = 0.996. The entire 500× sweep of τ
moves 12% of windows; F1 across that whole range spans 0.589–0.602. For this
window geometry, greedy and thresholded F1 are **identical** at 0.6250.

τ exists to exclude τ = 0, where every window fires. It is not a precision/recall
dial, and per **C7** this geometry has no high-precision regime at all —
`V2_w3072` cannot exceed precision 0.564 at any threshold. If your use is
sensitive to false boundaries, the fix is a different geometry, not a different
threshold.

*(The "+0.28 F1 from thresholding" figure that circulates for this project —
greedy 0.3291 → 0.6131 — belongs to `V2_w2048`, a different geometry. It does
not describe this model.)*

## Read this before citing a number

**[LIMITATIONS.md](LIMITATIONS.md)** — all eleven caveats. Every result here is
single-seed (C3), differences below 0.03 F1 are not distinguishable (C4), and
the domain is US broadcast news labelled by one annotator with no
inter-annotator agreement (C8, C9).

## Training data

Not included. The transcripts are licensed source material.

| | |
|---|---|
| transcripts | 1,000 sampled US TV news broadcasts, **998 annotated** |
| period | September 1992 – November 2020 |
| outlets | CNN 585 · FOX 200 · MSNBC 117 · ABC 76 · CBS 20 |
| labels | **2,829 story boundaries**, as word offsets |
| density | median 2 per transcript, max 18; **306 transcripts have none** |

Annotations are word offsets carrying no text, and are available for review on
request. To train on your own corpus, see [`schemas/`](schemas/) and
[`scripts/model-training/`](scripts/model-training/).

## License

MIT.
