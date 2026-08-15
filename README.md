<div>
# breakingnews
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/breakingnews?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/breakingnew) 
</div>
**Purpose.** A broadcast-news transcript arrives as an undifferentiated block of 2,000–17,000 words covering several unrelated stories. This package finds the word offsets where one story ends and the next begins, turns them into one row per story, and lets you put the rows back together again. It exists because content analysis needs a comparable unit: a television transcript is not one article, and treating it as one — or splitting it on speaker turns — gives the wrong denominator. A "boundary" means the broadcast moves to a **genuinely different story**: new topic, new event, different actors, and explicitly *not* a change of speaker, correspondent, location or sub-angle within a continuing story. Boundaries only — nothing here labels, classifies or summarises the segments it produces.

## Install

```bash
pip install breakingnews          # scoring, segments, merge, reconcile
pip install "breakingnews[gpu]"   # + torch/transformers/peft, for inference
```

The base install is pure Python. Inference needs the `[gpu]` extra and a GPU with at least 24 GB; the adapter is fetched from the Hugging Face Hub on first use and cached, and the Llama-3.1-8B base is a further ~16 GB download. Built with Llama.

## The workflow

```bash
# 1. Confirm the adapter is intact before trusting anything it produces.
breakingnews check-adapter sakshib3/Llama-3.1-breakingnews --revision v1

# 2. Transcripts -> boundaries.
breakingnews run sakshib3/Llama-3.1-breakingnews --revision v1 \
    --input transcripts.jsonl --out predictions.jsonl

# 3. Boundaries -> one row per story.
breakingnews segments --transcripts transcripts.jsonl \
    --predictions predictions.jsonl --out segments.jsonl --min-words 100

# 4. Rows -> whole documents again, byte for byte.
breakingnews merge --segments segments.jsonl --out rebuilt.jsonl
```

Two more commands help once you have output:

1. **`score`** compares a prediction file against your own gold annotations and reports precision, recall, F1 at three tolerances, plus Pk and WindowDiff. Every gold document counts, so a prediction file that is missing records is reported rather than quietly scoring higher.
2. **`reconcile`** maps segment ids from one run onto another by how much text they share, and tells you which ids carried over unchanged, which moved, and which were split or merged.

```bash
breakingnews score --predictions predictions.jsonl --gold annotations.jsonl
breakingnews reconcile --old run_a.jsonl --new run_b.jsonl
```

Everything except `run` and `sweep` works without a GPU.

### In Python

```python
from breakingnews import Segmenter, to_segments, merge_segments

seg = Segmenter.from_pretrained("sakshib3/Llama-3.1-breakingnews", revision="v1")
breaks = seg.segment(transcript)  # [909, 1333, 2351, ...]
rows = to_segments(record_id, transcript, breaks, min_words=100)
merge_segments(rows) == (record_id, transcript)  # byte for byte
```

Pin `revision`. Unpinned resolves to `main`, which moves.

## What you get

One row per story. `--minimal` emits just the first four fields; offsets index `transcript.split()`, and [`schemas/`](schemas/) documents all four JSONL formats.

| field | |
|---|---|
| `record_id` | the parent broadcast, on every row |
| `segment_id` | `{record_id}#{index:03d}` |
| `text` | the story |
| `n_cuts` | boundaries found in the parent record; `0` means it was never cut |
| `word_start` `word_end` `char_start` `char_end` `n_words` | offsets |

Three properties the package holds to:

- **Segmentation is a partition.** Every character lands in exactly one story, so `merge` reproduces the source byte for byte and refuses rather than guesses when it cannot.
- **Nothing is dropped silently.** `--min-words` *flags* short segments; a record that fails is named and the command exits non-zero.
- **Provenance survives.** `record_id` is a field on every row, never something you parse out of an id.

Re-running can shift a boundary by a few words, because batched bf16 generation is not bit-reproducible, so **join two runs with `reconcile`, never on `segment_id`** — that renumbers whenever a run finds a different number of stories.

## Accuracy

**τ (tau) is the decision threshold.** For every window the model emits a probability that a story boundary is present in it; τ is the cut-off above which that window's boundaries are kept. τ = 0.010 here, selected on validation and applied unchanged to the held-out test set.

| split | docs | boundaries | tolerance | precision | recall | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| **validation** | 117 | 322 | ±25 w | 0.573 | 0.621 | 0.596 |
| validation | 117 | 322 | ±100 w | 0.728 | 0.789 | 0.757 |
| test | 20 | 64 | ±25 w | 0.593 | 0.797 | 0.680 |
| test | 20 | 64 | ±100 w | 0.663 | 0.891 | 0.760 |

> Quote the validation row: it rests on 322 boundaries against the test split's 64, where the standard error on recall alone is ≈0.05. **Precision is a lower bound, not an estimate** — many scored false positives are real topic changes grouped into one thematic block, so do not compute a derived statistic that treats a false positive as clean error.

A prediction counts as correct if it lands within N words of a true boundary: ±25 asks "to within a sentence?", ±100 asks "did it find the seam at all?". Baselines on the same test set are 0.000 for predicting nothing and 0.062 for predicting N boundaries at uniform spacing.

**τ is not a tuning knob.** The confidences are saturated and bimodal — 49% of validation windows above 0.5, 38% below 0.001 — so any value in roughly [0.005, 0.5] gives the same answer, and it exists only to exclude τ = 0, where every window fires. This geometry has no high-precision regime: it cannot exceed precision 0.564 at any threshold, so if your use is sensitive to false boundaries the fix is a different geometry, not a different threshold.

## Training

Trained on 998 annotated transcripts containing 2,829 boundaries, sampled from US TV news broadcasts 1992–2020 (CNN, FOX, MSNBC, ABC, CBS). The transcripts are licensed and are not distributed; the annotations are word offsets carrying no text, available for review on request. To train on your own corpus, see [`schemas/`](schemas/) for the input formats and [`scripts/model-training/`](scripts/model-training/) for the procedure.

## License

The package is MIT. The model is a LoRA adapter on Llama-3.1-8B-Instruct and is governed by the Llama 3.1 Community License, whose terms pass through to anyone using the weights. Built with Llama.
