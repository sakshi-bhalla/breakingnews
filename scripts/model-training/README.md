# Training scripts

**These are not part of the installed package.** `pip install breakingnews`
gives you inference and scoring only; this directory ships in the repository so
the training procedure is inspectable, citable and runnable, not so that it is
importable.

## State: research copies, paths parameterised

Everything here is copied from the research tree that produced `V4_hard`, with
one change: `config.py` no longer hardcodes the cluster it ran on. Nothing else
has been refactored, and in particular these scripts do **not** use the
package's configuration objects. That is deliberate — the published numbers came
from exactly this code, and rewriting it risks changing what it does. It is
preserved rather than improved.

## Configuration

No file needs editing. Every path is an environment variable, and every default
reproduces the layout the published run used:

| variable | default | what |
|---|---|---|
| `BREAKINGNEWS_DATA_DIR` | `./data` | transcripts and annotations |
| `BREAKINGNEWS_TRANSCRIPTS` | `$DATA_DIR/sample_transcripts.jsonl` | transcripts JSONL |
| `BREAKINGNEWS_ANNOTATIONS` | `$DATA_DIR/all_annotations.jsonl` | annotations JSONL |
| `BREAKINGNEWS_BUILD_DIR` | `./build` | generated dataset + manifests |
| `BREAKINGNEWS_RUNS_DIR` | `./runs` | adapters and logs |
| `BREAKINGNEWS_PREPARED_BASE` | `./base_prepared` | the ~15 GB prepared base |
| `HF_HOME` | `~/.cache/huggingface` | model cache |

On a cluster, point `HF_HOME` and `BREAKINGNEWS_PREPARED_BASE` at shared storage
so the 16 GB download and the 15 GB prepared base happen once and are visible
from both login and compute nodes.

## The pipeline

| stage | script | what it does |
|---|---|---|
| 0.5 | `prepare_base.py` | registers `<\|STORY_BREAK\|>`, resizes the embedding matrix, rescales the new rows to the mean row norm. Run once per machine. |
| 1 | `build_dataset.py` | windows the annotated transcripts, routes each window through the edge guard, emits anchor-format training rows |
| 1b | `build_variants.py` | builds the same dataset at several window geometries |
| 2 | `train.py` | the LoRA fine-tune |
| 3 | `infer.py` | sliding-window inference |
| 3b | `threshold_sweep.py` | generates once per window with the decision forced, then sweeps τ for free |
| 4 | `evaluate.py` / `rescore.py` | ±tolerance P/R/F1, Pk, WindowDiff |
| 5 | `mine_errors.py` | hard-example mining, which is what makes `V4_hard` "hard" |
| — | `make_review.py` | builds the human adjudication HTML (see C1) |

## Data

Not included, and not obtainable from us — the transcripts are licensed. To
train on your own corpus you need two JSONL files matching
[`../../schemas/`](../../schemas/), then:

```bash
python ../validate_data.py transcripts.jsonl annotations.jsonl

export BREAKINGNEWS_TRANSCRIPTS=$PWD/transcripts.jsonl
export BREAKINGNEWS_ANNOTATIONS=$PWD/annotations.jsonl

python prepare_base.py      # once per machine
python build_dataset.py
python train.py
```

## Two things that will waste your day

**`trainable_token_indices` must cover both matrices.** Llama-3.1 has
`tie_word_embeddings=False`, so `embed_tokens` and `lm_head` are separate.
Training only the `embed_tokens` row leaves the model unable to *emit* the new
token — a run that scores F1 exactly 0 while `eval_loss` falls normally. The fix
is already in `train.py`; do not undo it.

**Window geometry must be recorded next to the adapter.** `train.py` writes
`segmentation_config.json` into the adapter directory. Without it the library
refuses to load, by design: an adapter run at the wrong window size fails
silently and looks like a bad model.

## Hardware

The published run used bf16 on an 80 GB card at window 3072. A 40 GB card works.
Free Colab and Kaggle tiers (16 GB) do **not** fit this model for training or for
bf16 inference.

## Reproducibility

Every published number is single-seed (seed 42, caveat C3), and differences
below 0.03 F1 are not distinguishable (C4). If you retrain and get a different
number, that is expected before it is meaningful. See
[`../../LIMITATIONS.md`](../../LIMITATIONS.md).
