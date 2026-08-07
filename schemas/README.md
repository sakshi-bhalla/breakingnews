# Data schemas

Four JSONL formats. Two are inputs you supply; two are produced for you.

| file | schema | who writes it |
|---|---|---|
| transcripts | [`transcript.schema.json`](transcript.schema.json) | you |
| annotations | [`annotation.schema.json`](annotation.schema.json) | you, to train or evaluate |
| predictions | [`prediction.schema.json`](prediction.schema.json) | `breakingnews run` |
| segments | [`segment.schema.json`](segment.schema.json) | `breakingnews segments` |

Predictions are boundaries. Segments are the *rows* most analysis actually
wants — one per story, each carrying its parent `record_id` — and the expansion
is reversible:

```bash
breakingnews segments --transcripts t.jsonl --predictions p.jsonl --out s.jsonl
breakingnews merge    --segments s.jsonl --out rebuilt.jsonl   # byte-identical to t.jsonl
```

Annotations are **offsets only, never text**. That is deliberate: it means an
annotation set can be published even when the transcripts it describes are
licensed and cannot be.

## The one invariant that matters

Every word offset in this project is an index into `body.split()`. Not into
characters, not into tokens, not into a normalised or re-tokenised version of
the text — into the whitespace split of the exact `body` string.

So `word_count` is checked three ways at load, and each mismatch is a hard
error rather than a warning:

1. a transcript's `word_count` must equal `len(body.split())`
2. an annotation's `word_count` must equal its transcript's
3. every break must satisfy `0 < b < word_count`

The second is the one that saves you. An annotation made against a slightly
different revision of the text — one extra header line, one collapsed double
space — still parses, still looks reasonable, and silently shifts every offset
in the document. There is no way to detect that downstream.

Offset `0` is not a boundary: every document begins a story. Neither is
`word_count`. Boundaries are strictly interior, mark the **first word of the
new story**, and must be ascending and unique.

An empty `breaks` array is meaningful, not missing data. 306 of the 998
training transcripts contain no boundary at all, and a model that never
predicts one scores F1 0.000 against them — those documents are what stop
recall being gamed.

## Minimal examples

Transcript:

```json
{"record_id": "0f1e2d3c4b5a6970", "outlet": "CNN", "date": "2012-09-21",
 "show": "EXAMPLE NIGHTLY", "type": "dump", "word_count": 6528,
 "body": "ANCHOR: Good evening. Our top story tonight ..."}
```

Annotation:

```json
{"record_id": "a1b2c3d4e5f60718", "outlet": "CNN", "date": "2014-10-11",
 "show": "CNN MONEY", "word_count": 3607, "category": "A",
 "breaks": [909, 1333, 2351, 2537, 2915, 3184],
 "agent_notes": "Six distinct story segments; see notes."}
```

A clean transcript:

```json
{"record_id": "9a8b7c6d5e4f3021", "word_count": 2414, "category": "C",
 "breaks": []}
```

## Fields the model ignores

`outlet`, `show`, `date`, `year`, `month`, `agent_notes` and `category` are
never read by training or inference. They exist for stratification, auditing
and adjudication. `category` is redundant with `breaks` by construction (`A`
iff `breaks` is non-empty) and is retained only because the dataset builder
routes the two categories differently.

## Segment identity, and why it is an ordinal

`segment_id` is `{record_id}#{index:03d}`. It encodes no offset on purpose:
predicted offsets are not bit-reproducible across batch sizes, so an id
derived from an offset — or a hash of the segment text — would churn on a
re-run that found exactly the same stories. An ordinal survives that.

What an ordinal does *not* survive is a re-segmentation that finds a different
**number** of stories: everything after the change renumbers. **Do not join two
runs on `segment_id`** — one extra boundary and every later row silently
mismatches.

Use `reconcile` instead, which pairs segments by how much text they actually
share, so it survives both the renumbering and the offset jitter:

```bash
breakingnews reconcile --old run_a.jsonl --new run_b.jsonl --out map.jsonl
```

```python
from breakingnews import reconcile, id_map

pairs = reconcile(old_segments, new_segments)
id_map(pairs)  # {'rec#001': 'rec#001', ...} -- one-to-one only
```

Each pairing gets a status: `same` (identical span), `moved` (same story, the
boundary shifted), `split` / `merged` (the correspondence is not one-to-one),
`added` / `removed` (no counterpart). Nothing is dropped from either side.

`id_map` returns only `same` and `moved`. A split or merged segment has no
single successor, and silently choosing one would quietly corrupt any analysis
carried across the two runs — those cases are yours to look at.

`record_id` is also carried as its own field on every row, so rolling a
segment-level result back up to the broadcast never requires parsing a string.

## The merge is exact, and refuses when it cannot be

Segments are cut on **character** spans that are contiguous and gapless, so
concatenating a record's segments in order reproduces the source byte-for-byte
— including newlines and runs of whitespace that `body.split()` would have
destroyed. Verified on the 20-document test split: 106 segments merged back to
20 records, zero differences.

`merge` refuses rather than guesses when the input cannot be a document: a
missing or duplicated index, segments from more than one `record_id`, a gap in
the character cover, or segments built with `--no-text`. Each of those would
otherwise produce a plausible-looking but wrong document.

That last point has a consequence worth stating: **dropping segments breaks the
merge, deliberately.** `--min-words` *flags* short segments and never removes
them, and `drop_flagged` returns both halves with a count. Once you discard
rows, the remainder is no longer a document, and `merge` will say so.

For a licensed corpus, `--no-text` emits offsets only — publishable, and a
consumer holding the source can slice it themselves.

## Validating

JSON Schema covers structure and the `category`/`breaks` consistency rule. It
cannot express the cross-file `word_count` checks, because those compare two
documents. Run those with:

```bash
python scripts/validate_data.py transcripts.jsonl annotations.jsonl
```

## If you are annotating your own corpus

The model's measured domain is US broadcast-news transcripts from five
outlets, labelled by a single annotator with no inter-annotator agreement
with no inter-annotator agreement. Behaviour on print, on other outlets, on other languages
or on non-news speech is unmeasured — not "probably fine", unmeasured. If your
domain differs, these schemas are what you need to build a training set of
your own; see `scripts/model-training/`.

Two things worth deciding before you start, because they are the ones that
turned out to matter:

- **Granularity.** A "boundary" here means a move to a genuinely different
  story — new topic, new event, different actors — and explicitly *not* a
  change of speaker, correspondent, location, or sub-angle within a continuing
  story. Most scored false positives in the published evaluation are real topic
  changes the annotator chose to group into one thematic block, so precision is a
  lower bound rather than an estimate. Write your rule down
  before you label, not after.
- **Placement.** Offsets mark the first word of the new story. Where the
  tease or handoff belongs — to the old story or the new — is a judgement call
  that the ±25-word match tolerance exists to absorb.
