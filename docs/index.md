# breakingnews

A broadcast-news transcript arrives as an undifferentiated block of 2,000–17,000 words covering several unrelated stories. This package finds the word offsets where one story ends and the next begins, turns them into one row per story, and puts the rows back together again byte for byte. A boundary means the broadcast moves to a genuinely different story — new topic, new event, different actors — and explicitly not a change of speaker, correspondent, location or sub-angle within a continuing story. Boundaries only: nothing here labels, classifies or summarises the segments it produces.

These pages are the API reference. For installation, the command-line workflow, and the accuracy figures with what they do and do not establish, see the [README](https://github.com/sakshi-bhalla/breakingnews#readme).

## Segmenting

The entry point. `score_windows` is the threshold-free primitive; `apply_threshold` is free to re-run, which is what makes a tau sweep cheap.

```{eval-rst}
.. autoclass:: breakingnews.Segmenter
   :members:
.. autoclass:: breakingnews.WindowScore
   :members:
.. autofunction:: breakingnews.spans
```

## Segments and the merge back

Segmentation is a partition on character spans, so every character lands in exactly one story and `merge_segments` reproduces the source exactly — or refuses, rather than returning a plausible but wrong document.

Re-running can shift a boundary by a few words, because batched bf16 generation is not bit-reproducible. Join two runs with `reconcile`, never on `segment_id`, which renumbers whenever a run finds a different number of stories.

```{eval-rst}
.. autoclass:: breakingnews.Segment
   :members:
.. autofunction:: breakingnews.to_segments
.. autofunction:: breakingnews.merge_segments
.. autofunction:: breakingnews.reconcile
.. autoclass:: breakingnews.Correspondence
   :members:
.. autofunction:: breakingnews.id_map
.. autofunction:: breakingnews.group_by_record
.. autofunction:: breakingnews.drop_flagged
.. autofunction:: breakingnews.make_segment_id
.. autofunction:: breakingnews.parse_segment_id
```

## Scoring

`match` pairs predictions to gold boundaries globally, nearest first, so one prediction cannot satisfy two gold boundaries. Gold defines the denominator.

```{eval-rst}
.. autofunction:: breakingnews.match
.. autofunction:: breakingnews.prf
.. autofunction:: breakingnews.score_documents
.. autofunction:: breakingnews.pk_and_windowdiff
.. autofunction:: breakingnews.baseline_none
.. autofunction:: breakingnews.baseline_uniform
```

## Configuration

Window geometry is trained into the adapter and travels with it, so `Geometry.from_adapter` refuses attempts to override the fields that were trained in — running at a different window size silently costs accuracy.

```{eval-rst}
.. autoclass:: breakingnews.Geometry
   :members:
.. autoclass:: breakingnews.PromptSpec
   :members:
.. autoclass:: breakingnews.DecodeSpec
   :members:
.. autodata:: breakingnews.DEFAULT_TAU
.. autodata:: breakingnews.DEFAULT_BASE_MODEL
```

## Loading and verification

```{eval-rst}
.. autofunction:: breakingnews.resolve_adapter
.. autofunction:: breakingnews.verify_adapter
.. autofunction:: breakingnews.export_break_token_rows
```
