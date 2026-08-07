# Changelog

Release notes are generated from the GitHub Release for each `vX.Y.Z` tag; this
file records anything worth curating by hand.

## Unreleased

Nothing released yet. The first release will provide:

- `Segmenter` for locating story boundaries, with a threshold-free
  `score_windows` primitive so a decision threshold can be swept without
  re-running generation.
- Metrics (`match`, `prf`, `score_documents`, `pk_and_windowdiff`, baselines),
  importable without a GPU.
- Segment construction and an exact merge back, with `reconcile` for mapping
  segment ids across two runs.
- A `breakingnews` CLI: `check-adapter`, `run`, `sweep`, `segments`, `merge`,
  `reconcile`, `score`.
- JSON Schemas for transcripts, annotations, predictions and segments.
