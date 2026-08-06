# Changelog

Release notes are generated from the GitHub Release for each `vX.Y.Z` tag; this
file records anything worth curating by hand.

## Unreleased

First packaging of the `V4_hard` adapter as a library. Nothing is released yet.

- `Segmenter` with a threshold-free `score_windows` primitive, so a decision
  threshold can be swept without re-running generation.
- Metrics (`match`, `prf`, `score_documents`, `pk_and_windowdiff`, baselines),
  importable without torch. Reproduce the published tables exactly.
- Segment/merge round trip with persistent `record_id` provenance; the merge is
  byte-exact and refuses input that cannot be a document.
- `breakingnews` CLI: `check-adapter`, `run`, `sweep`, `segments`, `merge`,
  `score`.
- JSON Schemas for transcripts, annotations, predictions and segments.
- `LIMITATIONS.md` carrying all eleven project caveats verbatim.
