#!/usr/bin/env python3
"""Validate transcript / annotation / prediction JSONL against the schemas.

    python scripts/validate_data.py transcripts.jsonl
    python scripts/validate_data.py transcripts.jsonl annotations.jsonl
    python scripts/validate_data.py transcripts.jsonl --predictions preds.jsonl
    python scripts/validate_data.py transcripts.jsonl --segments segments.jsonl

Structural validation uses the JSON Schemas in `schemas/` when `jsonschema` is
installed, and is skipped with a notice when it is not. The cross-file checks
always run: those compare two documents, so no schema can express them, and
they are the ones that catch the damaging error -- an annotation made against a
different revision of the text still parses and silently shifts every offset.

Exits non-zero if anything fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, reporting the line number of any parse failure.

    Args:
        path: File to read.

    Returns:
        One dict per non-blank line.

    Raises:
        SystemExit: If a line is not valid JSON.
    """
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{i}: invalid JSON -- {exc}")
    return out


def structural(records: list[dict], schema_name: str, label: str) -> list[str]:
    """Validate records against a JSON Schema, if jsonschema is available.

    Args:
        records: Parsed records.
        schema_name: Filename in `schemas/`.
        label: Human-readable name for error messages.

    Returns:
        Problem descriptions, empty when clean.
    """
    try:
        import jsonschema
    except ImportError:
        print(f"  (jsonschema not installed -- skipping structural checks on {label})")
        return []

    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    problems = []
    for i, rec in enumerate(records, 1):
        for err in validator.iter_errors(rec):
            loc = "/".join(str(p) for p in err.path) or "(root)"
            problems.append(f"{label} line {i} [{loc}]: {err.message}")
    return problems


def cross_file(transcripts: list[dict], annotations: list[dict]) -> list[str]:
    """Run the checks that span two files.

    Args:
        transcripts: Parsed transcripts.
        annotations: Parsed annotations, possibly empty.

    Returns:
        Problem descriptions, empty when clean.
    """
    problems: list[str] = []
    by_id: dict[str, dict] = {}
    for t in transcripts:
        rid = t["record_id"]
        if rid in by_id:
            problems.append(f"duplicate transcript record_id {rid!r}")
        by_id[rid] = t

    # The whitespace split is the unit of analysis; word_count must agree with it.
    for t in transcripts:
        actual = len(t["body"].split())
        if t["word_count"] != actual:
            problems.append(
                f"transcript {t['record_id']}: word_count {t['word_count']} != "
                f"len(body.split()) {actual}"
            )

    seen: set[str] = set()
    for a in annotations:
        rid = a["record_id"]
        if rid in seen:
            problems.append(f"duplicate annotation record_id {rid!r}")
        seen.add(rid)

        t = by_id.get(rid)
        if t is None:
            problems.append(f"annotation {rid}: no matching transcript")
            continue

        # The check that saves you: an annotation made against a different
        # revision of the text parses fine and shifts every offset silently.
        if a["word_count"] != t["word_count"]:
            problems.append(
                f"annotation {rid}: word_count {a['word_count']} != transcript's "
                f"{t['word_count']} -- annotated against different text"
            )
            continue

        n = t["word_count"]
        bad = [b for b in a["breaks"] if not 0 < b < n]
        if bad:
            problems.append(f"annotation {rid}: offsets outside (0, {n}): {bad}")
        if a["breaks"] != sorted(a["breaks"]):
            problems.append(f"annotation {rid}: breaks are not ascending")
    return problems


def main() -> int:
    """Entry point.

    Returns:
        0 when everything validates, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("transcripts", type=Path)
    ap.add_argument("annotations", type=Path, nargs="?")
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--segments", type=Path)
    args = ap.parse_args()

    problems: list[str] = []

    transcripts = load_jsonl(args.transcripts)
    print(f"{len(transcripts)} transcripts from {args.transcripts}")
    problems += structural(transcripts, "transcript.schema.json", "transcript")

    annotations: list[dict] = []
    if args.annotations:
        annotations = load_jsonl(args.annotations)
        n_breaks = sum(len(a["breaks"]) for a in annotations)
        n_clean = sum(1 for a in annotations if not a["breaks"])
        print(
            f"{len(annotations)} annotations from {args.annotations} "
            f"({n_breaks} breaks, {n_clean} with none)"
        )
        problems += structural(annotations, "annotation.schema.json", "annotation")

    problems += cross_file(transcripts, annotations)

    if args.predictions:
        preds = load_jsonl(args.predictions)
        print(f"{len(preds)} predictions from {args.predictions}")
        problems += structural(preds, "prediction.schema.json", "prediction")

    if args.segments:
        segs = load_jsonl(args.segments)
        n_rec = len({s["record_id"] for s in segs})
        print(f"{len(segs)} segments from {args.segments} ({n_rec} records)")
        problems += structural(segs, "segment.schema.json", "segment")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:50]:
            print(f"  - {p}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        return 1
    print("\nOK -- all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
