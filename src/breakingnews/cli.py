"""Command line interface.

    breakingnews check-adapter ADAPTER
    breakingnews run ADAPTER --input transcripts.jsonl --out preds.jsonl
    breakingnews sweep ADAPTER --input transcripts.jsonl --gold annotations.jsonl
    breakingnews score --predictions preds.jsonl --gold annotations.jsonl
    breakingnews segments --transcripts t.jsonl --predictions p.jsonl --out s.jsonl
    breakingnews merge --segments s.jsonl --out rebuilt.jsonl
    breakingnews reconcile --old old.jsonl --new new.jsonl

Only `run` and `sweep` need the `[gpu]` extra and a GPU with at least 24 GB.
Everything else is pure Python -- including `segments` and `merge`, which are
inverses of each other and reproduce the source text byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .config import DEFAULT_TAU, DecodeSpec
from .loading import resolve_adapter, verify_adapter
from .metrics import DEFAULT_TOLERANCE_WORDS, pk_and_windowdiff, score_documents
from .segments import (
    STABLE_STATUSES,
    drop_flagged,
    group_by_record,
    id_map,
    merge_segments,
    reconcile,
    to_segments,
)

TOLERANCES = (25, 50, 100)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file.

    Args:
        path: File to read.

    Returns:
        One dict per non-blank line.
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_segmenter(args: argparse.Namespace):
    """Build a Segmenter, reporting a useful error if torch is absent.

    Args:
        args: Parsed arguments carrying `adapter`, `base_model`, `batch_size`.

    Returns:
        A ready `Segmenter`.
    """
    # segmenter.py imports torch inside function bodies, so importing it here
    # always succeeds and the extras guard never fired -- torch's absence
    # surfaced later as a raw traceback. Probe the real dependency instead.
    import importlib.util

    absent = [
        m
        for m in ("torch", "transformers", "peft", "safetensors")
        if importlib.util.find_spec(m) is None
    ]
    if absent:  # pragma: no cover - depends on install extras
        sys.exit(
            f"`{args.command}` needs the inference extra; missing: "
            f"{', '.join(absent)}.\n  pip install 'breakingnews[gpu]'"
        )
    from .segmenter import Segmenter

    return Segmenter.from_pretrained(
        args.adapter,
        revision=args.revision,
        base_model=args.base_model,
        decode=DecodeSpec(gen_batch_size=args.batch_size),
    )


def cmd_check_adapter(args: argparse.Namespace) -> int:
    """Report the silent-failure modes of an adapter directory.

    Args:
        args: Parsed arguments carrying `adapter`.

    Returns:
        0 when the adapter is sound, 1 otherwise.
    """
    adapter_dir = resolve_adapter(args.adapter, revision=args.revision)
    problems = verify_adapter(adapter_dir)
    print(f"adapter: {adapter_dir}")
    try:
        from .config import Geometry, PromptSpec

        g = Geometry.from_adapter(adapter_dir)
        pspec = PromptSpec.from_adapter(adapter_dir)
        print(
            f"  geometry: window {g.window_tokens} / stride {g.stride_tokens} / "
            f"guard {g.edge_guard_lo}\n"
            f"  anchors : {pspec.anchor_pre_words} pre / "
            f"{pspec.anchor_post_words} post"
        )
    except FileNotFoundError:
        pass
    if not problems:
        print("OK -- no problems found")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Segment a corpus and write predictions JSONL.

    Args:
        args: Parsed arguments.

    Returns:
        0.
    """
    docs = _read_jsonl(args.input)
    if args.limit:
        docs = docs[: args.limit]
    seg = _load_segmenter(args)

    written = 0
    lost: list[str] = []
    with args.out.open("w", encoding="utf-8") as f:
        # Chunked so a large corpus does not build one giant window pool.
        for start in range(0, len(docs), args.chunk):
            chunk = docs[start : start + args.chunk]
            # A chunk is generated as one batch, so a GPU failure takes the whole
            # chunk. Losing it must not also end the run: the remaining chunks
            # are independent, and the reconciliation below reports exactly which
            # records are absent rather than leaving a short file that parses.
            try:
                scored = seg.score_windows([d["body"] for d in chunk])
            except Exception as exc:
                lost.extend(d["record_id"] for d in chunk)
                print(
                    f"  chunk {start}-{start + len(chunk)} FAILED ({exc}); "
                    f"{len(chunk)} record(s) lost",
                    flush=True,
                )
                continue
            for d, per_doc in zip(chunk, scored, strict=True):
                f.write(
                    json.dumps(
                        {
                            "record_id": d["record_id"],
                            "outlet": d.get("outlet"),
                            "date": d.get("date"),
                            "word_count": len(d["body"].split()),
                            "pred_breaks": seg.apply_threshold(per_doc, tau=args.tau),
                            "tau": args.tau,
                            "adapter": str(args.adapter),
                            "adapter_revision": args.revision,
                            "n_unlocatable_anchors": sum(
                                s.unlocatable for s in per_doc
                            ),
                            "n_ambiguous_anchors": sum(s.ambiguous for s in per_doc),
                        }
                    )
                    + "\n"
                )
                written += 1
            print(f"  {written}/{len(docs)} documents", flush=True)

    # Records in must equal records out. Stated, checked, and non-zero on
    # failure -- a short output file that parses is the failure that looks
    # like success.
    print(f"\n{len(docs)} records in, {written} out  ({args.out})")
    if written != len(docs):
        print(f"{len(lost)} record(s) ABSENT from the output:")
        for rid in lost[:10]:
            print(f"  - {rid}")
        if len(lost) > 10:
            print(f"  ... and {len(lost) - 10} more")
        return 1
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Score every threshold from a single generation pass.

    Because the decision token is read separately from the anchors, an
    arbitrary number of thresholds costs one pass rather than one pass each.

    Args:
        args: Parsed arguments.

    Returns:
        0.
    """
    docs = _read_jsonl(args.input)
    if args.limit:
        docs = docs[: args.limit]
    gold_by_id = {a["record_id"]: sorted(a["breaks"]) for a in _read_jsonl(args.gold)}
    docs = [d for d in docs if d["record_id"] in gold_by_id]
    if not docs:
        sys.exit(
            f"no record_id in --input appears in --gold.\n"
            f"  --input has {len(_read_jsonl(args.input))} record(s), "
            f"e.g. {[d['record_id'] for d in _read_jsonl(args.input)[:3]]}\n"
            f"  --gold has {len(gold_by_id)} record(s), "
            f"e.g. {sorted(gold_by_id)[:3]}\n"
            "  The two files must use the same record_id values."
        )

    seg = _load_segmenter(args)
    scored = seg.score_windows([d["body"] for d in docs])
    gold = [gold_by_id[d["record_id"]] for d in docs]

    n_win = sum(len(s) for s in scored)
    greedy = sum(s.greedy_would_fire for per in scored for s in per)
    print(f"\n{len(docs)} documents, {n_win} windows")
    print(f"greedy (argmax) would fire on {greedy}/{n_win} windows\n")
    print(
        f"{'tau':>8}{'firing':>8}{'preds':>7}{'tp':>5}{'fn':>5}{'fp':>5}"
        f"{'P':>7}{'R':>7}{'F1':>8}"
    )
    print("-" * 60)

    rows = []
    for tau in args.thresholds:
        pred = [seg.apply_threshold(per, tau=tau) for per in scored]
        s = score_documents(gold, pred, args.tolerance)
        firing = sum(w.p_break > tau for per in scored for w in per)
        rows.append((tau, s))
        print(
            f"{tau:>8.3f}{firing:>8}{sum(len(p) for p in pred):>7}"
            f"{s.tp:>5}{s.fn:>5}{s.fp:>5}"
            f"{s.precision:>7.3f}{s.recall:>7.3f}{s.f1:>8.4f}"
        )

    best = max(rows, key=lambda r: r[1].f1)
    print(f"\nbest F1 {best[1].f1:.4f} at tau {best[0]:.3f}")
    print(
        "note: on a saturated, bimodal model this curve is expected to be flat. "
        "A wide plateau is not a tuning opportunity."
    )
    if args.out:
        args.out.write_text(
            json.dumps([{"tau": t, **s._asdict()} for t, s in rows], indent=1),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a prediction file against gold. No model, no GPU.

    Args:
        args: Parsed arguments.

    Returns:
        0 normally; 2 when gold documents are absent from the prediction file
        and --allow-missing was not given.
    """
    preds = _read_jsonl(args.predictions)
    gold_rows = _read_jsonl(args.gold)
    gold_by_id = {a["record_id"]: sorted(a["breaks"]) for a in gold_rows}
    pred_by_id = {p["record_id"]: p for p in preds}

    # THE GOLD SET DEFINES THE DENOMINATOR, NOT THE PREDICTION FILE.
    #
    # Iterating predictions and keeping those with gold silently drops every
    # gold document that has no prediction row -- so its misses leave the
    # denominator and the score goes UP. Measured: dropping half a corpus moved
    # F1 from 0.667 to 1.000, exit 0, no warning. The asymmetry made it worse,
    # because the opposite direction (predictions without gold) did print a
    # note, so silence here read as "checked and clean".
    #
    # An absent document is scored as zero predictions: all of its gold counts
    # as misses, which is what it actually is.
    expected = [a["record_id"] for a in gold_rows]
    absent = [rid for rid in expected if rid not in pred_by_id]
    orphans = [p["record_id"] for p in preds if p["record_id"] not in gold_by_id]

    if not set(expected) & set(pred_by_id):
        sys.exit(
            f"no record_id in --predictions appears in --gold.\n"
            f"  --predictions has {len(preds)} record(s), "
            f"e.g. {[p['record_id'] for p in preds[:3]]}\n"
            f"  --gold has {len(gold_by_id)} record(s), e.g. {expected[:3]}\n"
            "  The two files must use the same record_id values."
        )
    if orphans:
        print(f"note: {len(orphans)} prediction(s) have no gold; skipped")
    if absent:
        at_stake = sum(len(gold_by_id[rid]) for rid in absent)
        print(
            f"\n[ERROR] {len(absent)} of {len(expected)} gold documents are ABSENT "
            f"from the\n        prediction file, holding {at_stake} gold breaks. "
            "They are scored as zero\n        predictions (all misses)."
        )
        for rid in absent[:10]:
            print(f"          - {rid}")
        if len(absent) > 10:
            print(f"          ... and {len(absent) - 10} more")
        if not args.allow_missing:
            print("\n        Pass --allow-missing to score a partial file anyway.")
            return 2

    # Gold and predictions must describe the same text.
    gold_wc = {a["record_id"]: a["word_count"] for a in gold_rows if "word_count" in a}
    drift = [
        f"{rid}: prediction {pred_by_id[rid]['word_count']} words, gold {gold_wc[rid]}"
        for rid in expected
        if rid in pred_by_id
        and rid in gold_wc
        and "word_count" in pred_by_id[rid]
        and pred_by_id[rid]["word_count"] != gold_wc[rid]
    ]
    if drift:
        sys.exit(
            f"{len(drift)} record(s) scored against different text:\n  "
            + "\n  ".join(drift[:10])
            + "\nEvery offset would be shifted; the score would be meaningless."
        )

    gold = [gold_by_id[rid] for rid in expected]
    pred = [pred_by_id.get(rid, {}).get("pred_breaks", []) for rid in expected]
    n_words = [
        pred_by_id[rid]["word_count"]
        if rid in pred_by_id and "word_count" in pred_by_id[rid]
        else gold_wc.get(rid, 0)
        for rid in expected
    ]

    print(
        f"\n{len(expected)} documents, {sum(len(g) for g in gold)} gold boundaries, "
        f"{sum(len(p) for p in pred)} predicted\n"
    )
    print(f"{'tolerance':>10}{'tp':>5}{'fn':>5}{'fp':>5}{'P':>8}{'R':>8}{'F1':>9}")
    print("-" * 50)
    for tol in TOLERANCES:
        s = score_documents(gold, pred, tol)
        print(
            f"{f'+/-{tol} w':>10}{s.tp:>5}{s.fn:>5}{s.fp:>5}"
            f"{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>9.4f}"
        )

    wds = [
        pk_and_windowdiff(g, p, n) for g, p, n in zip(gold, pred, n_words, strict=True)
    ]
    usable = [(a, b) for a, b in wds if not math.isnan(a)]
    if usable:
        print(
            f"\nmean Pk {sum(a for a, _ in usable) / len(usable):.4f}   "
            f"mean WindowDiff {sum(b for _, b in usable) / len(usable):.4f}"
        )

    print(
        "\nPrecision is a LOWER BOUND: many scored false "
        "positives\nare real topic changes grouped into one thematic block. Do not "
        "compute a\nderived statistic that treats a false positive as clean error."
    )
    return 0


def cmd_segments(args: argparse.Namespace) -> int:
    """Expand transcripts + predicted boundaries into one row per story.

    Args:
        args: Parsed arguments.

    Returns:
        0.
    """
    bodies = {d["record_id"]: d for d in _read_jsonl(args.transcripts)}
    preds = _read_jsonl(args.predictions)

    missing = [p["record_id"] for p in preds if p["record_id"] not in bodies]
    if missing:
        sys.exit(f"{len(missing)} prediction(s) have no transcript, e.g. {missing[:3]}")

    # Every offset indexes body.split(). A prediction made against a different
    # revision of the transcript still joins on record_id and still produces
    # segments -- just silently mis-cut ones, with every story starting in the
    # wrong place. The schemas call this the single most damaging silent error
    # in the pipeline; until now only validate_data.py checked it, and only for
    # annotations.
    # A transcript with a missing or non-string body is a per-RECORD problem,
    # handled in the loop below so the rest of the corpus still gets written.
    # Dereferencing it here would abort the whole run with nothing written --
    # which is exactly the failure the per-record handler exists to prevent,
    # reintroduced one guard earlier.
    drift = [
        f"{p['record_id']}: prediction says {p['word_count']} words, "
        f"transcript has {len(bodies[p['record_id']]['body'].split())}"
        for p in preds
        if "word_count" in p
        and isinstance(bodies[p["record_id"]].get("body"), str)
        and p["word_count"] != len(bodies[p["record_id"]]["body"].split())
    ]
    if drift:
        sys.exit(
            f"{len(drift)} prediction(s) were made against different text:\n  "
            + "\n  ".join(drift[:10])
            + "\nEvery offset would be shifted. Re-run `breakingnews run` "
            "against these transcripts."
        )

    n_seg = n_flagged = 0
    failures: list[str] = []
    carry = ("outlet", "show", "date")
    minimal = ("record_id", "segment_id", "text", "n_cuts")
    with args.out.open("w", encoding="utf-8") as f:
        for p in preds:
            rec = bodies[p["record_id"]]
            if not isinstance(rec.get("body"), str):
                failures.append(
                    f"{p['record_id']}: transcript has no usable `body` "
                    f"(got {type(rec.get('body')).__name__})"
                )
                continue
            # One unusable record must not kill a corpus run and leave a file
            # that is short but perfectly valid JSON -- the failure mode that
            # looks like success.
            try:
                segs = to_segments(
                    p["record_id"],
                    rec["body"],
                    p["pred_breaks"],
                    min_words=args.min_words,
                    include_text=not args.no_text,
                )
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                # ValueError alone was too narrow: JSON Schema accepts 50.0 as
                # an integer, and a float offset raises TypeError deep in the
                # slicing, killing the corpus run instead of skipping one record.
                failures.append(f"{p['record_id']}: {type(exc).__name__}: {exc}")
                continue
            for seg in segs:
                row = seg.to_dict(include_text=not args.no_text)
                row.update({k: rec[k] for k in carry if k in rec})
                if args.minimal:
                    row = {k: row[k] for k in minimal if k in row}
                f.write(json.dumps(row) + "\n")
            n_seg += len(segs)
            n_flagged += sum(1 for s in segs if s.flags)

    ok = len(preds) - len(failures)
    print(f"{ok} of {len(preds)} records -> {n_seg} segments  ({args.out})")
    if args.min_words:
        print(
            f"{n_flagged} flagged below {args.min_words} words. Flagged, not "
            "dropped -- filter downstream if that is what you want."
        )
    if args.no_text:
        print("text omitted: offsets only, so `merge` cannot rebuild from these")
    if args.minimal:
        print(
            "minimal columns: record_id, segment_id, text, n_cuts. Offsets are "
            "dropped,\nso `merge` and `reconcile` cannot read this file -- keep a "
            "full copy if you\nwill need either."
        )
    if failures:
        print(
            f"\n{len(failures)} record(s) produced no segments and are ABSENT "
            f"from {args.out}:"
        )
        for msg in failures[:10]:
            print(f"  - {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        return 1
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Reassemble segments back into whole documents.

    The inverse of `segments`, and exact: the rebuilt body is byte-identical to
    the text the segments were cut from.

    Args:
        args: Parsed arguments.

    Returns:
        0 if every record rebuilt, 1 if any failed.
    """
    from .segments import Segment

    rows = _read_jsonl(args.segments)
    try:
        segs = [Segment.from_dict(r) for r in rows]
    except ValueError as exc:
        sys.exit(str(exc))

    # Record the input universe BEFORE dropping. Without this, a record whose
    # segments are ALL flagged simply has no group after the drop, so it never
    # reaches the loop below, never lands in `failures`, and vanishes from the
    # output with exit 0. The bias is not random: a short single-story broadcast
    # is ONE segment, so a single `short` flag deletes the whole record, and
    # those are precisely the records most likely to be flagged.
    records_in = set(group_by_record(segs))

    if args.drop_flagged:
        kept, dropped = drop_flagged(segs)
        print(f"dropped {len(dropped)} flagged segments; the rest is not a cover")
        segs = kept

    vanished = sorted(records_in - set(group_by_record(segs)))

    failures = []
    with args.out.open("w", encoding="utf-8") as f:
        for rid, group in sorted(group_by_record(segs).items()):
            try:
                record_id, body = merge_segments(group)
            except ValueError as exc:
                failures.append(f"{rid}: {exc}")
                continue
            f.write(
                json.dumps(
                    {
                        "record_id": record_id,
                        "word_count": len(body.split()),
                        "body": body,
                    }
                )
                + "\n"
            )

    print(
        f"{len(rows)} segments -> {len(rows) and len(group_by_record(segs))} "
        f"records ({args.out}); {len(records_in)} record(s) in"
    )
    if vanished:
        print(
            f"\n{len(vanished)} record(s) had EVERY segment dropped and are "
            f"ABSENT from {args.out}:"
        )
        for rid in vanished[:10]:
            print(f"  - {rid}")
        if len(vanished) > 10:
            print(f"  ... and {len(vanished) - 10} more")
        print(
            "  These are whole records, not segments. Short single-story "
            "broadcasts are\n  the most likely to be lost this way, so the "
            "survivors are a biased sample."
        )
    if failures:
        print(f"\n{len(failures)} record(s) could not be rebuilt:")
        for msg in failures[:10]:
            print(f"  - {msg}")
    return 1 if (failures or vanished) else 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    """Map segment ids from one run onto another.

    `segment_id` is an ordinal, so joining two runs on it is unsafe: one extra
    boundary renumbers everything after it. This pairs segments by shared text.

    Args:
        args: Parsed arguments.

    Returns:
        0.
    """
    from .segments import Segment

    def load(path: Path) -> list[Segment]:
        return [Segment.from_dict(r) for r in _read_jsonl(path)]

    old, new = load(args.old), load(args.new)
    pairs = reconcile(old, new, min_overlap=args.min_overlap)

    counts: dict[str, int] = {}
    for c in pairs:
        counts[c.status] = counts.get(c.status, 0) + 1
    print(f"\n{len(old)} old segments, {len(new)} new\n")
    for status in ("same", "moved", "split", "merged", "added", "removed"):
        if counts.get(status):
            print(f"  {status:>8}  {counts[status]}")

    # Only one-to-one pairings: on a split or merged row `start_shift` is the
    # distance from the host segment's start, not how far a boundary moved, so
    # pooling them would report a meaningless maximum.
    shifts = [
        abs(c.start_shift)
        for c in pairs
        if c.start_shift and c.status in STABLE_STATUSES
    ]
    if shifts:
        print(
            f"\nboundary shifts among one-to-one pairs: {len(shifts)} nonzero, "
            f"max {max(shifts)} words"
        )

    mapping = id_map(pairs)
    unstable = len(old) - len(mapping)
    print(
        f"\n{len(mapping)} of {len(old)} old ids map one-to-one; {unstable} do not. "
        "split/merged\nsegments are excluded on purpose -- they have no single "
        "successor, and picking\none would quietly corrupt any result carried "
        "across the two runs."
    )

    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            for c in pairs:
                f.write(json.dumps(c.__dict__) + "\n")
        print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The parser, with one subcommand per verb.
    """
    ap = argparse.ArgumentParser(prog="breakingnews", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("adapter", help="local adapter directory or Hub repo id")
        p.add_argument("--base-model", default=None, dest="base_model")
        p.add_argument(
            "--revision",
            default=None,
            help="pin the adapter to a Hub tag or commit; unpinned means main, "
            "which moves",
        )
        p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
        p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("check-adapter", help="check an adapter for silent failures")
    p.add_argument("adapter")
    p.add_argument("--revision", default=None)
    p.set_defaults(func=cmd_check_adapter)

    p = sub.add_parser("run", help="segment a corpus")
    add_model_args(p)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--chunk", type=int, default=64)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="score every threshold from one pass")
    add_model_args(p)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE_WORDS)
    p.add_argument(
        "--thresholds",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0],
    )
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("segments", help="expand predictions into one row per story")
    p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--min-words",
        type=int,
        default=None,
        dest="min_words",
        help="flag (never drop) segments shorter than this",
    )
    p.add_argument(
        "--no-text",
        action="store_true",
        dest="no_text",
        help="emit offsets only, for a corpus whose text cannot be redistributed",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="emit only record_id, segment_id, text and n_cuts; drops the "
        "offsets that `merge` and `reconcile` need",
    )
    p.set_defaults(func=cmd_segments)

    p = sub.add_parser("merge", help="reassemble segments into whole documents")
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--drop-flagged",
        action="store_true",
        dest="drop_flagged",
        help="discard flagged segments first; the merge will then refuse the gaps",
    )
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("reconcile", help="map segment ids between two runs")
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--min-overlap",
        type=float,
        default=0.5,
        dest="min_overlap",
        help="shared-text fraction required to call two segments the same story",
    )
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("score", help="score predictions against gold (no GPU)")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument(
        "--allow-missing",
        action="store_true",
        dest="allow_missing",
        help="score a partial prediction file anyway; absent gold documents are "
        "still counted as all-misses, and the error is still printed",
    )
    p.set_defaults(func=cmd_score)

    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to `sys.argv[1:]`.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
