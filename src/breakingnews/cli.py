"""Command line interface.

    breakingnews check-adapter ADAPTER
    breakingnews run ADAPTER --input transcripts.jsonl --out preds.jsonl
    breakingnews sweep ADAPTER --input transcripts.jsonl --gold annotations.jsonl
    breakingnews score --predictions preds.jsonl --gold annotations.jsonl
    breakingnews segments --transcripts t.jsonl --predictions p.jsonl --out s.jsonl
    breakingnews merge --segments s.jsonl --out rebuilt.jsonl

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
from .segments import drop_flagged, group_by_record, merge_segments, to_segments

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
    try:
        from .segmenter import Segmenter
    except ImportError as exc:  # pragma: no cover - depends on install extras
        sys.exit(
            f"this command needs the inference extra: pip install "
            f"'breakingnews[gpu]'  ({exc})"
        )
    return Segmenter.from_pretrained(
        args.adapter,
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
    adapter_dir = resolve_adapter(args.adapter)
    problems = verify_adapter(adapter_dir)
    print(f"adapter: {adapter_dir}")
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
    with args.out.open("w", encoding="utf-8") as f:
        # Chunked so a large corpus does not build one giant window pool.
        for start in range(0, len(docs), args.chunk):
            chunk = docs[start : start + args.chunk]
            scored = seg.score_windows([d["body"] for d in chunk])
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
    print(f"wrote {args.out}")
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
        sys.exit("no documents in --input have gold annotations in --gold")

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
        "A wide plateau is not a tuning opportunity -- see LIMITATIONS.md C7."
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
        0.
    """
    preds = _read_jsonl(args.predictions)
    gold_by_id = {a["record_id"]: sorted(a["breaks"]) for a in _read_jsonl(args.gold)}

    paired = [
        (p, gold_by_id[p["record_id"]]) for p in preds if p["record_id"] in gold_by_id
    ]
    if not paired:
        sys.exit("no record_id in --predictions appears in --gold")
    if len(paired) < len(preds):
        print(f"note: {len(preds) - len(paired)} predictions have no gold; skipped")

    pred = [p["pred_breaks"] for p, _ in paired]
    gold = [g for _, g in paired]
    n_words = [p["word_count"] for p, _ in paired]

    print(
        f"\n{len(paired)} documents, {sum(len(g) for g in gold)} gold boundaries, "
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
        "\nPrecision is a LOWER BOUND (LIMITATIONS.md C1): many scored false "
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

    n_seg = n_flagged = 0
    carry = ("outlet", "show", "date")
    with args.out.open("w", encoding="utf-8") as f:
        for p in preds:
            rec = bodies[p["record_id"]]
            segs = to_segments(
                p["record_id"],
                rec["body"],
                p["pred_breaks"],
                min_words=args.min_words,
                include_text=not args.no_text,
            )
            for seg in segs:
                row = seg.to_dict(include_text=not args.no_text)
                row.update({k: rec[k] for k in carry if k in rec})
                f.write(json.dumps(row) + "\n")
            n_seg += len(segs)
            n_flagged += sum(1 for s in segs if s.flags)

    print(f"{len(preds)} records -> {n_seg} segments  ({args.out})")
    if args.min_words:
        print(
            f"{n_flagged} flagged below {args.min_words} words. Flagged, not "
            "dropped -- filter downstream if that is what you want."
        )
    if args.no_text:
        print("text omitted: offsets only, so `merge` cannot rebuild from these")
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
    segs = [
        Segment(
            segment_id=r["segment_id"],
            record_id=r["record_id"],
            index=r["index"],
            word_start=r["word_start"],
            word_end=r["word_end"],
            char_start=r["char_start"],
            char_end=r["char_end"],
            n_words=r["n_words"],
            text=r.get("text", ""),
            flags=tuple(r.get("flags", ())),
        )
        for r in rows
    ]

    if args.drop_flagged:
        kept, dropped = drop_flagged(segs)
        print(f"dropped {len(dropped)} flagged segments; the rest is not a cover")
        segs = kept

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
        f"records ({args.out})"
    )
    if failures:
        print(f"\n{len(failures)} record(s) could not be rebuilt:")
        for msg in failures[:10]:
            print(f"  - {msg}")
        return 1
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
        p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
        p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("check-adapter", help="check an adapter for silent failures")
    p.add_argument("adapter")
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

    p = sub.add_parser("score", help="score predictions against gold (no GPU)")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
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
