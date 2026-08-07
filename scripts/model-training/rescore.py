"""
Rescore existing predictions at several match tolerances.

  python rescore.py --predictions build/pred_V4_hard_tau0.01_val.jsonl --split val

A prediction counts as a hit if it lands within N words of a gold break. N=25 is
the project default and is deliberately strict. Widening it does NOT change the
model or the training - it changes the question being asked:

    +-25   "did it find the boundary to within a sentence?"
    +-100  "did it find the boundary at all?"

Which is right depends on the downstream use. For slicing a transcript into
stories, a 40-word offset is immaterial; for aligning to a timecode it is not.

The mean offset on MATCHED breaks is reported alongside, because it is what
tells you whether a wider tolerance is admitting near-misses or slop. If mean
offset stays far below the tolerance, the extra matches are genuinely close.
"""
import argparse
import json

import config as C
import build_dataset as B
import evaluate as E


def score(preds, gold, tol, expected):
    """
    Iterate the EXPECTED document set, not the predictions.

    The previous version looped over `preds` and skipped any record with no
    gold. That silently dropped the OPPOSITE case - a gold document with no
    prediction row - out of the denominator entirely, so its misses were never
    counted. Losing half a prediction file moved F1 from 0.667 to 1.000, exit
    code 0, no warning, and the error direction is always favourable. Worse,
    the function DID warn about predictions lacking gold, so a reader could
    reasonably infer the unwarned direction was clean.

    A document absent from `preds` now contributes its full gold count as false
    negatives, which is what "the model predicted nothing here" means.
    """
    by_id = {p["record_id"]: p for p in preds}
    tp = fn = fp = 0
    offs, pks, wds = [], [], []
    for rid in expected:
        g = gold[rid]
        p = by_id.get(rid)
        pr = sorted(p["pred_breaks"]) if p else []
        m, miss, extra = E.match(g, pr, tol)
        tp += len(m); fn += miss; fp += extra
        offs += [x[2] for x in m]
        if p is not None:
            pk, wd = E.pk_and_windowdiff(g, pr, p["word_count"])
            if pk is not None:
                pks.append(pk); wds.append(wd)
    P, R, F = E.prf(tp, fn, fp)
    return dict(tol=tol, tp=tp, fn=fn, fp=fp, P=P, R=R, F1=F,
                mean_off=(sum(offs) / len(offs) if offs else 0.0),
                pk=(sum(pks) / len(pks) if pks else float("nan")),
                wd=(sum(wds) / len(wds) if wds else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tolerances", default="25,50,100")
    ap.add_argument("--label", default=None)
    ap.add_argument("--allow_missing", action="store_true",
                    help="score anyway when the prediction file is missing "
                         "documents from the split, counting them as zero "
                         "predictions. Default is to refuse (exit 2).")
    args = ap.parse_args()

    transcripts, ann = B.load_sources()
    ann = B.validate(transcripts, ann)
    gold = {a["record_id"]: sorted(a["breaks"]) for a in ann}
    preds = B.load_jsonl(args.predictions)

    # The split defines the denominator, NOT the prediction file. Anything else
    # lets a truncated prediction file quietly shrink what is being scored.
    expected = sorted({r["record_id"] for r in
                       B.load_jsonl(C.BUILD_DIR / f"dataset_{args.split}.jsonl")}
                      & set(gold))
    have = {p["record_id"] for p in preds}
    missing = [r for r in expected if r not in have]
    orphan = [p["record_id"] for p in preds if p["record_id"] not in gold]
    if orphan:
        print(f"  [warn] {len(orphan)} predictions have no gold; ignored")
    if missing:
        lost = sum(len(gold[r]) for r in missing)
        print(f"  [ERROR] {len(missing)} of {len(expected)} {args.split} documents "
              f"are ABSENT from the prediction file, holding {lost} gold breaks.")
        print(f"          They are being scored as zero predictions (all misses). "
              f"If that is not what you\n          intend, the prediction file is "
              f"incomplete and this number is not comparable.")
        if not args.allow_missing:
            raise SystemExit(2)

    n_gold = sum(len(gold[r]) for r in expected)
    print(f"\n{args.label or args.predictions}  —  {args.split}: "
          f"{len(expected)} documents, {n_gold} gold breaks, "
          f"{sum(len(p['pred_breaks']) for p in preds)} predicted")
    print(f"  {'tol':>5}{'tp':>6}{'fn':>6}{'fp':>6}{'P':>8}{'R':>8}{'F1':>9}"
          f"{'Pk':>8}{'WD':>8}{'mean off':>10}")

    rows = []
    for t in [int(x) for x in args.tolerances.split(",")]:
        r = score(preds, gold, t, expected)
        rows.append(r)
        print(f"  {r['tol']:>5}{r['tp']:>6}{r['fn']:>6}{r['fp']:>6}{r['P']:>8.3f}"
              f"{r['R']:>8.3f}{r['F1']:>9.4f}{r['pk']:>8.3f}{r['wd']:>8.3f}"
              f"{r['mean_off']:>9.1f}w")

    base = rows[0]
    for r in rows[1:]:
        print(f"  +-{r['tol']} vs +-{base['tol']}: F1 {r['F1']-base['F1']:+.4f}, "
              f"{base['fp']-r['fp']} fewer false positives, "
              f"{base['fn']-r['fn']} fewer misses "
              f"(mean offset {base['mean_off']:.1f}w -> {r['mean_off']:.1f}w)")

    out = args.predictions.replace(".jsonl", "_rescore.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
