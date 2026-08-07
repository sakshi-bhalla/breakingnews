"""
Stage 4 — score predicted breaks against the gold annotations.

  python evaluate.py --predictions build/predictions_test.jsonl --split test

Reports three families, because none alone is sufficient:

  1. Tolerance P/R/F1 — a prediction counts if it lands within
     MATCH_TOLERANCE_WORDS of a gold break, matched one-to-one.  Directly
     interpretable, but binary: a 26-word miss scores the same as a 2,000-word
     miss.

  2. Pk and WindowDiff — the standard text-segmentation metrics (Beeferman et
     al. 1999; Pevzner & Hearst 2002).  Both are ERROR rates in [0,1], lower is
     better, and both degrade gracefully with near-misses.  WindowDiff fixes
     Pk's known insensitivity to false positives inside a segment, so trust it
     over Pk where they disagree.

  3. Two null baselines, scored identically.  A segmenter must beat both to
     have earned anything: predicting NO breaks is strong whenever segments are
     long, and predicting breaks at a UNIFORM spacing is strong whenever
     segment lengths are regular.  Reporting F1 without these is how
     segmentation papers accidentally publish noise.
"""
import argparse
import json
from collections import defaultdict

import config as C
from build_dataset import load_jsonl, load_sources, validate


# ── Matching ───────────────────────────────────────────────────────────────────

def match(gold, pred, tolerance):
    """
    One-to-one greedy match by increasing distance: each gold break may absorb
    at most one prediction and vice versa, so duplicate predictions clustered on
    one true boundary are counted as false positives rather than free credit.
    """
    pairs = sorted(
        ((abs(g - p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)
         if abs(g - p) <= tolerance)
    )
    used_g, used_p, matched = set(), set(), []
    for dist, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gold[gi], pred[pi], dist))
    return matched, len(gold) - len(used_g), len(pred) - len(used_p)


def prf(tp, fn, fp):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ── Pk / WindowDiff ────────────────────────────────────────────────────────────

def _seg_id(breaks, n_words):
    """Segment index for every word position, from a sorted break list."""
    ids, seg, bi = [0] * n_words, 0, 0
    bs = sorted(breaks)
    for i in range(n_words):
        while bi < len(bs) and bs[bi] == i:
            seg += 1
            bi += 1
        ids[i] = seg
    return ids


def pk_and_windowdiff(gold, pred, n_words, k=None):
    """
    k defaults to half the mean gold segment length, the convention from
    Beeferman et al.  Returns (Pk, WindowDiff); both are error rates.
    """
    if n_words < 4:
        return None, None
    if k is None:
        mean_seg = n_words / (len(gold) + 1)
        k = max(2, int(round(mean_seg / 2)))
    k = min(k, n_words - 1)

    g_id, p_id = _seg_id(gold, n_words), _seg_id(pred, n_words)
    gs, ps = sorted(gold), sorted(pred)

    n_probe = n_words - k
    if n_probe <= 0:
        return None, None

    pk_err = wd_err = 0
    for i in range(n_probe):
        j = i + k
        if (g_id[i] == g_id[j]) != (p_id[i] == p_id[j]):
            pk_err += 1
        gb = sum(1 for b in gs if i < b <= j)
        pb = sum(1 for b in ps if i < b <= j)
        if gb != pb:
            wd_err += 1
    return pk_err / n_probe, wd_err / n_probe


# ── Baselines ──────────────────────────────────────────────────────────────────

def baseline_none(_gold, _n_words):
    return []


def baseline_uniform(gold, n_words):
    """Same NUMBER of breaks as gold, placed at equal spacing. Tests whether the
    model is doing more than counting."""
    n = len(gold)
    if n == 0:
        return []
    step = n_words / (n + 1)
    return [int(round(step * (i + 1))) for i in range(n)]


# ── Scoring one system ─────────────────────────────────────────────────────────

def score(docs, predict_fn, tolerance):
    tp = fn = fp = 0
    pks, wds, per_doc = [], [], []
    for d in docs:
        gold, n_words = d["gold"], d["word_count"]
        pred = predict_fn(gold, n_words) if predict_fn else d["pred"]
        m, miss, extra = match(gold, pred, tolerance)
        tp += len(m); fn += miss; fp += extra
        pk, wd = pk_and_windowdiff(gold, pred, n_words)
        if pk is not None:
            pks.append(pk); wds.append(wd)
        per_doc.append({"record_id": d["record_id"], "n_gold": len(gold),
                        "n_pred": len(pred), "tp": len(m), "fn": miss, "fp": extra,
                        "pk": pk, "wd": wd,
                        "mean_offset": (sum(x[2] for x in m) / len(m)) if m else None})
    p, r, f = prf(tp, fn, fp)
    return {
        "tp": tp, "fn": fn, "fp": fp,
        "precision": p, "recall": r, "f1": f,
        "pk": sum(pks) / len(pks) if pks else None,
        "windowdiff": sum(wds) / len(wds) if wds else None,
        "per_doc": per_doc,
    }


def fmt(name, s):
    pk = f"{s['pk']:.3f}" if s["pk"] is not None else "  -  "
    wd = f"{s['windowdiff']:.3f}" if s["windowdiff"] is not None else "  -  "
    return (f"  {name:<22} P {s['precision']:.3f}  R {s['recall']:.3f}  "
            f"F1 {s['f1']:.3f}   Pk {pk}  WD {wd}   "
            f"(tp {s['tp']} fn {s['fn']} fp {s['fp']})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tolerance", type=int, default=C.MATCH_TOLERANCE_WORDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    transcripts, annotations = load_sources()
    annotations = validate(transcripts, annotations)
    gold_by_id = {a["record_id"]: a for a in annotations}

    preds = load_jsonl(args.predictions)
    by_id = {p["record_id"]: p for p in preds}

    # The SPLIT defines what is being scored, not the prediction file. Iterating
    # `preds` instead drops any gold document that produced no prediction row
    # straight out of the denominator, so its misses are never counted - and the
    # error direction is always favourable. Losing half a prediction file took
    # F1 from 0.667 to 1.000 with exit 0 and no warning. The asymmetry made it
    # worse: the reverse case (a prediction with no gold) DID warn, so silence
    # in this direction read as clean.
    expected = sorted({r["record_id"] for r in
                       load_jsonl(C.BUILD_DIR / f"dataset_{args.split}.jsonl")}
                      & set(gold_by_id))
    docs, absent = [], []
    for rid in expected:
        a = gold_by_id[rid]
        p = by_id.get(rid)
        if p is None:
            absent.append(rid)
        docs.append({
            "record_id": rid,
            "outlet": (p.get("outlet") if p else None) or a["outlet"],
            "gold": sorted(a["breaks"]),
            "pred": sorted(p["pred_breaks"]) if p else [],
            "word_count": a["word_count"],
        })

    orphan = [r for r in by_id if r not in gold_by_id]
    if orphan:
        print(f"[WARN] {len(orphan)} predicted docs have no gold annotation; "
              f"excluded from scoring.")
    if absent:
        lost = sum(len(gold_by_id[r]["breaks"]) for r in absent)
        print(f"[ERROR] {len(absent)} of {len(expected)} {args.split} documents "
              f"are ABSENT from the prediction file,\n        holding {lost} gold "
              f"breaks. They are scored as zero predictions (all misses).\n"
              f"        If the prediction file is simply incomplete, this number "
              f"is NOT comparable.")
        raise SystemExit(2)

    n_gold = sum(len(d["gold"]) for d in docs)
    n_pred = sum(len(d["pred"]) for d in docs)
    unloc = sum(p.get("n_unlocatable_anchors", 0) for p in preds)

    print("\n" + "=" * 78)
    print(f"SEGMENTATION EVAL — split={args.split}  tolerance=±{args.tolerance} words")
    print("=" * 78)
    print(f"documents {len(docs)}   gold breaks {n_gold}   predicted breaks {n_pred}")
    if unloc:
        print(f"[WARN] {unloc} generated anchors could not be grounded in their "
              f"window\n       (model paraphrased instead of quoting) — these are "
              f"lost recall, not errors.")

    model_s = score(docs, None, args.tolerance)
    none_s = score(docs, baseline_none, args.tolerance)
    unif_s = score(docs, baseline_uniform, args.tolerance)

    print("\nlower Pk/WD is better; higher F1 is better")
    print(fmt("MODEL", model_s))
    print(fmt("baseline: no breaks", none_s))
    print(fmt("baseline: uniform", unif_s))

    beats = (model_s["f1"] > unif_s["f1"] and
             (model_s["windowdiff"] or 1) < (unif_s["windowdiff"] or 1) and
             (model_s["windowdiff"] or 1) < (none_s["windowdiff"] or 1))
    print(f"\n  -> model beats both nulls on F1 and WindowDiff: "
          f"{'YES' if beats else 'NO — do not ship this adapter'}")

    by_outlet = defaultdict(list)
    for d in docs:
        by_outlet[d["outlet"]].append(d)
    if len(by_outlet) > 1:
        print("\nby outlet")
        for o, ds in sorted(by_outlet.items(), key=lambda kv: -len(kv[1])):
            print(fmt(o, score(ds, None, args.tolerance)))
        print("  NOTE: only CNN and FOX exist in the target corpus; the others "
              "are\n        train-only outlets and their scores do not transfer.")

    hits = [x for d in model_s["per_doc"] if d["mean_offset"] is not None
            for x in [d["mean_offset"]]]
    if hits:
        print(f"\nmean |predicted - gold| on matched breaks: "
              f"{sum(hits)/len(hits):.1f} words")

    out = args.out or str(C.BUILD_DIR / f"eval_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"split": args.split, "tolerance": args.tolerance,
                   "model": model_s, "baseline_none": none_s,
                   "baseline_uniform": unif_s}, f, indent=2)
    print(f"\nwrote {out}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
